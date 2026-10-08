#!/usr/bin/env python3
"""Emit every HAL the Loom single-token decode forward needs for a shard.

usage: emit_decode.py <model.gguf> <outdir> [max_context]       (default 4096, multiple of 256)

  gb_<fmts>_<Ms>_<K>.hal         tools/gen_gemv.py gen_bands: a layer's input projections in one dispatch (the decoder's default)
  gv_<kind>_<fmts>_<M>_<K>.hal   tools/gen_gemv.py, one per distinct (kind, formats, shape) on the shard:
                                 plain for the input projections and the head,
                                 resid for attn_output / ssm_out / ffn_down (residual add fused),
                                 swiglu for ffn_gate + ffn_up
  dattn_{kvappend,part,reduce}   tools/gen_decode_attn.py: attention over the paged fp16 KV pools of the prefill
  dattn_{kappend,vappend,part}_q only with a quantized YAH_KV (kv8 | kv4 | k8v4 | k4v8, same as the prefill set):
                                 the kv8a16 / kv4a16 pools; decode.txt gets "kv q KB VB"
  rmsnorm, deltanet_conv, embed  tools/gen_decode_misc.py (deltanet_conv also runs the decode conv)
  unpack, rope, argmax           hand-written kernels in loom/*.loom.
                                 rope's own cache write goes to a one-row dummy (max_context=1); dattn_kvappend writes the pools.
  grid_*.bin, ksigns_iq2xs.bin   the IQ tables (loom/tables)
  decode.txt                     "ctx <max_context>", GEMV geometry and grids
"""
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LOOM = os.path.abspath(os.path.join(HERE, ".."))
EMIT = os.path.join(LOOM, "emit_hal.py")
sys.path.insert(0, HERE)
import emit_prefill as EP  # noqa: E402  (GGUF tensor table)
import gen_decode_attn as DA  # noqa: E402
import gen_decode_misc as DM  # noqa: E402
import gen_gemv as GV  # noqa: E402
import gen_kvq  # noqa: E402

NUM_HEADS, NUM_KV, HEAD_DIM, ROTARY = 24, 4, 256, 64
# rows per wave R and waves per workgroup W per GEMV kind; recorded in decode.txt ("rw kind R W")
RW = {k: (2, 4) for k in ("plain", "resid", "swiglu", "bands")}
# The tuner's table (tune_table.py) may override them: {"rw": {"bands": [4, 4], ...}}. Every row is still reduced over
# the 32 lanes of one wave, so R and W change no result bits.


def emit_src(text, name, outdir, configs=("nop=0",)):
    tmp = os.path.join(outdir, ".emit_tmp")
    os.makedirs(tmp, exist_ok=True)
    src = os.path.join(tmp, name + ".loom")
    open(src, "w").write(text)
    emit_file(src, name, outdir, configs)


def emit_file(path, name, outdir, configs):
    tmp = os.path.join(outdir, ".emit_tmp")
    os.makedirs(tmp, exist_ok=True)
    r = subprocess.run([sys.executable, EMIT, path, tmp] + list(configs), capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("emit failed for %s:\n%s%s" % (path, r.stdout[-3000:], r.stderr[-3000:]))
    shutil.copy(r.stdout.strip().splitlines()[0], os.path.join(outdir, name + ".hal"))


def kv_bits():
    """(K bits, V bits) of YAH_KV; must match the prefill set.

    The decode attention has no mixed fp16 / quantized form, so K and V are both quantized or both fp16.
    """
    kb, vb = gen_kvq.kv_bits()
    if (kb == 16) != (vb == 16):
        raise SystemExit("decode: YAH_KV must quantize both K and V (kv8, kv4, k8v4, k4v8) or neither")
    return kb, vb


def gemv_name(kind, fmts, M, K):
    return "gv_%s_%s_%d_%d" % (kind, "_".join(fmts), M, K)


def bands_name(fmts, Ms, K):
    return "gb_%s_%s_%d" % ("_".join(fmts), "_".join(map(str, Ms)), K)


def bands_set(model):
    """{name: (fmts, Ms, K)}: each layer's input projections as one band-fused GEMV.

    The bands are attn_qkv / attn_gate / ssm_alpha / ssm_beta (DeltaNet layers) or attn_q / attn_k / attn_v (attention layers).
    """
    t = {}
    for nm, dims, ty in EP.parse(model):
        t[nm] = (GV.GGML.get(ty), int(dims[0]), int(dims[1]) if len(dims) > 1 else 1)
    out = {}
    layers = sorted({int(n.split(".")[1]) for n in t if n.startswith("blk.") and n.split(".")[1].isdigit()})
    for l in layers:
        p = "blk.%d." % l
        if p + "ffn_gate.weight" not in t:
            continue
        group = ("attn_qkv", "attn_gate", "ssm_alpha", "ssm_beta") if p + "attn_qkv.weight" in t else ("attn_q", "attn_k", "attn_v")
        ns = [p + n + ".weight" for n in group]
        fmts, K, Ms = [t[n][0] for n in ns], t[ns[0]][1], [t[n][2] for n in ns]
        out[bands_name(fmts, Ms, K)] = (fmts, Ms, K)
    return out


def gemv_set(model):
    """{name: (kind, fmts, M, K)} for every decode projection on the shard."""
    t = {}
    for nm, dims, ty in EP.parse(model):
        t[nm] = (GV.GGML.get(ty), int(dims[0]), int(dims[1]) if len(dims) > 1 else 1)
    out = {}

    def add(kind, names):
        fmts = [t[n][0] for n in names]
        if None in fmts:
            raise SystemExit("no GEMV decoder for " + ", ".join(names))
        K, M = t[names[0]][1], t[names[0]][2]
        out[gemv_name(kind, fmts, M, K)] = (kind, fmts, M, K)

    layers = sorted({int(n.split(".")[1]) for n in t if n.startswith("blk.") and n.split(".")[1].isdigit()})
    for l in layers:
        p = "blk.%d." % l
        if p + "ffn_gate.weight" not in t:
            continue
        for n in ("attn_q", "attn_k", "attn_v", "attn_qkv", "attn_gate", "ssm_alpha", "ssm_beta"):
            if p + n + ".weight" in t:
                add("plain", [p + n + ".weight"])
        for n in ("attn_output", "ssm_out", "ffn_down"):
            if p + n + ".weight" in t:
                add("resid", [p + n + ".weight"])
        add("swiglu", [p + "ffn_gate.weight", p + "ffn_up.weight"])
    add("plain", ["output.weight"])
    return out


def main():
    model, outdir = sys.argv[1], sys.argv[2]
    import tune_table
    RW.update({k: tuple(v) for k, v in tune_table.load(model).get("rw", {}).items()})
    T = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
    if T % 256:
        raise SystemExit("max_context must be a multiple of 256")
    os.makedirs(outdir, exist_ok=True)
    gs = gemv_set(model)
    # Exact launch grid of every GEMV kernel; the decoder refuses any other.
    # Loom may drop row clamps it proves redundant for this grid, so a larger grid reads out of bounds and can hang the GPU.
    grids = {}
    for name, (kind, fmts, M, K) in sorted(gs.items()):
        R, W = RW[kind]
        emit_src(GV.gen(kind, fmts, M, K, R, W), name, outdir)
        grids[name] = M // GV.rows_per_wg(R, W)
    bs = bands_set(model)
    for name, (fmts, Ms, K) in sorted(bs.items()):
        emit_src(GV.gen_bands(fmts, Ms, K, *RW["bands"]), name, outdir)
        grids[name] = sum(Ms) // GV.rows_per_wg(*RW["bands"])
    for which in ("kvappend", "part", "reduce"):
        emit_src(DA.gen(which, T), "dattn_" + which, outdir)
    kb, vb = kv_bits()
    if kb != 16:
        emit_src(DA.gen_kappend_q(T, kb), "dattn_kappend_q", outdir)
        emit_src(DA.gen_vappend_q(T, vb), "dattn_vappend_q", outdir)
        emit_src(DA.gen_part_q(T, kb, vb), "dattn_part_q", outdir)
    L = lambda f: os.path.join(LOOM, f)
    emit_src(DM.gen_rmsnorm(), "rmsnorm", outdir)
    emit_file(L("yah_unpack_qg_f32.loom"), "unpack", outdir,
              ["yah_unpack_qg.batch=1", "yah_unpack_qg.num_heads=%d" % NUM_HEADS, "yah_unpack_qg.head_dim=%d" % HEAD_DIM])
    emit_file(L("yah_fused_qk_rope_f32.loom"), "rope", outdir, [
        "yah_fused_qk_rope.layer_idx=0", "yah_fused_qk_rope.max_context=1",
        "yah_fused_qk_rope.num_heads=%d" % NUM_HEADS, "yah_fused_qk_rope.num_kv_heads=%d" % NUM_KV,
        "yah_fused_qk_rope.head_dim=%d" % HEAD_DIM, "yah_fused_qk_rope.rotary_dim=%d" % ROTARY,
        "yah_fused_qk_rope.q_elems=%d" % (NUM_HEADS * HEAD_DIM), "yah_fused_qk_rope.kv_elems=%d" % (NUM_KV * HEAD_DIM),
        "yah_fused_qk_rope.cache32_elems=%d" % (NUM_KV * HEAD_DIM),
        "yah_fused_qk_rope.cache16_elems=%d" % (NUM_KV * HEAD_DIM)])
    emit_src(DM.gen_deltanet(), "deltanet_conv", outdir)   # also runs the decode conv
    emit_src(DM.gen_embed_iq4xs(), "embed", outdir)      # token_embd row from the device token stream
    emit_file(L("yah_argmax_f32.loom"), "argmax", outdir, ["yah_argmax.vocab=248320"])
    emit_src(DM.gen_sample(), "sample", outdir)          # temperature / top-p sampling into the token stream
    for f in os.listdir(os.path.join(LOOM, "tables")):
        shutil.copy(os.path.join(LOOM, "tables", f), os.path.join(outdir, f))
    open(os.path.join(outdir, "decode.txt"), "w").write(
        "ctx %d\n" % T + ("kv q %d %d\n" % (kb, vb) if kb != 16 else "") + "".join("rw %s %d %d\n" % (k, r, w) for k, (r, w) in sorted(RW.items())) +
        "".join("grid %s %d 0\n" % (n, g) for n, g in sorted(grids.items())))
    shutil.rmtree(os.path.join(outdir, ".emit_tmp"), ignore_errors=True)
    print("emitted %d GEMV + %d band GEMV + the decode HALs (max context %d) to %s" % (len(gs), len(bs), T, outdir))


if __name__ == "__main__":
    main()
