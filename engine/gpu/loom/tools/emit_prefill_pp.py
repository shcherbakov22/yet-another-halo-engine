#!/usr/bin/env python3
"""Emit the prefill HAL set for B-token chunks: every kernel compiled for its shape, plus dispatch.txt.

Every token dimension is bound to B: GEMM token_tiles = B / tile (grid y), fixed kernels batch = B, residual dim = 5120 * B.
YAH_CTX=T (a multiple of B, default B) sizes the KV cache and emits one rope / attention HAL per chunk.
The driver runs T / B chunks and carries the DeltaNet and conv states. YAH_KV selects the KV format (gen_kvq.kv_bits).
dispatch.txt has one row per HAL, "<hal> <tokens per workgroup> <row groups> <token_tiles>", plus mode marker rows.
HAL names follow emit_prefill.py.

usage: emit_prefill_pp.py <model.gguf> <outdir> [tokens]   (default 2048 tokens)
"""
import dataclasses, functools, json, os, re, subprocess, sys, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import emit_prefill as E  # noqa: E402
import gen_attn_fa  # noqa: E402
import gen_deltanet_hip  # noqa: E402
import gen_gdn_chunk  # noqa: E402
import gen_conv_kq  # noqa: E402
import gen_kres_persist  # noqa: E402
import gen_half_norm  # noqa: E402
import gen_kvq  # noqa: E402
import gen_npu_gemm as GN  # noqa: E402


def rope_kpaged(text):
    """Rewrite yah_fused_qk_rope_batched into the paged-K variant: K rows go straight to the paged K pool.
    K row = ptab[cur / 256] * 256 + cur % 256, page index clamped into the pool.
    V still writes the one-chunk scratch for yah_vtpage."""
    def r(a, b):
        nonlocal text
        assert text.count(a) == 1, a[:60]
        text = text.replace(a, b)
    r("config.decl @yah_fused_qk_rope_batched.cache16_elems : %value: index where [range(%value, 1, 1073741824)]",
      "config.decl @yah_fused_qk_rope_batched.cache16_elems : %value: index where [range(%value, 1, 1073741824)]\n"
      "config.decl @yah_fused_qk_rope_batched.k16_elems : %value: index where [range(%value, 1, 1073741824)]")
    r("%k_cache_f16: buffer, %v_cache_f16: buffer, %eps_buf: buffer) {",
      "%k_cache_f16: buffer, %v_cache_f16: buffer, %eps_buf: buffer, %ptab: buffer) {")
    r("  %cache16_elems = config.get @yah_fused_qk_rope_batched.cache16_elems : index\n",
      "  %cache16_elems = config.get @yah_fused_qk_rope_batched.cache16_elems : index\n"
      "  %k16_elems = config.get @yah_fused_qk_rope_batched.k16_elems : index\n")
    r("  %k16_view = buffer.view %k16_na[%base] : buffer -> view<[%cache16_elems]xf16>",
      "  %k16_view = buffer.view %k16_na[%base] : buffer -> view<[%k16_elems]xf16>")
    r("  %cur = index.min %cur_nn, %ctx_m1 : index\n",
      "  %cur = index.min %cur_nn, %ctx_m1 : index\n"
      "  %pg256 = index.constant 256 : index\n"
      "  %pg255 = index.constant 255 : index\n"
      "  %pgrows = index.div %k16_elems, %kv_width : index\n"
      "  %pgn0 = index.add %pgrows, %pg255 : index\n"
      "  %pgnp = index.div %pgn0, %pg256 : index\n"
      "  %pt_view = buffer.view %ptab[%base] : buffer -> view<[%pgnp]xi32>\n"
      "  %pglp = index.div %cur, %pg256 : index\n"
      "  %pglp1 = index.sub %pgnp, %c1 : index\n"
      "  %pglpc = index.min %pglp, %pglp1 : index\n"
      "  %pgv = view.load %pt_view[%pglpc] : view<[%pgnp]xi32> -> i32\n"
      "  %pgu = index.cast %pgv : i32 to index\n"
      "  %pgz = index.max %pgu, %c0 : index\n"
      "  %pg = index.min %pgz, %pglp1 : index\n"           # clamp into the pool (both sides)
      "  %pgb = index.mul %pg, %pg256 : index\n"
      "  %pgo = index.rem %cur, %pg256 : index\n"
      "  %krow = index.add %pgb, %pgo : index\n")
    r("    %f16_off = index.min %f16_off_raw, %cache16_max : index\n",
      "    %f16_off = index.min %f16_off_raw, %cache16_max : index\n"
      "    %k16_pre = index.mul %krow, %kv_width : index\n"
      "    %k16_raw = index.add %k16_pre, %hbkq : index\n"
      "    %k16_max = index.sub %k16_elems, %head_dim : index\n"
      "    %k16_base = index.min %k16_raw, %k16_max : index\n")
    for a in ("%k16_off0 = index.add %f16_off, %p3 :", "%k16_off1 = index.add %f16_off, %p3b :", "%k16_off3 = index.add %f16_off, %i3 :"):
        r(a, a.replace("%f16_off", "%k16_base"))
    text, n = re.subn(r"(%k16_view\[[^\]]*\] : f16, )view<\[%cache16_elems\]xf16>", r"\1view<[%k16_elems]xf16>", text)
    assert n == 3, n
    return text


def postnorm_tiled(text):
    """Rewrite yah_ssm_postnorm_fp16 to store fragment-major for an afrag ssm_out (gen_gemm_tile Tile.atiled over
    K = 6144 = 48 heads x 128): row (token, head) column c is k = head * 128 + c of that token."""
    def r(a, b):
        nonlocal text
        assert text.count(a) == 1, a[:60]
        text = text.replace(a, b)
    r("  %c128 = index.constant 128 : index\n",
      "  %c128 = index.constant 128 : index\n"
      "  %pt16 = index.constant 16 : index\n"
      "  %pt48 = index.constant 48 : index\n"
      "  %pt256 = index.constant 256 : index\n"
      "  %pt384 = index.constant 384 : index\n")
    r("      view.store %half, %out_view[%off2] : f16, view<[%io_total]xf16>\n",
      "      %pt_tok = index.div %head, %pt48 : index\n"
      "      %pt_hh = index.rem %head, %pt48 : index\n"
      "      %pt_k0 = index.mul %pt_hh, %c128 : index\n"
      "      %pt_k = index.add %pt_k0, %column : index\n"
      "      %pt_tt = index.div %pt_tok, %pt16 : index\n"
      "      %pt_tr = index.rem %pt_tok, %pt16 : index\n"
      "      %pt_kt = index.div %pt_k, %pt16 : index\n"
      "      %pt_kr = index.rem %pt_k, %pt16 : index\n"
      "      %pt_a = index.mul %pt_tt, %pt384 : index\n"
      "      %pt_b = index.add %pt_a, %pt_kt : index\n"
      "      %pt_c = index.mul %pt_b, %pt256 : index\n"
      "      %pt_d = index.mul %pt_tr, %pt16 : index\n"
      "      %pt_e = index.add %pt_c, %pt_d : index\n"
      "      %pt_off = index.add %pt_e, %pt_kr : index\n"
      "      view.store %half, %out_view[%pt_off] : f16, view<[%io_total]xf16>\n")
    return text


def postnorm_heads(text, h0, nh, B):
    """Restrict yah_ssm_postnorm_fp16 to heads h0 .. h0 + nh - 1 (config head_count = B * nh rows): row r is token r / nh,
    head h0 + r % nh, i.e. flat row (r / nh) * 48 + h0 + r % nh of the full [B][48][128] buffers. Same math per row."""
    def r(a, b):
        nonlocal text
        assert text.count(a) == 1, a[:60]
        text = text.replace(a, b)
    r("  %io_total = index.mul %head_count, %c128 : index\n",
      f"  %pn_full = index.constant {48 * B * 128} : index\n"
      "  %io_total = index.add %pn_full, %c0 : index\n")
    r("  %head = index.add %head0, %c0 : index\n  %in_range = index.cmp ult, %head, %head_count : index\n",
      f"  %pn_nh = index.constant {nh} : index\n"
      f"  %pn_h0 = index.constant {h0} : index\n"
      "  %pn_c48 = index.constant 48 : index\n"
      "  %pn_tok = index.div %head0, %pn_nh : index\n"
      "  %pn_hr = index.rem %head0, %pn_nh : index\n"
      "  %pn_h = index.add %pn_hr, %pn_h0 : index\n"
      "  %pn_f = index.mul %pn_tok, %pn_c48 : index\n"
      "  %head = index.add %pn_f, %pn_h : index\n"
      "  %in_range = index.cmp ult, %head0, %head_count : index\n")
    return text


# O16_MT: kstore shapes whose outputs are stored as f16 (their consumers read f16): 640 = attn_qkv (conv, prep_ab), 384 =
# attn_gate (the DeltaNet z: postnorm). Not bit-exact; gate N (see engine/run/gate) with the f16 qkv + z simulation:
# mean KL 1.8e-6, p99.9 1.3e-4, 0 flips on 16 windows.
O16_MT = (640, 384)

# DNSPLIT: DeltaNet as two dispatches, heads 0-39 (80 workgroups: two full rounds of 40 slots) and 40-47 (16), with the
# postnorm split the same way; the second part writes its own output buffer, so the first part's postnorm depends only on
# the first DeltaNet part and runs beside the second (whose 16 workgroups fill 16 of 40 slots). Same math per head.
DNSPLIT = (40, 8)


def _rewrite(text, pairs):
    for a, b in pairs:
        assert text.count(a) == 1, a
        text = text.replace(a, b)
    return text


def conv_tb_ok(B):
    """The f16 hand-off conv per (256 channels, gen_conv_kq.TB tokens) workgroup applies."""
    return 640 in O16_MT and B % gen_conv_kq.TB == 0 and B % 32 == 0 and gen_gdn_chunk.CONV16


def conv_x16(text):
    """yah_ssm_conv_kq reading the qkv GEMM's f16 output (O16_MT 640): each tap widened back to f32 at its load."""
    pairs = [("%x_view = buffer.view %x_noalias[%base] : buffer -> view<[%x_total]xf32>",
              "%x_view = buffer.view %x_noalias[%base] : buffer -> view<[%x_total]xf16>")]
    for v in ("s3", "x1", "x2", "x3"):
        off = "%s3_off" if v == "s3" else f"%{v}_off"
        pairs.append((f"    %{v} = view.load %x_view[{off}] : view<[%x_total]xf32> -> f32\n",
                      f"    %{v}h = view.load %x_view[{off}] : view<[%x_total]xf16> -> f16\n"
                      f"    %{v} = scalar.extf %{v}h : f16 to f32\n"))
    return _rewrite(text, pairs)


def prepab_x16(text):
    """yah_deltanet_prep_ab reading the qkv GEMM's f16 output for the conv ring (O16_MT 640)."""
    return _rewrite(text, [
        ("%qkv_view = buffer.view %qkv_na[%base] : buffer -> view<[%qkv_total]xf32>",
         "%qkv_view = buffer.view %qkv_na[%base] : buffer -> view<[%qkv_total]xf16>"),
        ("%v = view.load %qkv_view[%src] : view<[%qkv_total]xf32> -> f32\n",
         "%vh = view.load %qkv_view[%src] : view<[%qkv_total]xf16> -> f16\n        %v = scalar.extf %vh : f16 to f32\n")])


def postnorm_g16(text):
    """yah_ssm_postnorm_fp16 reading the DeltaNet z gate as the f16 output of its GEMM (O16_MT 384)."""
    return _rewrite(text, [
        ("%gate_view = buffer.view %gate_na[%base] : buffer -> view<[%io_total]xf32>",
         "%gate_view = buffer.view %gate_na[%base] : buffer -> view<[%io_total]xf16>"),
        ("%g = view.load %gate_view[%off2] : view<[%io_total]xf32> -> f32\n",
         "%gh = view.load %gate_view[%off2] : view<[%io_total]xf16> -> f16\n      %g = scalar.extf %gh : f16 to f32\n")])


def conv_n16(text):
    """yah_ssm_conv_kq handing DeltaNet (gen_gdn_chunk CONV16) its f16 inputs: conv_out holds f16(v) for the v channels and
    f16(k * inv_k), f16(q * q_scale) for the q / k channels, the f32 products and rounding DeltaNet did itself on the f32
    conv_out, so the result is bit-identical. Lane 0 of a q / k workgroup also leaves inv_k / q_scale in LDS for the others."""
    def rep(a, b):
        nonlocal text
        assert text.count(a) == 1, a
        text = text.replace(a, b)
    rep("%out_view = buffer.view %out_noalias[%base] : buffer -> view<[%x_total]xf32>",
        "%out_view = buffer.view %out_noalias[%base] : buffer -> view<[%x_total]xf16>")
    rep("%kq_bytes = index.constant 1024 : offset", "%kq_bytes = index.constant 1040 : offset")
    text = text.replace("view<256xf32>", "view<258xf32>")
    rep("    view.store %result, %out_view[%s3_off] : f32, view<[%x_total]xf32>\n", """    %is_v = index.cmp uge, %wg, %nkh : index
    scf.if %is_v {
      %result_h = scalar.fptrunc %result : f32 to f16
      view.store %result_h, %out_view[%s3_off] : f16, view<[%x_total]xf16>
    }
""")
    rep("        view.store %kq_all, %scales_view[%o2] : f32, view<[%scale_total]xf32>\n      }\n    }\n  }\n", """        view.store %kq_all, %scales_view[%o2] : f32, view<[%scale_total]xf32>
        %ks_slot = index.constant 256 : index
        %qs_slot = index.constant 257 : index
        view.store %inv_k, %kq_lds[%ks_slot] : f32, view<258xf32>
        view.store %q_scaled, %kq_lds[%qs_slot] : f32, view<258xf32>
      }
    }
    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
    %ks_at = index.constant 256 : index
    %qs_at = index.constant 257 : index
    %ks = view.load %kq_lds[%ks_at] : view<258xf32> -> f32
    %qs = view.load %kq_lds[%qs_at] : view<258xf32> -> f32
    %own_scale = scf.select %lane_hi, %ks, %qs : f32
    %own = view.load %kq_lds[%lane] : view<258xf32> -> f32
    %own_n = scalar.mulf %own, %own_scale : f32
    %own_h = scalar.fptrunc %own_n : f32 to f16
    %own_row = index.mul %t, %qkv_dim : index
    %own_off = index.add %own_row, %c : index
    view.store %own_h, %out_view[%own_off] : f16, view<[%x_total]xf16>
  }
""")
    return text


NORM_SPLIT = 2


def rope_live(text):
    """yah_fused_qk_rope_batched without its dead stores: nothing in the prefill reads the f32 K / V cache copies
    (k_cache / v_cache) or the in-place f32 k_out (the attention reads the f16 caches)."""
    n = 0
    out = []
    for line in text.split("\n"):
        if re.match(r"\s*view\.store %\w+, %(ko|kc|vc)_view\[", line):
            n += 1
            continue
        out.append(line)
    assert n >= 5, n
    return "\n".join(out)


def rope_q16(text):
    """yah_fused_qk_rope_batched storing Q as f16(q * 0.0625): the attention's own Q scale and rounding (gen_attn_fa
    Q16), done where Q is produced (half the Q bytes written and read; the same bits)."""
    a = "%qo_view = buffer.view %qo_na[%base] : buffer -> view<[%q_elems]xf32>"
    assert text.count(a) == 1
    text = text.replace(a, a.replace("xf32>", "xf16>") + "\n  %qsc16 = scalar.constant 0.0625 : f32")
    stores = re.findall(r"view\.store (%\w+), %qo_view\[(%\w+)\] : f32, view<\[%q_elems\]xf32>", text)
    assert len(stores) == 3, stores
    for v, i in stores:
        text = text.replace(f"view.store {v}, %qo_view[{i}] : f32, view<[%q_elems]xf32>",
                            f"%{v[1:]}q16s = scalar.mulf {v}, %qsc16 : f32\n      %{v[1:]}q16h = scalar.fptrunc %{v[1:]}q16s : f32 to f16\n"
                            f"      view.store %{v[1:]}q16h, %qo_view[{i}] : f16, view<[%q_elems]xf16>")
    return text


def qk_of(fmt):
    return next(qk for f, _, qk in E.FMT.values() if f == fmt)


def gemm(fmt, mt, kb, B, out, outdir, kind="kstore"):
    """Emit the tile GEMM (tools/gen_gemm_tile.py) of this shape and kind; return its dispatch.txt row, or None."""
    r = tile_kstore(fmt, mt, kb, B, out, outdir, kind)
    if not r and mt < 4:
        # the 48-row ssm_alpha/ssm_beta: one 16-row tile per workgroup, 64 tokens over 2 waves, grid (3, B/64)
        r = tile_kstore(fmt, mt, kb, B, out, outdir, kind, geom=(16, 64, 1, 2))
    return r


OCC3_FROM = 86016   # first context position whose chunk's attention runs 3 workgroups per WGP (see emit: write_attn)
OCC3_LDS = 33792    # the LDS floor that caps a WGP at 3 attention workgroups (128 KB / 4 < 33 KB)


# From KT2_FROM keys of context on (fp16 attention), a chunk runs the 32-key-tile role-split kernel (gen_attn_fa KT2):
# 3 workgroups per WGP, one barrier pair / reduction / rescale vote per 32 keys; -0.8% at 16K, -1.8% at 32K, -3..-5.6% at
# 88K..126K vs the 4-workgroup kernel / the deep twin (standalone); not bit-identical (the running max moves per 32 keys).
KT2_FROM = int(os.environ.get("YAH_KT2_FROM", "16384"))
KT2_FLAGS = dict(PAIR=True, PAIR_LATE=False, QKF=1, PVF=2, KT2=True, KT2_IL=1)
KT2_VGPR = 240   # 3 workgroups per WGP: 6 waves x 240 (VGPRs allocate in granules of 24 on gfx1151)


def deep(path):
    return path.replace(".loom", "_deep.loom")


def kt2(path):
    return path.replace(".loom", "_kt2.loom")


def alt(path):
    return path.replace(".loom", "_alt.loom")


ATTN_VGPR = 192   # every attention variant fits here unless the allocator rotates the O accumulators


def hal_vgprs(hal):
    notes = subprocess.run(["/opt/rocm/llvm/bin/llvm-readelf", "--notes", hal], capture_output=True, text=True).stdout
    m = re.search(r"\.vgpr_count:\s*(\d+)", notes)
    return int(m.group(1)) if m else 0


def hal_spills(hal):
    notes = subprocess.run(["/opt/rocm/llvm/bin/llvm-readelf", "--notes", hal], capture_output=True, text=True).stdout
    m = re.search(r"\.private_segment_fixed_size:\s*(\d+)", notes)
    return int(m.group(1)) if m else 0


QROT_FROM = 12288   # kv4: the first context position whose chunk rotates Q in its own kernel (see the attention)
MODEL = None  # set by main(): tiles() reads this model's table
# Decode-free GEMMs (decode_free): off by default (YAH_DECODE_FREE=1): +12% pp2048 as built (2026-10-03): the f16 swiglu
# epilogue streams its gate in bursts, the K=17408 down projection is memory-bound in f16, and the dequant pass costs
# ~25% of a GEMM. kstore / kqg alone are -9% (standalone, bit-identical).
DECODE_FREE = os.environ.get("YAH_DECODE_FREE", "0") == "1"
DECODE_FREE_KINDS = ("kstore", "kqg")


@functools.cache
def tiles():
    """The autotuner's table for MODEL (tune_table.py: YAH_TILES or engine/tune/tables/<model>.json):
    {"tiles": {hal: knobs}, "variants": {hal: [knobs incl. "bn", ...]}, "pick": {hal: {max tokens: bn}},
     "norm": {...}, "rw": {...}}. A flat {hal: knobs, ...} file is read as its "tiles"."""
    import tune_table
    t = tune_table.load(MODEL) if MODEL else {}
    if "tiles" not in t:
        t = {"tiles": {k: v for k, v in t.items() if k.startswith("gemm_")},
             **{k: v for k, v in t.items() if not k.startswith("gemm_")}}
    return t


# Narrower token tiles for chunks with few real tokens; the driver picks one per chunk (LoomPrefill::PickGemm).
# A narrow tile pads less but costs more per token row (x1.12 at 128, x1.6 at 64, measured): an 18-token prompt
# 595 -> 404 ms with 64, a 300-token prompt 909 -> 802 ms with 128. Same values as the default tile.
NARROW = ((128, 2), (64, 2))


VARIANTS = {}  # GEMM HAL -> [(variant HAL, Tile)] that narrow_variants emitted


def calibration_menu(model, outdir, B):
    """Emit "<gemm>.m<i>.hal": the prefill calibration menu (engine/tune/tune.py menu) of each tile GEMM in this set, built
    in parallel. Every menu entry and narrow variant must hash the same as its GEMM on the GPU (gemm_bench, real
    weights); a menu entry that does not is dropped, a narrow variant that does not stops the emit. Returns the rows."""
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "tune")))
    import tempfile
    from concurrent.futures import ProcessPoolExecutor
    import gen_gemm_tile as TG
    import tune
    if B != tune.CHUNK:
        return []
    work = tempfile.mkdtemp(prefix="yah-menu-")
    kernels = [k for k in tune.inventory(model) if os.path.exists(os.path.join(outdir, k.hal + ".hal"))]
    todo = []
    for k in kernels:
        base = dataclasses.replace(TG.default_tile(k.fmt, k.kind, k.kb), **tiles()["tiles"].get(k.hal, {}))
        have = [t for _, t in VARIANTS.get(k.hal + ".hal", [])]
        for i, t in enumerate(tune.menu(k, base, have)):
            todo.append((k, t, os.path.join(work, "%s.m%d" % (k.hal, i)), i))
    with ProcessPoolExecutor(os.cpu_count()) as pool:
        built = list(pool.map(tune.compile_one, [(k, t, d) for k, t, d, _ in todo]))
    # Each entry against its GEMM at a full and a partial chunk, twice: a race shows as a hash that differs on some runs.
    CHECKS = [(B, 0), (B, 1), (777, 0), (777, 1)]
    jobs, meta = [], []
    for k in kernels:
        base = dataclasses.replace(TG.default_tile(k.fmt, k.kind, k.kb), **tiles()["tiles"].get(k.hal, {}))
        roles = tune.roles_of(TG.gen(k.fmt, k.kind, base, B % base.bn != 0))
        entries = [(base, os.path.join(outdir, k.hal + ".hal"), None, None)]
        entries += [(t, os.path.join(outdir, hal), hal, None) for hal, t in VARIANTS.get(k.hal + ".hal", [])]
        entries += [(t, path, "%s.m%d.hal" % (k.hal, i), path)
                    for (kk, t, d, i), (path, _) in zip(todo, built) if kk is k and path]
        for t, path, hal, copy in entries:
            for n, _ in CHECKS:
                jobs.append((k, t, path, roles, n, 0))
            meta.append((k, t, hal, copy))
    res = [h for _, h in tune.bench(model, tune.tables(work), jobs, work, "menu", counters=False)]
    rows, ref, dropped = [], None, 0
    for j, (k, t, hal, copy) in enumerate(meta):
        h = tuple(res[j * len(CHECKS):(j + 1) * len(CHECKS)])
        if hal is None:
            ref = h
            if len(set(h[0:2])) > 1 or len(set(h[2:4])) > 1:
                raise SystemExit("%s.hal is nondeterministic" % k.hal)
        elif h != ref:
            if not copy:
                raise SystemExit("variant %s computes different values from %s.hal" % (hal, k.hal))
            dropped += 1
            print("calibration menu: %s differs from %s.hal (%s), dropped" % (hal, k.hal, tune.knobs(t, TG.default_tile(
                k.fmt, k.kind, k.kb))))
        elif copy:
            shutil.copy(copy, os.path.join(outdir, hal))
            rows.append((hal, t.bn, t.rowgrp, -(-B // t.bn)))
    shutil.rmtree(work, ignore_errors=True)
    print("calibration menu: %d entries for %d GEMMs (%d failed to build, %d differed)"
          % (len(rows), len(kernels), sum(1 for p, _ in built if not p), dropped))
    return rows


def narrow_variants(fmt, mt, kb, B, out, outdir, kind):
    """Emit "<hal>.t<BN>.hal" for each narrow token tile of this GEMM (the tuner's "variants", else NARROW on the
    default tile), plus the tuner's per-bucket choices as "pick:<hal>:<max tokens> <max tokens> <bn> 0" rows."""
    import gen_gemm_tile as TG
    if fmt not in TILE_FMTS or mt < 4:
        return []
    base, rows = TG.default_tile(fmt, kind, kb), []
    full = dataclasses.replace(base, **tiles()["tiles"].get(out[:-4], {}))
    VARIANTS[out] = []
    tuned = tiles().get("variants", {}).get(out[:-4])
    for knobs in (tuned if tuned is not None else [{"bn": bn, "wn": wn} for bn, wn in NARROW]):
        t = dataclasses.replace(base, **knobs)
        bn = t.bn
        try:
            TG.check(t)
        except ValueError:
            continue
        if bn >= full.bn or mt % t.rowgrp:
            continue
        masked = B % bn != 0
        r = _emit_gen(lambda f, k: TG.gen(f, k, t, masked), bn, fmt, mt, kb, B, out[:-4] + ".t%d.hal" % bn, outdir, kind,
                      t.rowgrp, masked)
        if r:
            rows.append(r)
            VARIANTS[out].append((r[0], t))
    for maxtok, bn in sorted(tiles().get("pick", {}).get(out[:-4], {}).items(), key=lambda kv: int(kv[0])):
        rows.append(("pick:%s:%s" % (out, maxtok), int(maxtok), int(bn), 0))
    return rows


# Formats the tile GEMM (tools/gen_gemm_tile.py) is verified bit-identical on in the pp2048 pipeline.
TILE_FMTS = ("iq3s", "iq4xs", "iq3xxs", "q4k", "q5k", "q6k", "iq2xxs", "iq2xs", "q3k", "q8_0")


def decode_free(fmt, mt, kb, B, kind, outdir, done):
    """Decode-free GEMMs: "dq_<fmt>_<mt>_<kb>.hal" decodes the weights to f16 [rows][K] once per chunk (the tile GEMM's own
    decode, so the same f16 values), "gemm_<kind>_f16_<mt>_<kb>.hal" multiplies them without decode. Shared per shape.
    The driver dequantizes GEMM k+1's weights beside GEMM k. Returns the new dispatch.txt rows."""
    import gen_gemm_tile as TG
    # only where it wins (2026-10-03, clock-free): kstore -9%, kqg -6%; kres at K = 17408 is memory-bound in f16 and the
    # swiglu epilogue's gate stream lands in bursts under the token-fastest order (+5%)
    if fmt not in TILE_FMTS or mt < 4 or kind not in DECODE_FREE_KINDS:
        return []
    rows = []
    kblk = kb // TG.G.FMTS[fmt].get("kdiv", 1)   # 256-wide blocks (q8_0 counts 32-wide ones)
    dq = "dq_%s_%d_%d.hal" % (fmt, mt, kb)
    if dq not in done:
        # 64 rows, double-buffered: 2 x 9 KB (+ the IQ grid table) fits the 20 KB of LDS two GEMM workgroups leave free
        # on a WGP, so the dequant runs beside the GEMM instead of after it
        t = TG.default_tile(fmt, "kstore", kb)
        t = dataclasses.replace(t, bm=64, wm=2, wn=4, decahead=False, ksub=64, dbuf=False)
        if mt % t.rowgrp == 0 and kblk % TG.DQ_BLOCKS == 0:
            r = _emit_gen(lambda f, k: TG.gen(f, "dequant", t), t.bn, fmt, mt, kb, B, dq, outdir, "dequant", t.rowgrp)
            if r:
                rows.append((dq, TG.DQ_WGS, t.rowgrp, 1))   # persistent: <workgroups> in the token-tile column
                done.add(dq)
    gf = "gemm_%s_f16_%d_%d.hal" % (kind, mt, kblk)
    if dq in done and gf not in done:
        t = TG.default_tile("f16", kind, kblk)
        r = _emit_gen(lambda f, k: TG.gen("f16", k, t), t.bn, "f16", mt, kblk, B, gf, outdir, kind, t.rowgrp)
        if r:
            rows.append(r)
            done.add(gf)
    return rows


# Activations-from-global GEMMs (gen_gemm_tile Tile.afrag): "<hal>.af.hal" reads a fragment-major input (Tile.atiled; the
# FFN norm "norm_t.hal" and the swiglu "<hal>.af.to.hal" write it). 512-token tiles, only the decoded weights in LDS.
# Same values as the GEMM. YAH_AFRAG=0 leaves them out.
AFRAG = os.environ.get("YAH_AFRAG", "1") != "0"
AF_TILE = dict(bm=128, bn=512, wm=1, wn=16, ksub=128, dbuf=False, decahead=False, afrag=True, atiled=True, stg_minwg=0,
               tallepi=True, ecoal=True, esr=32)
AF_PIPE = dict(wlate=True, bpre=2, b0early=True)
# IQ3_S: step 1's B prefetch issued with step 0's, before the decode; otherwise the allocator copies one of its fragments
# right after its load, which drains vmcnt(0) once per phase (IQ4_XS: 200 VGPRs with it, fused IQ3_S ffn: 216)
AF_PIPE3 = dict(AF_PIPE, b0early=2)
AF_KQ = dict(rhs_outer=False, rhs_fence=0)   # K-quants without B prefetch (Q6_K: 208 VGPRs with it)
# Q4_K / Q3_K / Q5_K with the B prefetch: their decoders came down to 184-192 VGPRs (Q4DFMA, Q3_FMIX), so it fits now:
# -3.6..-4.0% clock-free (Q4_K swiglu -5.6%, kstore 64 -5.5%)
AF_KQP = dict(AF_KQ, **AF_PIPE, lhs_stream=4)
# The GEMMs with an afrag form and their knobs on top of AF_TILE; clock-free vs the GEMM (one round each, 2026-10-03).
# The best lhs_stream depends on the decoder's register allocation: IQ3_XXS kstore without it waits vmcnt(0) twice per
# phase on the step-1 B prefetch. Q3_K: the prefetch needs 208 VGPRs (one workgroup per WGP). IQ3_XXS: lhs_stream=1 with
# the gen_gemm_decode.IQ3_SGTAB sign table (deeper A streaming splits the A-fragment tuples at the 192-VGPR tier: 8 copies
# per k step); alone it is neutral, with the table -0.7..-1.0% (one process A/B 2026-10-04). Not the fused ffn: its
# fragment-major-output form loses +1.3% with the table although its loop drops 32 VALU; nor swiglu (same, +1.3% in pp2048).
# IQ3_S (256-entry table of the block's sign bytes): kres 320x68 -0.7%, kqg -1.0%, kstore 384 -0.5%; kstore 1088 / 640
# +0.5 / +0.2% and the fragment-major ffn / swiglu +0.7 / +2.2% keep the spread.
# f16p (IQ3 decode bytes as f16 subnormal pairs, one v_perm for the odd bytes; -11..-15 loop VALU): IQ3_S kstore 1088 /
# kres 320x68 / kqg / swiglu -0.25..-0.55%, fused ffn IQ3_XXS / IQ3_S -0.35%; IQ3_XXS kstore / kqg neutral, kres +1.8% (not).
# gaddr (IQ3 grid / sign-table lookups at pre-scaled byte offsets on byte views of the LDS tables: IQ3_XXS -15, IQ3_S
# -17..-23 loop VALU; one process A/B 2026-10-04): IQ3_XXS kstore -0.3..-0.8%, kqg -0.25%, kres -0.6..-1.2%, swiglu -0.7%;
# IQ3_S kstore 640 / kqg -0.25..-0.3%, kres -0.8..-1.0%. Not: IQ3_S kstore 384 (200-208 VGPRs at every lhs_stream, +12%),
# IQ3_S kstore 1088 / swiglu and both fused ffn (neutral).
# f16s (IQ3 signed f16 subnormal pairs: two v_perm per grid word instead of sign xor / add + pair split, fma addend +0;
# IQ3_XXS loop VALU 156 -> 114; one process A/B 2026-10-04): IQ3_XXS kstore / kqg -1.2..-1.5% (lhs_stream=3), kres -0.85%
# (wlate=False: 188 VGPRs, else 200-208), swiglu -0.4%, fused ffn -1.0% (no sign table); IQ3_S kstore -0.8..-1.5%,
# kqg -1%, kres -0.7..-1.7%, fused ffn -0.5%. Not: IQ3_S swiglu (+0.2..+25%).
# tallepi (kstore / kres): the LDS epilogue in 16-row slabs; the fragment stores cost the K = 6144 kres 15%.
AF = {
    ("iq3s", "kstore", 1088, 20): dict(AF_PIPE3, lhs_stream=2, f16p=True, f16s=True, sgtab=True),  # -9.1%
    ("iq4xs", "kstore", 1088, 20): dict(AF_PIPE, lhs_stream=2),                 # -12.9%
    ("iq3xxs", "kstore", 1088, 20): dict(AF_PIPE, lhs_stream=3, sgtab=True, gaddr=True, f16s=True),  # -11.3%
    ("q3k", "kstore", 1088, 20): dict(AF_KQP),                                  # -7.2%
    ("iq3s", "swiglu", 1088, 20): dict(AF_PIPE3, lhs_stream=2, swepi=False, f16p=True),     # -11.6%
    ("iq4xs", "swiglu", 1088, 20): dict(AF_PIPE, lhs_stream=2, swepi=False),    # -10.9%
    ("iq3xxs", "swiglu", 1088, 20): dict(AF_PIPE, lhs_stream=2, swepi=False, gaddr=True, f16s=True),  # -12.1%
    ("iq3s", "kres", 320, 68): dict(AF_PIPE3, sgtab=True, f16p=True, gaddr=True, f16s=True,
                                    persist=dict(b0early=True, lhs_stream=2)),               # -8.6%
    ("iq4xs", "kres", 320, 68): dict(AF_PIPE, lhs_stream=4, respre=1, persist=True), # -10.4%
    ("iq3xxs", "kres", 320, 68): dict(AF_PIPE, lhs_stream=1, sgtab=True, gaddr=True, f16s=True, wlate=False, respre=1,
                                      persist=True),                                         # -8.2%
    ("q4k", "kres", 320, 68): dict(AF_KQP, respre=1, persist=True),             # -10.4%
    ("q4k", "swiglu", 1088, 20): dict(AF_KQP, swepi=False),                     # -7.7%
    # attention o-proj / DeltaNet ssm_out (K = 6144; input: the attention output, postnorm_t.hal)
    ("iq3s", "kres", 320, 24): dict(AF_PIPE3, gaddr=True, f16s=True, sgtab=True,
                                    persist=dict(b0early=True, lhs_stream=2)),               # -8.8%
    ("iq3xxs", "kres", 320, 24): dict(AF_PIPE, lhs_stream=1, sgtab=True, gaddr=True, f16s=True, wlate=False, respre=1,
                                      persist=True),                                         # -8.5%
    ("iq4xs", "kres", 320, 24): dict(AF_PIPE, lhs_stream=4, respre=1, persist=True), # -8.8%
    ("q4k", "kres", 320, 24): dict(AF_KQP, respre=1, persist=True),             # -5.3%
    # DeltaNet qkv (10240 rows) / gate (6144 rows); input: norm_rt.hal (alpha / beta keep the row-major copy)
    ("iq3s", "kstore", 640, 20): dict(AF_PIPE3, lhs_stream=3, gaddr=True, f16s=True),       # -9.6%
    ("iq4xs", "kstore", 640, 20): dict(AF_PIPE, lhs_stream=2),                  # -10.3%
    ("iq3xxs", "kstore", 640, 20): dict(AF_PIPE, lhs_stream=3, sgtab=True, gaddr=True, f16s=True),  # -11.4%
    ("q4k", "kstore", 640, 20): dict(AF_KQP),                                   # -7.4%
    ("q3k", "kstore", 640, 20): dict(AF_KQP),                                   # -7.7%
    ("iq3s", "kstore", 384, 20): dict(AF_PIPE3, lhs_stream=3, sgtab=True, f16s=True),       # -9.4%
    ("iq4xs", "kstore", 384, 20): dict(AF_PIPE, lhs_stream=2),                  # -9.9%
    ("iq3xxs", "kstore", 384, 20): dict(AF_PIPE, lhs_stream=3, sgtab=True, gaddr=True, f16s=True),  # -10.9%
    ("q4k", "kstore", 384, 20): dict(AF_KQP),                                   # -6.0%
    ("q3k", "kstore", 384, 20): dict(AF_KQP),                                   # -6.4%
    ("q5k", "kstore", 384, 20): dict(AF_KQP, ksl=True),                         # -12%
    ("q5k", "kres", 320, 24): dict(AF_KQP, ksl=True, respre=1, persist=True),   # -4.6%
    # attention q (kqg: 12288 rows, q / gate split) and k / v (1024 rows); input: attn_norm (norm_t / norm_rt)
    ("iq3s", "kqg", 768, 20): dict(AF_PIPE3, lhs_stream=1, sgtab=True, f16p=True, gaddr=True, f16s=True),  # -8.4%
    ("iq4xs", "kqg", 768, 20): dict(AF_PIPE, lhs_stream=2),                     # -9.5%
    ("iq3xxs", "kqg", 768, 20): dict(AF_PIPE, lhs_stream=3, sgtab=True, gaddr=True, f16s=True),  # -9.3%
    ("q4k", "kqg", 768, 20): dict(AF_KQP),                                      # -6.7%
    ("q3k", "kqg", 768, 20): dict(AF_KQP),                                      # -5.0%
    ("q5k", "kqg", 768, 20): dict(AF_KQP, ksl=True),                            # -10.8%
    ("iq4xs", "kstore", 64, 20): dict(AF_PIPE),                                 # -15.7%
    ("q4k", "kstore", 64, 20): dict(AF_KQP),                                    # -16.0%
    ("q5k", "kstore", 64, 20): dict(AF_KQP, ksl=True),                          # -10.4%
    ("q6k", "kstore", 64, 20): dict(AF_KQ, ksl=True, wlate=True),               # -10.2% (208 VGPRs: 32 workgroups anyway)
    # IQ2: the grid / sign lookups leave no room for B prefetch (bpre 1-2: 208-216 VGPRs, one workgroup per WGP)
    ("iq2xxs", "kstore", 1088, 20): dict(ksl=True, wlate=True, b0early=True, lhs_stream=2),   # -19.3%
    ("iq2xxs", "swiglu", 1088, 20): dict(ksl=True, wlate=True, b0early=True, lhs_stream=2),   # -16.6%
    ("iq2xs", "kstore", 1088, 20): dict(ksl=True, wlate=True, b0early=True, lhs_stream=2),    # -18.4%
}


# Fused ffn_gate + ffn_up (gen_gemm_tile kind "ffn"): one afrag GEMM writes f16(silu(gate) * up) for layers whose gate
# and up share a format (the driver binds both tensors as one: the GGUF stores ffn_up right after ffn_gate). Same
# values as kstore + swiglu. vs the two afrag GEMMs, clock-free (2026-10-03, fast reciprocal in both): IQ3_S -3.5%,
# IQ4_XS -0.9%, IQ3_XXS -3.1%.
AF_FFN = {
    "iq3s": dict(AF_PIPE, lhs_stream=2, swepi=False, f16p=True, f16s=True),
    "iq4xs": dict(AF_PIPE, lhs_stream=2, swepi=False),
    # IQ3_XXS: f16s without a sign table (-1.0%; with the packed sgtw=1 table +0.9..+1.6%, which itself had been -0.5%
    # over the plain f16p form; the two-word table +1.3%: LDS bank conflicts 13.9 -> 23.8 M cycles, LDS issue waits +62%)
    "iq3xxs": dict(AF_PIPE, lhs_stream=1, swepi=False, f16p=True, gaddr=True, f16s=True),
}


def ffn_fused(rows, B, outdir):
    """Emit "gemm_ffn_<fmt>_<mt>_<kb>.af.hal" / ".af.to.hal" for each AF_FFN format some layer uses for both its gate
    and up; return the dispatch.txt rows."""
    import gen_gemm_tile as TG
    if not AFRAG or B % AF_TILE["bn"]:
        return []
    by = {}
    for nm, dims, ty in rows:
        parts = nm.split(".")
        if len(parts) > 2 and parts[0] == "blk" and parts[2] in ("ffn_gate", "ffn_up") and E.FMT.get(ty):
            by.setdefault(parts[1], {})[parts[2]] = (E.FMT[ty][0], dims[1] // 16, dims[0] // E.FMT[ty][2])
    out = []
    for fmt, mt, kb in sorted({l["ffn_gate"] for l in by.values() if l.get("ffn_gate") == l.get("ffn_up")}):
        if fmt not in AF_FFN:
            continue
        t = dataclasses.replace(TG.default_tile(fmt, "swiglu", kb), **AF_TILE, **AF_FFN[fmt], ffn=True)
        for suffix, tt in ((".af.hal", t), (".af.to.hal", dataclasses.replace(t, tout=True))):
            TG.check(tt)
            r = _emit_gen(lambda f, k: TG.gen(f, k, tt, False), tt.bn, fmt, mt, kb, B,
                          "gemm_ffn_%s_%d_%d%s" % (fmt, mt, kb, suffix), outdir, "ffn", tt.rowgrp)
            if r:
                out.append(r)
    return out


def afrag_variants(fmt, mt, kb, B, out, outdir, kind):
    """Emit "<hal>.af.hal" (and for swiglu "<hal>.af.to.hal", fragment-major output) if this GEMM has an afrag form."""
    import gen_gemm_tile as TG
    knobs = AF.get((fmt, kind, mt, kb))
    if not AFRAG or knobs is None or B % AF_TILE["bn"]:
        return []
    knobs = dict(knobs)
    persist = knobs.pop("persist", False)
    t = dataclasses.replace(TG.default_tile(fmt, kind, kb), **AF_TILE, **knobs)
    rows = []
    for suffix, tt in [(".af.hal", t)] + ([(".af.to.hal", dataclasses.replace(t, tout=True))] if kind == "swiglu" else []):
        TG.check(tt)
        r = _emit_gen(lambda f, k: TG.gen(f, k, tt, False), tt.bn, fmt, mt, kb, B, out[:-4] + suffix, outdir, kind,
                      tt.rowgrp)
        if r:
            rows.append(r)
    if persist and kind == "kres" and B // t.bn > 1 and 2 * kb >= gen_kres_persist.SLICES:
        # the persistent kres (gen_kres_persist): grid y = 1, each workgroup runs every token tile of its row block
        # persist: True, or a dict of knobs for the persistent variant (its slice loads need registers: IQ3_S drops b0early 2)
        pt = dataclasses.replace(t, **persist) if isinstance(persist, dict) else t
        text = gen_kres_persist.persist(TG.gen(fmt, "kres", pt, False),
                                        TG.gen(fmt, "kstore", dataclasses.replace(pt, respre=0), False), B // t.bn)
        r = _emit_gen(lambda f, k: text, t.bn, fmt, mt, kb, B, out[:-4] + ".af.p.hal", outdir, kind, t.rowgrp)
        if r:
            rows.append(r)
    return rows


# NPU column split (YAH_NPU_SPLIT="<site>=<NPU rows>,...", rows multiples of 640, q also of 512; empty: none): the NPU (XDNA2, HRX .xdna)
# computes the trailing rows of a GEMM, the GPU the leading ones. Sites (rows x K):
#   qkv  DeltaNet attn_qkv (10240 x 5120, f16 out)     gate  DeltaNet attn_gate (6144 x 5120, f16 out)
#   q    attention attn_q (12288 x 5120, kqg: whole heads of [256 q | 256 gate])
#   out  ssm_out / attn_output (5120 x 6144, kres)      down  ffn_down (5120 x 17408, kres)
# The NPU runs one image (K = 5120 per call: switching images costs ~0.55 ms each): out and down take K chunks of 5120 and
# the GPU computes the rest of K for the NPU's rows (npu_rem_tile, gen_gemm_tile.KWIN), which the unpack adds.
#   ffn  ffn_gate + ffn_up (17408 x 5120 each): the same NPU rows of both, silu(gate) * up in the unpack
# Per site and format, the GPU's share of every variant the driver may run ("<hal>.npu.hal": "<hal>" any of the set's
# GEMM HALs, its afrag / persistent / fragment-major-output forms included; gen_gemm_tile.OSTRIDE) and the NPU's rows
# decoded straight to its BFP16 weight stream ("dqbfp_<fmt>_<mt>_<kb>[_c<chunk>].hal", gen_gemm_tile.DQ_BFP). Per K the
# activation encoders ("npu_enc_<K>[_c<chunk>][_t].hal", _t: fragment-major input) and the NPU GEMM image
# ("npu_gemm_<K>.xdna", 8 columns of gen_npu_gemm.TN = 640 rows per call: "npurows"); per site the unpack ("npu_unpack_<site>.hal", ffn also
# "npu_unpack_ffn_t.hal" with fragment-major output; qkv: "npu_unpack_qkv_tail.hal" + the conv split "convkq_g.hal" /
# "convkq_c.hal", which reads C itself). dispatch.txt: "npusplit_<site> <rows> 0 0", "npubytes_<K> <A>
# <W per call> <C per call>", and per kernel "<hal> <workgroups> <workgroup size> 0" (the split GEMMs have GEMM rows).
NPU_SPLIT = dict((k, int(v)) for k, v in (x.split("=") for x in os.environ.get("YAH_NPU_SPLIT", "").split(",") if x))
NPU_KS = GN.KS   # k-blocks per pass and K-slice row; K = 1024 * passes
NPU_ROWS = 8 * GN.TN   # output rows per NPU call (8 columns)
NPU_SITES = {"qkv": ("kstore", 640, 20), "gate": ("kstore", 384, 20), "q": ("kqg", 768, 20), "out": ("kres", 320, 24),
             "down": ("kres", 320, 68), "ffn": ("ffn", 1088, 20)}
NPU_PASSES = 5   # one NPU image: K = 5120 per call (gen_npu_gemm passes)


def npu_chunks(site):
    """(k offset, passes) of the NPU's K chunks of a site; the GPU computes the rest of K (npu_rem) for the NPU's rows."""
    K = NPU_SITES[site][2] * 256
    return [(1024 * NPU_PASSES * i, NPU_PASSES) for i in range(K // (1024 * NPU_PASSES))]


def npu_rem(site):
    """The K columns past the NPU's chunks (out: 1024 of 6144, down: 2048 of 17408): the GPU's npurem_<site>_<fmt>.hal."""
    return NPU_SITES[site][2] * 256 % (1024 * NPU_PASSES)


def npu_rem_tile(fmt, kbw, kbt):
    """The K remainder's tile (gen_gemm_tile.KWIN, kbw of kbt blocks): the afrag kstore with the knobs of the format's
    K = kbt * 256 kres (emit_prefill_pp.AF; its residual and persistence knobs dropped)."""
    import gen_gemm_tile as TG
    knobs = dict(AF.get((fmt, "kres", 320, kbt)) or AF[(fmt, "kres", 320, 24)])
    knobs.pop("persist", None)
    knobs["respre"] = 0
    return dataclasses.replace(TG.default_tile(fmt, "kstore", kbw), **AF_TILE, **knobs)


def npu_split(rows, combos, B, outdir):
    import gen_bfp16_encode as GE
    import gen_gemm_tile as TG
    import gen_npu_unpack as GU
    sys.path.insert(0, os.path.dirname(HERE))
    import hrx_paths
    bad = set(NPU_SPLIT) - set(NPU_SITES)
    assert not bad, "unknown NPU split sites %s" % sorted(bad)
    assert all(v % NPU_ROWS == 0 and 0 < v < NPU_SITES[k][1] * 16 for k, v in NPU_SPLIT.items()) and B % 512 == 0
    assert NPU_SPLIT.get("q", 0) % 512 == 0, "q: whole heads"
    out = []
    tmp = os.path.join(outdir, ".emit_tmp")
    os.makedirs(tmp, exist_ok=True)

    def split(fmt, kind, mt, kb, t, name, nn, persist=None, tokens=B):
        """The GPU's share: rows [0, mt * 16 - nn) of the GEMM HAL name (tile t) at the full stride, over tokens."""
        mtg = mt - nn // 16
        assert mtg % t.rowgrp == 0 and tokens % t.bn == 0, (name, nn)
        TG.check(t)
        if persist is not None:
            gen = lambda f, k: gen_kres_persist.persist(TG.gen(f, "kres", persist, False),
                                                       TG.gen(f, "kstore", dataclasses.replace(persist, respre=0), False),
                                                       tokens // t.bn)
        else:
            gen = lambda f, k: TG.gen(f, k, t, False)
        out.append(_emit_gen(gen, t.bn, fmt, mtg, kb, tokens, name[:-4] + ".npu.hal", outdir, kind, t.rowgrp,
                             ostride=mt * 16))

    def variants(fmt, kind, mt, kb, base, nn):
        """Every form of GEMM base the set emits (tile_kstore, afrag_variants): its split."""
        t = TG.default_tile(fmt, kind, kb)
        if base[:-4] in tiles()["tiles"]:
            t = dataclasses.replace(t, **tiles()["tiles"][base[:-4]])
        if fmt in TILE_FMTS and mt % t.rowgrp == 0:
            split(fmt, kind, mt, kb, t, base, nn)
        knobs = AF.get((fmt, kind, mt, kb))
        if AFRAG and knobs is not None and B % AF_TILE["bn"] == 0:
            knobs = dict(knobs)
            persist = knobs.pop("persist", False)
            ta = dataclasses.replace(TG.default_tile(fmt, kind, kb), **AF_TILE, **knobs)
            split(fmt, kind, mt, kb, ta, base[:-4] + ".af.hal", nn)
            if kind == "swiglu":
                split(fmt, kind, mt, kb, dataclasses.replace(ta, tout=True), base[:-4] + ".af.to.hal", nn)
            if persist and kind == "kres" and B // ta.bn > 1 and 2 * kb >= gen_kres_persist.SLICES:
                pt = dataclasses.replace(ta, **persist) if isinstance(persist, dict) else ta
                # fewer row blocks than CUs: the persistent kres over all but the last token tile, run beside the tiled
                # kres on that tile (LoomPrefill::RunResidual)
                if B // ta.bn > 2:
                    split(fmt, kind, mt, kb, ta, base[:-4] + ".af.p3.hal", nn, persist=pt, tokens=B - ta.bn)

    def dqbfp(fmt, mt, kb, nn, ci=None, chunk=None):
        """The NPU's rows (the last nn of mt * 16) decoded to its weight stream; chunk ci: (k offset, passes) of a K chunk."""
        dt = dataclasses.replace(TG.default_tile(fmt, "kstore", kb), bm=64, wm=2, wn=4, decahead=False,
                                 ksub=TG.DQ_BFP_KSUB, dbuf=False)
        name = "dqbfp_%s_%d_%d%s.hal" % (fmt, mt, kb, "" if chunk is None else "_c%d" % ci)
        if chunk is None:
            TG.DQ_BFP, kbc, kfull = (NPU_KS, kb * qk_of(fmt) // 1024), kb, 0
        else:
            TG.DQ_BFP = (NPU_KS, chunk[1], chunk[0] // 256, kb)
            kbc, kfull = chunk[1] * 4, kb
        try:
            _emit_gen(lambda f, k: TG.gen(f, "dequant", dt), dt.bn, fmt, nn // 16, kbc, B, name, outdir, "dequant",
                      dt.rowgrp, sym="yah_dequant_%s_bfp16" % fmt, kfull=kfull)
        finally:
            TG.DQ_BFP = None
        out.append((name, TG.dq_wgs(nn // 16, kbc, dt, fmt), dt.lanes, 0))

    done = set()
    for kind, fmt, port, mt, kb in sorted(combos):
        for site, (skind, smt, skb) in NPU_SITES.items():
            if site not in NPU_SPLIT or mt != smt or kb != skb or fmt not in TILE_FMTS:
                continue
            nn = NPU_SPLIT[site]
            if site == "ffn":   # the gate kstore and the up swiglu of each format (the fused ffn below)
                if kind == "kstore":
                    variants(fmt, "kstore", mt, kb, "gemm_kstore_%s_%d_%d.hal" % (fmt, mt, kb), nn)
                elif kind == "swiglu":
                    variants(fmt, "swiglu", mt, kb, "gemm_swiglu_%s_%d_%d.hal" % (fmt, mt, kb), nn)
                else:
                    continue
            elif skind == "kres":
                if kind != "residual":
                    continue
                variants(fmt, "kres", mt, kb, "gemm_kres_%s_%d_%d.hal" % (fmt, mt, kb), nn)
            elif kind == "kstore":
                variants(fmt, skind, mt, kb, "gemm_%s_%s_%d_%d.hal" % (skind, fmt, mt, kb), nn)
            else:
                continue
            chunks = npu_chunks(site)
            chunked = len(chunks) > 1 or npu_rem(site)
            for ci, chunk in (enumerate(chunks) if chunked else ((None, None),)):
                if (fmt, mt, kb, chunk) not in done:
                    done.add((fmt, mt, kb, chunk))
                    dqbfp(fmt, mt, kb, NPU_SPLIT[site], ci, chunk)
            if npu_rem(site) and (site, fmt) not in done:   # the GPU's K remainder of the NPU's rows (f32 [B][nn])
                done.add((site, fmt))
                kbw = npu_rem(site) // 256
                tr = npu_rem_tile(fmt, kbw, kb)

                def gen_rem(f, k, tr=tr, kb=kb, kbw=kbw):
                    TG.KWIN = (kb - kbw, kb)
                    try:
                        return TG.gen(f, "kstore", tr, False)
                    finally:
                        TG.KWIN = None
                assert nn // 16 not in O16_MT, "the remainder writes f32"
                out.append(_emit_gen(gen_rem, tr.bn, fmt, nn // 16, kbw, B, "npurem_%s_%s.hal" % (site, fmt), outdir,
                                     "kstore", tr.rowgrp, kfull=kb))
    if "ffn" in NPU_SPLIT and AFRAG and B % AF_TILE["bn"] == 0:   # the fused gate + up GEMMs (ffn_fused)
        by = {}
        for nm, dims, ty in rows:
            p = nm.split(".")
            if len(p) > 2 and p[0] == "blk" and p[2] in ("ffn_gate", "ffn_up") and E.FMT.get(ty):
                by.setdefault(p[1], {})[p[2]] = (E.FMT[ty][0], dims[1] // 16, dims[0] // E.FMT[ty][2])
        for fmt, mt, kb in sorted({l["ffn_gate"] for l in by.values() if l.get("ffn_gate") == l.get("ffn_up")}):
            if fmt in AF_FFN:
                t = dataclasses.replace(TG.default_tile(fmt, "swiglu", kb), **AF_TILE, **AF_FFN[fmt], ffn=True)
                for sfx, tt in ((".af.hal", t), (".af.to.hal", dataclasses.replace(t, tout=True))):
                    split(fmt, "ffn", mt, kb, tt, "gemm_ffn_%s_%d_%d%s" % (fmt, mt, kb, sfx), NPU_SPLIT["ffn"])
    # activation encoders: per K (chunked: per chunk), row-major and fragment-major input
    encs = set()
    for site in NPU_SPLIT:
        K = NPU_SITES[site][2] * 256
        chunks = npu_chunks(site)
        chunked = len(chunks) > 1 or npu_rem(site)
        for ci, chunk in enumerate(chunks):
            encs.add((K, ci if chunked else None, chunk))
    # the ffn unpack writes down's BFP16 input for its columns (gen_npu_unpack bfp): down's chunks there need only their
    # leading k-blocks encoded ("npu_enc_17408_c<i>p")
    ffn_bfp = "ffn" in NPU_SPLIT and "down" in NPU_SPLIT
    if ffn_bfp:
        ffn_off = NPU_SITES["ffn"][1] * 16 - NPU_SPLIT["ffn"]
        assert NPU_SITES["ffn"][1] * 16 == NPU_SITES["down"][2] * 256
        for ci, (k_off, passes) in enumerate(npu_chunks("down")):
            cover = min(k_off + 1024 * passes, ffn_off) - k_off
            if 0 < cover < 1024 * passes:
                for tiled in (False, True):
                    name = "npu_enc_17408_c%dp%s.hal" % (ci, "_t" if tiled else "")
                    src = os.path.join(tmp, name[:-4] + ".loom")
                    open(src, "w").write(GE.gen("act", B, list(NPU_KS), passes, tiled=tiled, k_off=k_off, k_src=17408,
                                                kb_count=cover // 8))
                    E.emit(src, ["nop=0"], name, outdir)
                    out.append((name, B // 8 * (cover // 8) // GE.WG, GE.WG, 0))
        out.append(("npuffnbfp", ffn_off, 0, 0))
    for K, ci, (k_off, passes) in sorted(encs, key=lambda x: (x[0], -1 if x[1] is None else x[1])):
        for tiled in (False, True):
            name = "npu_enc_%d%s%s.hal" % (K, "" if ci is None else "_c%d" % ci, "_t" if tiled else "")
            src = os.path.join(tmp, name[:-4] + ".loom")
            open(src, "w").write(GE.gen("act", B, list(NPU_KS), passes, tiled=tiled, k_off=k_off,
                                        k_src=K if 1024 * passes != K else None))
            E.emit(src, ["nop=0"], name, outdir)   # emit_hal.py wants a config; the encoder has none
            out.append((name, B // 8 * (passes * sum(NPU_KS)) // GE.WG, GE.WG, 0))
    # the norms that feed K = 5120 sites also write the NPU's BFP16 input (gen_half_norm.gen_split bfp): "<norm>_bfp.hal"
    if {"qkv", "q", "ffn"} & set(NPU_SPLIT):
        for tiled, name in ((False, "norm_bfp.hal"), (True, "norm_t_bfp.hal"), ("both", "norm_rt_bfp.hal")):
            src = os.path.join(tmp, name[:-4] + ".loom")
            open(src, "w").write(gen_half_norm.gen_split(5120, wpr=4, split=NORM_SPLIT, tiled=tiled,
                                                         bfp=(B, list(NPU_KS), NPU_PASSES)))
            E.emit(src, ["yah_half_norm.rows=%d" % B, "yah_half_norm.dim=5120", "yah_half_norm.eps=1e-06",
                         "yah_half_norm.fused=0"], name, outdir)
        out.append(("npunormbfp", 1, 0, 0))
    # the DeltaNet postnorm over heads 0 .. DNSPLIT[0] - 1 (exactly ssm_out's NPU K chunk) also writes the NPU's BFP16
    # input (gen_postnorm_bfp): "postnorm[_t]_a_bfp.hal"
    if "out" in NPU_SPLIT and DNSPLIT and DNSPLIT[0] * 128 == 1024 * NPU_PASSES:
        import gen_postnorm_bfp
        for tiled, name in ((False, "postnorm_a_bfp.hal"), (True, "postnorm_t_a_bfp.hal")):
            src = os.path.join(tmp, name[:-4] + ".loom")
            open(src, "w").write(gen_postnorm_bfp.gen(B, 0, DNSPLIT[0], list(NPU_KS), NPU_PASSES, tiled, 384 in O16_MT))
            E.emit(src, ["yah_ssm_postnorm_fp16.head_count=%d" % (DNSPLIT[0] * B)], name, outdir)
        out.append(("npupostnormbfp", 1, 0, 0))
    # unpacks
    for site, n in sorted(NPU_SPLIT.items()):
        skind, mt, kb = NPU_SITES[site]
        N = mt * 16
        forms = {"qkv": [dict(out16=True)], "gate": [dict(out16=True)], "q": [dict(qg=True)],
                 "out": [dict(resid=True, parts=len(npu_chunks("out")), rem=True)],
                 "down": [dict(resid=True, parts=len(npu_chunks("down")), rem=True)],
                 "ffn": [dict(swiglu=True), dict(swiglu=True, tiled=True)]}[site]
        if site == "ffn" and ffn_bfp:
            forms = [dict(f, bfp=(NPU_KS, NPU_PASSES, 1024 * NPU_PASSES, len(npu_chunks("down")))) for f in forms]
        for f in forms:
            name = "npu_unpack_%s%s.hal" % (site, "_t" if f.get("tiled") else "")
            src = os.path.join(tmp, name[:-4] + ".loom")
            open(src, "w").write(GU.gen(B, n // GN.TN, N // 2 if site == "q" else N, N - n, **f))
            E.emit(src, ["nop=0"], name, outdir)
            out.append((name, B * n // 8 // GU.wg(n // GN.TN), GU.wg(n // GN.TN), 0))
        if site == "qkv" and conv_tb_ok(B):
            # the conv reads the NPU's v channels straight from C (gen_conv_kq.gen_c) and the GPU's from the f16 output
            # (gen climit); the unpack writes only the last 8 tokens, the conv ring that prep_ab reads
            for hal, text, cfg in (
                    ("npu_unpack_qkv_tail.hal", GU.gen(B, n // GN.TN, N, N - n, out16=True, last8=True), ["nop=0"]),
                    ("convkq_g.hal", gen_conv_kq.gen(N - n), ["yah_ssm_conv_kq.batch=%d" % B,
                                                            "yah_ssm_conv_kq.qkv_dim=10240",
                                                            "yah_ssm_conv_kq.num_key_heads=16"]),
                    ("convkq_c.hal", gen_conv_kq.gen_c(N - n, n // GN.TN, B), ["nop=0"])):
                src = os.path.join(tmp, hal[:-4] + ".loom")
                open(src, "w").write(text)
                E.emit(src, cfg, hal, outdir)
            out.append(("npu_unpack_qkv_tail.hal", n // GU.wg(n // GN.TN), GU.wg(n // GN.TN), 0))   # 8 tokens x n / 8 items
            out.append(("convkq_g.hal", gen_conv_kq.tiles(N - n), 256, 0))
            out.append(("convkq_c.hal", n // gen_conv_kq.CW, gen_conv_kq.CW, B // gen_conv_kq.TC))
        out.append(("npusplit_" + site, n, 0, 0))
    # one graph per chunk (YAH_NPU_ONEGRAPH): the GPU waits for each NPU job on a flag word (gen_npu_unpack.gen_flag_wait)
    src = os.path.join(tmp, "npu_flag_wait.loom")
    open(src, "w").write(GU.gen_flag_wait())
    E.emit(src, ["nop=0"], "npu_flag_wait.hal", outdir)
    out.append(("npu_flag_wait.hal", 1, 32, 0))
    # NPU images, one per pass count
    env = dict(hrx_paths.env(), **GN.LOOM_ENV)
    for passes in sorted({p for s in NPU_SPLIT for _, p in npu_chunks(s)}):
        K = 1024 * passes
        cfg = GN.Config(8, B // 64, NPU_KS, passes)
        src = os.path.join(tmp, "npu_gemm_%d.loom" % K)
        open(src, "w").write(GN.gen(cfg))
        r = subprocess.run([hrx_paths.LOOM_COMPILE, src, "--root=@" + cfg.entry,
                            "--target=amd.xdna.aie2p:amd.xdna.strix_halo.17f0_11",
                            "--output=" + os.path.join(outdir, "npu_gemm_%d.xdna" % K)], capture_output=True, text=True,
                           env=env)
        if r.returncode:
            raise SystemExit("NPU GEMM compile failed: " + r.stderr[-800:])
        out.append(("npubytes_%d" % K,) + tuple(GN.stream_bytes(cfg)))
    out.append(("npurows", NPU_ROWS, 0, 0))
    return out


def tile_kstore(fmt, mt, kb, B, out, outdir, kind, geom=None):
    """Emit the tile GEMM (tools/gen_gemm_tile.py) for this shape if it covers it; return its dispatch.txt row, else None.
    geom=(BM, BN, WM, WN) overrides the workgroup geometry for this one kernel."""
    import gen_gemm_tile as TG
    if fmt not in TILE_FMTS:
        return None
    t = TG.default_tile(fmt, kind, kb, geom)
    if not geom and out[:-4] in tiles()["tiles"]:
        t = dataclasses.replace(t, **tiles()["tiles"][out[:-4]])
    if mt % t.rowgrp:
        return None
    # a token tile that does not divide the chunk: the last tile is masked (gen_gemm_tile.gen masked=)
    masked = B % t.bn != 0
    return _emit_gen(lambda f, k: TG.gen(f, k, t, masked), t.bn, fmt, mt, kb, B, out, outdir, kind, t.rowgrp, masked)


def _emit_gen(gen, tile, fmt, mt, kb, B, out, outdir, kind, rowgrp, masked=False, ostride=0, sym=None, kfull=0):
    """ostride: a GEMM over the leading mt * 16 rows of an ostride-row output (gen_gemm_tile.OSTRIDE, the NPU split);
    f16 output then follows the full shape (O16_MT). sym: the kernel symbol, if not the GEMM's or the dequant's.
    kfull: a K-chunk dequant's whole-row k_blocks (gen_gemm_tile.DQ_BFP)."""
    if B % tile and not masked:
        return None
    import gen_gemm_tile as _TG
    _TG.OUT16 = kind == "kstore" and (ostride // 16 if ostride else mt) in O16_MT
    _TG.OSTRIDE = ostride
    tt = -(-B // tile)
    tmp = os.path.join(outdir, ".emit_tmp")
    os.makedirs(tmp, exist_ok=True)
    src = os.path.join(tmp, "yah_sgemm_%s_%s.loom" % (fmt, kind))
    try:
        with open(src, "w") as fh:
            fh.write(gen(fmt, kind))
    finally:
        _TG.OSTRIDE = 0
    if sym is None:
        sym = "yah_ffn_gemm_%s%s" % (fmt.replace(":", "_"), {"swiglu": "_swiglu", "kres": "_kres", "kqg": "_kqg", "ffn": "_ffn"}.get(kind, ""))
        if kind == "dequant":
            sym = "yah_dequant_" + fmt
    # Refuse before emitting if any declared operand footprint exceeds the buffer the driver binds: an overrun hangs the ring.
    import subprocess
    gate = subprocess.run([sys.executable, os.path.join(HERE, "footprint_gate.py"), src, sym, fmt,
                           kind, str(mt), str(kb), str(tt), str(B)] + (["masked"] if masked else [])
                          + (["ostride=%d" % ostride] if ostride else []) + (["kfull=%d" % kfull] if kfull else []),
                          capture_output=True, text=True)
    if gate.returncode != 0:
        raise SystemExit("footprint gate refused %s: %s" % (out, (gate.stdout + gate.stderr).strip()[-400:]))
    E.emit(src, ["%s.m_tiles=%d" % (sym, mt), "%s.k_blocks=%d" % (sym, kb), "%s.token_tiles=%d" % (sym, tt)]
           + (["%s.tokens=%d" % (sym, B)] if masked else []), out, outdir)
    return (out, tile, rowgrp, tt)


def gemm_combos(rows):
    """{(kind, fmt, port, m_tiles, k_blocks): [tensor names]} of the model's GEMM tensors; kind is kstore, residual or
    swiglu. rows: emit_prefill.parse(model)."""
    combos = {}
    for nm, dims, ty in rows:
        suffix = nm.split(".", 2)[2] if nm.startswith("blk.") else nm
        info = E.FMT.get(ty)
        if not info:
            continue
        fmt, port, qk = info
        if suffix in E.KSTORE:
            kind = "kstore"
        elif suffix in E.RESIDUAL:
            kind = "residual"
        elif suffix in E.SWIGLU:
            kind = "swiglu"
        else:
            continue
        combos.setdefault((kind, fmt, port, dims[1] // 16, dims[0] // qk), []).append(nm)
    return combos


def gemm_kinds(kind, mt):
    """HAL kinds emitted for a combo, in order; the driver runs the last one (kres over kstore, kqg over kstore)."""
    return {"kstore": ["kstore"] + (["kqg"] if mt == 768 else []), "residual": ["kstore", "kres"],
            "swiglu": ["swiglu"]}[kind]


def main():
    global MODEL
    model, outdir = sys.argv[1], sys.argv[2]
    MODEL = model
    B = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
    # GEMM tokens per workgroup of the hand-written sources
    TILE = 64
    if B % TILE:
        raise SystemExit("tokens=%d must be a multiple of the token tile=%d" % (B, TILE))
    TT = B // TILE
    # Chunked prefill: YAH_CTX=T (> B) emits every kernel at the chunk size B and the KV cache (rope, attention, V^T) at T.
    # One rope / attention HAL per chunk i (start_pos = i * B): rope_c<i>.hal / wmma_c<i>.hal; rope.hal / wmma.hal are chunk 0.
    # The driver runs T / B passes over the 64 layers. dispatch.txt row "ctx" records T.
    T = int(os.environ.get("YAH_CTX", str(B)))
    if T % B:
        raise SystemExit("YAH_CTX must be a multiple of the chunk size")
    NCH = T // B
    KC = T * 4 * 256          # KV cache elements: T rows x 4 kv heads x 256
    os.makedirs(outdir, exist_ok=True)
    rows = E.parse(model)
    combos = gemm_combos(rows)

    n = 0
    df_done = set()
    geom = []  # (<hal>, <tokens per workgroup>, <row groups>, <token_tiles>)
    for kind, fmt, port, mt, kb in sorted(combos):
        name = lambda k: "gemm_%s_%s_%d_%d.hal" % (k, fmt, mt, kb)
        # A residual projection gets a kstore and the fused-residual kres; the attention q projection (12288 rows =
        # 24 heads x [q | gate]) also gets kqg, which stores q and gate unpacked. The driver prefers kres and kqg.
        kinds = gemm_kinds(kind, mt)
        for k in kinds:
            r = gemm(fmt, mt, kb, B, name(k), outdir, kind=k)
            if r:
                geom.append(r)
                if T > B:  # chunked sets: their last chunk may be short; a one-pass set always runs all B tokens
                    geom.extend(narrow_variants(fmt, mt, kb, B, name(k), outdir, k))
                if DECODE_FREE:
                    geom.extend(decode_free(fmt, mt, kb, B, k, outdir, df_done))
                geom.extend(afrag_variants(fmt, mt, kb, B, name(k), outdir, k))
            elif k == "kstore" and fmt == "q2k":
                # Q2_K has no tile decoder; its only tensors here are the 48-row ssm_alpha / ssm_beta.
                sym = E.sym_of("yah_ffn_gemm_q2k_f32.loom")
                E.emit("yah_ffn_gemm_q2k_f32.loom", ["%s.m_tiles=%d" % (sym, mt), "%s.k_blocks=%d" % (sym, kb),
                                                     "%s.token_tiles=%d" % (sym, TT)], name(k), outdir)
                geom.append((name(k), TILE, 1, TT))
            elif k == kinds[0]:
                raise SystemExit("no GEMM kernel for %s %s (%d rows, K = %d)" % (fmt, k, mt * 16, kb * qk_of(fmt)))
        n += 1

    geom.extend(ffn_fused(rows, B, outdir))
    if NPU_SPLIT:
        geom.extend(npu_split(rows, combos, B, outdir))

    # The calibration menu (calibration_menu, for engine/model/prefill_calib.hpp) is shelved with the calibration;
    # YAH_CALIB_MENU=1 still emits it.
    if T > B and os.environ.get("YAH_CALIB_MENU") == "1":
        geom.extend(calibration_menu(model, outdir, B))

    # Attention: tools/gen_attn_fa.py, 32 tokens x 2 heads per workgroup; reads V^T (vtrans.hal, or the paged / quantized pools).
    # KV paging (256-token pages) needs a context that is a multiple of 256; otherwise the caches stay contiguous.
    kv_paged = T % 256 == 0
    if not kv_paged:
        print("KV paging off: needs the context to be a multiple of 256")
    gen_attn_fa.PAGED = gen_kvq.PAGED = kv_paged
    gen_attn_fa.MAX_TOKENS = max(B, 2048)
    # YAH_ATTN_SHAPE=<heads>x<tokens>: attention workgroup shape. 6 x 16 (a whole GQA group: each K / V tile staged once for
    # its 6 heads) is bit-identical to 2 x 32, pp2048 attention -4.9%, pp8192 even
    a_hpw, a_qt = map(int, os.environ.get("YAH_ATTN_SHAPE", "2x32").split("x"))
    gen_attn_fa.configure(a_hpw, a_qt)
    tmp = os.path.join(outdir, ".emit_tmp")
    os.makedirs(tmp, exist_ok=True)
    # kv4 / kv8 (paged): decode once per attention call instead of in every workgroup's staging (llama.cpp's prefill
    # flash attention converts quantized K / V to f16 the same way): yah_kdeq / yah_vdeq (per chunk) fill an f16 pool pair
    # with the staging decode's own arithmetic and the attention is the fp16 paged kernel (kv4: on yah_qrot's rotated Q,
    # kv8: on RoPE's f16 Q as before), bit-identical. kv4 attention -12% at 32K depth, kv8 -21%. dispatch.txt "kvdeq".
    kvdeq = kv_paged and ((gen_attn_fa.K4 and gen_attn_fa.VQ4) or (gen_attn_fa.K8 and gen_attn_fa.VQ8))
    deq_bits = 4 if gen_attn_fa.K4 else 8
    kv8deq = kvdeq and deq_bits == 8
    A = gen_attn_fa
    kv_flags = (A.K4, A.K8, A.KDEC, A.VQ4, A.VQ8)
    if kv8deq:   # the main sources are the fp16 paged kernel
        A.K8 = A.KDEC = A.VQ8 = False
    # From OCC3_FROM keys of context on, a chunk's attention runs 3 workgroups per WGP (an LDS floor of 33 KB) instead of
    # 4: past ~84K the 4-workgroup kernel falls off a cache cliff (73% -> 60% of the WMMA floor at 126K; 3 workgroups
    # hold ~70% at every depth, but lose ~5% up to 80K). Each attention source has a "_deep" twin for those chunks.
    # Each source also gets a "_alt" twin without DIAG_SWAP: Loom's allocator can land a variant in its O-accumulator
    # back-edge rotation (the 8 O tuples one placement off the P.V results: ~128 loop moves, 200+ VGPRs); the emit
    # loop re-emits such a HAL from the twin (attn_guard).
    # The deep twin is also the role-split kernel (gen_attn_fa PAIR: one wave of each pair runs QK + the softmax once,
    # the other P.V; bit-identical): at 3 workgroups per WGP it needs no more than the 256 VGPRs that allows.
    DEEP_FLAGS = dict(PAIR=True, PAIR_LATE=False, QKF=2, PVF=4)

    def write_attn(path):
        A = gen_attn_fa
        keys = ("LDS_MIN", "DIAG_SWAP") + tuple(DEEP_FLAGS) + tuple(KT2_FLAGS)
        saved = {k: getattr(A, k) for k in keys}
        twins = [(path, {}), (deep(path), dict(DEEP_FLAGS, LDS_MIN=OCC3_LDS))]
        if not A.KDEC:   # the KT2 twin (fp16 sources only)
            twins.append((kt2(path), dict(KT2_FLAGS, LDS_MIN=OCC3_LDS)))
        for name, flags in twins:
            for k, v in flags.items():
                setattr(A, k, v)
            open(name, "w").write(A.gen())
            A.DIAG_SWAP = False
            open(alt(name), "w").write(A.gen())
            for k in keys:
                setattr(A, k, saved[k])
    attn_src = os.path.join(tmp, "yah_attn_hip.loom")
    write_attn(attn_src)
    # afrag o-projections (K = 6144 kres) read the attention output fragment-major: wmma_t[_c<i>].hal
    af24 = any(h.startswith("gemm_kres_") and h.endswith("_320_24.af.hal") for h, *_ in geom)
    attn_t_src = os.path.join(tmp, "yah_attn_hip_t.loom")
    if af24:
        gen_attn_fa.TILED_OUT = True
        write_attn(attn_t_src)
        gen_attn_fa.TILED_OUT = False
    A.K4, A.K8, A.KDEC, A.VQ4, A.VQ8 = kv_flags
    vtrans_src = os.path.join(tmp, "yah_transpose_v16.loom")
    open(vtrans_src, "w").write(gen_attn_fa.gen_vtrans())
    # kv4 at depth: the Q rotation in its own kernel (gen_attn_fa QROT) and the attention on direct Q fragments, from the
    # chunk at QROT_FROM tokens on: 4 workgroups per WGP instead of 3, -4.2% attention at 8K..120K depth (the rotation's
    # 1.5 M cycles break even at ~8K). dispatch.txt "qrot <first chunk>"; the driver rotates q into its own f16 buffer.
    # kv4 under kvdeq: Q rotated from chunk 0 (its K is H256-rotated)
    qrot_first = 0 if kvdeq and deq_bits == 4 else -(-QROT_FROM // B) if gen_attn_fa.K4 and kv_paged and NCH > 1 else NCH
    attn_qr_src = os.path.join(tmp, "yah_attn_hip_qr.loom")
    attn_qr_t_src = os.path.join(tmp, "yah_attn_hip_qr_t.loom")
    qrot_src = os.path.join(tmp, "yah_qrot.loom")
    kdeq_src = os.path.join(tmp, "yah_kdeq%d.loom" % deq_bits)
    vdeq_src = os.path.join(tmp, "yah_vdeq%d.loom" % deq_bits)
    if qrot_first < NCH:
        A = gen_attn_fa
        saved = (A.Q16, A.QROT, A.VSWZ_T, A.K4, A.K8, A.KDEC, A.VQ4, A.VQ8)
        if kvdeq:   # the fp16 paged kernel (the f16 pools have the int4 pools' page layout)
            A.K4 = A.K8 = A.KDEC = A.VQ4 = A.VQ8 = False
            A.Q16 = True
        else:
            # VSWZ_T off: with direct Q the swizzled V staging tips Loom into rotating the O accumulators at the back edge
            A.Q16 = A.QROT = True
            A.VSWZ_T = False
        write_attn(attn_qr_src)
        if af24:
            A.TILED_OUT = True
            write_attn(attn_qr_t_src)
            A.TILED_OUT = False
        A.Q16, A.QROT, A.VSWZ_T, A.K4, A.K8, A.KDEC, A.VQ4, A.VQ8 = saved
        open(qrot_src, "w").write(A.gen_qrot())
        geom.append(("qrot", qrot_first, 0, 0))
    if kvdeq:
        open(kdeq_src, "w").write(gen_attn_fa.gen_kdeq(deq_bits))
        open(vdeq_src, "w").write(gen_attn_fa.gen_vdeq(deq_bits))
        geom.append(("kvdeq", 0, 0, 0))
    geom.append(("wmma.hal", a_qt, a_hpw, (B + a_qt - 1) // a_qt))
    # Quantized KV (YAH_KV, gen_kvq.kv_bits; engine/run/kvq/README.md).
    # K: int8 (yah_kq8) or H256 + asymmetric int4 (yah_kq4) after yah_kmean centres it; the attention decodes it to f16.
    kv_k, kv_v = gen_kvq.kv_bits()
    kq4_mode = kv_k == 4
    kq8_on = kv_k != 16
    if kq8_on:
        kmean_src = os.path.join(tmp, "yah_kmean.loom")
        kq8_src = os.path.join(tmp, "yah_kq8.loom")
        open(kmean_src, "w").write(gen_kvq.gen_kmean())
        open(kq8_src, "w").write(gen_kvq.gen_kq4() if kq4_mode else gen_kvq.gen_kq8())
        geom.append(("attn_kq4" if kq4_mode else "attn_kq8", 0, 0, 0))
    # V^T as bytes / nibbles per channel per 16-key tile + (S, C') by yah_vq8 / yah_vq4 instead of the f16 transpose.
    vq4_on = kv_v == 4
    # Quantized V: the decoder's open 16-key tile is seeded from the last chunk's f16 V rows (prefill ending mid-tile).
    vseed_src = os.path.join(tmp, "yah_vseed.loom")
    if kv_v != 16:
        open(vseed_src, "w").write(gen_kvq.gen_vseed())
    if vq4_on:
        vq4_src = os.path.join(tmp, "yah_vq4.loom")
        open(vq4_src, "w").write(gen_kvq.gen_vq4())
        geom.append(("attn_vq4", 0, 0, 0))
    vq8_on = kv_v == 8
    if vq8_on:
        vq8_src = os.path.join(tmp, "yah_vq8.loom")
        open(vq8_src, "w").write(gen_kvq.gen_vq8())
        geom.append(("attn_vq8", 0, 0, 0))
    # marker: the attention HAL stores f16 straight into the o-projection input
    geom.append(("attn_f16out", 0, 0, 0))
    geom.append(("vtrans.hal", 0, 0, 0))

    # DeltaNet: chunked WY Gated DeltaNet (tools/gen_gdn_chunk.py), f16 WMMA inputs.
    # Its grid (2, heads) is recorded as rowsplit.hal's row group.
    # It needs B % 32 == 0; else the recurrent kernel (tools/gen_deltanet_hip.py, same ABI and grid).
    dn_src = os.path.join(tmp, "yah_deltanet_hip_f32.loom")
    open(dn_src, "w").write(gen_gdn_chunk.gen() if B % 32 == 0 else gen_deltanet_hip.gen())
    # the conv pairs with the DeltaNet kernel: f16 normalized hand-off to the chunked one, f32 conv_out otherwise
    # the f16 hand-off conv per (256 channels, gen_conv_kq.TB tokens) workgroup: bit-identical, inputs read once
    conv_tb = conv_tb_ok(B)
    if conv_tb:
        text = gen_conv_kq.gen()
    else:
        text = open(os.path.join(E.LOOM, "yah_ssm_conv_kq_f32.loom")).read()
        if 640 in O16_MT:
            text = conv_x16(text)
        if B % 32 == 0 and gen_gdn_chunk.CONV16:
            text = conv_n16(text)
    conv_src = os.path.join(tmp, "yah_ssm_conv_kq.loom")
    open(conv_src, "w").write(text)
    prepab_src = os.path.join(tmp, "yah_deltanet_prep_ab.loom")
    text = open(os.path.join(E.LOOM, "yah_deltanet_prep_ab_f32.loom")).read()
    open(prepab_src, "w").write(prepab_x16(text) if 640 in O16_MT else text)
    postnorm_base = open(os.path.join(E.LOOM, "yah_ssm_postnorm_gate_f16.loom")).read()
    if 384 in O16_MT:
        postnorm_base = postnorm_g16(postnorm_base)
    postnorm_src = os.path.join(tmp, "yah_ssm_postnorm.loom")
    open(postnorm_src, "w").write(postnorm_base)
    geom.append(("rowsplit.hal", 0, 2, 0))
    dn_parts = []
    if B % 32 == 0 and DNSPLIT:
        ha, hb = DNSPLIT
        for tag, h0, nh in (("a", 0, ha), ("b", ha, hb)):
            src = os.path.join(tmp, f"yah_deltanet_{tag}.loom")
            open(src, "w").write(gen_gdn_chunk.gen(h0, nh))
            dn_parts.append((tag, h0, nh, src))
        geom.append(("dnsplit", 0, ha, hb))
    if conv_tb:
        geom.append(("convtb", gen_conv_kq.TB, 0, 0))

    # loom_forward_pp reads the launch geometry from dispatch.txt instead of recomputing the grid.
    # So the dispatch site and the compiled kernel cannot disagree; a mismatch is silent and wrong, not a crash.
    # Paged K / V: the f16 KV cache is only a scratch; f16 K / V go to the paged pools (RoPE K store, yah_vtpage per chunk).
    # Quantized K and V: attention never reads the f16 KV cache, so it is a one-layer, one-chunk scratch (kv16_scratch).
    if kv_paged:
        geom.append(("kv_paged", 0, 0, 0))
    kv16_scratch = kv_paged or (kq8_on and (vq4_on or vq8_on))
    if kv16_scratch:
        geom.append(("kv16_scratch", 0, 0, 0))
    paged_f16k = kv_paged and not kq8_on
    paged_f16v = kv_paged and not (vq4_on or vq8_on)
    rope_src = "yah_fused_qk_rope_batched_f32.loom"
    if paged_f16k:   # RoPE writes K rows straight into the paged pool
        rope_src = os.path.join(tmp, "yah_fused_qk_rope_batched_kpaged.loom")
        open(rope_src, "w").write(rope_kpaged(open(os.path.join(E.LOOM, "yah_fused_qk_rope_batched_f32.loom")).read()))
        geom.append(("rope_kpaged", 0, 0, 0))
    text = open(rope_src if os.path.isabs(rope_src) else os.path.join(E.LOOM, rope_src)).read()
    rope_src = os.path.join(tmp, "yah_fused_qk_rope_batched_live.loom")
    open(rope_src, "w").write(rope_live(text))
    if gen_attn_fa.Q16:   # RoPE hands the attention f16(q / 16) (not with K4: its H256 rotates the f32 Q)
        text = open(rope_src if os.path.isabs(rope_src) else os.path.join(E.LOOM, rope_src)).read()
        rope_src = os.path.join(tmp, "yah_fused_qk_rope_batched_q16.loom")
        open(rope_src, "w").write(rope_q16(text))
        geom.append(("q16", 0, 0, 0))
    if paged_f16v:
        vtpage_src = os.path.join(tmp, "yah_vtpage.loom")
        open(vtpage_src, "w").write(gen_kvq.gen_vtpage())
        vtrans_src = None    # replaced by the paged per-chunk transpose
    if NCH > 1:
        # quantized KV: K quantizers run per chunk on cache slices (kmean on chunk 0 only), V quantizers per chunk (vq*_c<i>.hal)
        geom.append(("ctx", B, 0, T))     # chunk size, total context
    # Each row over NORM_SPLIT waves (gen_half_norm.gen_split: the same per-lane chain order, so the same bits):
    # 0.383 -> 0.299 ms per call at 2048 rows (latency-bound -> ~214 GB/s). The driver reads the split from dispatch.txt.
    geom.append(("norm_split", 0, NORM_SPLIT, 0))
    with open(os.path.join(outdir, "dispatch.txt"), "w") as fh:
        for hal, tk, rg, tt in geom:
            fh.write("%s %d %d %d\n" % (hal, tk, rg, tt))

    # The output norm (tools/gen_half_norm.py): fully unrolled, all loads issued up front.
    norm_src = os.path.join(tmp, "yah_half_norm_unrolled.loom")
    # 4 rows (waves) per workgroup: 46.1 -> 41.9 ms per pp2048 vs one row per workgroup. The tuner's table may set it
    # ({"norm": {"wpr": 2}}); the driver derives the grid from the kernel's workgroup size.
    open(norm_src, "w").write(gen_half_norm.gen_split(5120, wpr=4, split=NORM_SPLIT))
    # the FFN norm for afrag GEMMs: the same values, stored fragment-major
    normt_src = os.path.join(tmp, "yah_half_norm_tiled.loom")
    open(normt_src, "w").write(gen_half_norm.gen_split(5120, wpr=4, split=NORM_SPLIT, tiled=True))
    any_af = any(h.endswith(".af.hal") for h, *_ in geom)
    # attn_norm where some of its GEMMs are afrag (DeltaNet alpha / beta never): both layouts in one pass
    normrt_src = os.path.join(tmp, "yah_half_norm_both.loom")
    open(normrt_src, "w").write(gen_half_norm.gen_split(5120, wpr=4, split=NORM_SPLIT, tiled="both"))
    # the DeltaNet postnorm, fragment-major for an afrag ssm_out
    postnorm_t_src = os.path.join(tmp, "yah_ssm_postnorm_tiled.loom")
    open(postnorm_t_src, "w").write(postnorm_tiled(postnorm_base))
    pn_parts = []
    for tag, h0, nh, _ in dn_parts:
        base = postnorm_base
        for t_, txt in (("", postnorm_heads(base, h0, nh, B)), ("_t", postnorm_tiled(postnorm_heads(base, h0, nh, B)))):
            src = os.path.join(tmp, f"yah_ssm_postnorm{t_}_{tag}.loom")
            open(src, "w").write(txt)
            pn_parts.append((f"postnorm{t_}_{tag}.hal", src, nh))
    def occ(path, c, use_kt2=True):   # the 3-workgroup twins: KT2 from KT2_FROM keys of context on, else PAIR from OCC3_FROM
        if use_kt2 and c * B >= KT2_FROM and os.path.exists(kt2(path)):
            return kt2(path)
        return deep(path) if c * B >= OCC3_FROM else path
    fixed = [
        ("yah_residual_add_1d_f32.loom", "accum.hal",
         ["yah_residual_1d.dim=%d" % (5120 * B)]),
        (norm_src, "norm.hal",
         ["yah_half_norm.rows=%d" % B, "yah_half_norm.dim=5120",
          "yah_half_norm.eps=1e-06", "yah_half_norm.fused=0"]),
        *([(normt_src, "norm_t.hal",
            ["yah_half_norm.rows=%d" % B, "yah_half_norm.dim=5120",
             "yah_half_norm.eps=1e-06", "yah_half_norm.fused=0"])] if any_af else []),
        *([(normrt_src, "norm_rt.hal",
            ["yah_half_norm.rows=%d" % B, "yah_half_norm.dim=5120",
             "yah_half_norm.eps=1e-06", "yah_half_norm.fused=0"])] if any_af else []),
        *([(postnorm_t_src, "postnorm_t.hal", ["yah_ssm_postnorm_fp16.head_count=%d" % (48 * B)])] if af24 else []),
        # the conv with the q / k L2 norm (prep_kq) fused in
        (conv_src, "convkq.hal",
         ["yah_ssm_conv_kq.batch=%d" % B, "yah_ssm_conv_kq.qkv_dim=10240",
          "yah_ssm_conv_kq.num_key_heads=16"]),
        (prepab_src, "prepab.hal",
         ["yah_deltanet_prep_ab.batch=%d" % B,
          "yah_deltanet_prep_ab.qkv_size=10240",
          "yah_deltanet_prep_ab.num_heads=48"]),
        (dn_src, "rowsplit.hal",
         ["yah_deltanet.batch=%d" % B, "yah_deltanet.qkv_size=10240",
          "yah_deltanet.inner_size=6144", "yah_deltanet.num_key_heads=16",
          "yah_deltanet.num_heads=48"]),
        (postnorm_src, "postnorm.hal",
         ["yah_ssm_postnorm_fp16.head_count=%d" % (48 * B)]),
        *[(src, f"rowsplit_{tag}.hal",
           ["yah_deltanet.batch=%d" % B, "yah_deltanet.qkv_size=10240",
            "yah_deltanet.inner_size=6144", "yah_deltanet.num_key_heads=16",
            "yah_deltanet.num_heads=48"]) for tag, _, _, src in dn_parts],
        *[(src, hal, ["yah_ssm_postnorm_fp16.head_count=%d" % (nh * B)]) for hal, src, nh in pn_parts],
        ("yah_unpack_qg_f32.loom", "unpack.hal",
         ["yah_unpack_qg.batch=%d" % B, "yah_unpack_qg.num_heads=24",
          "yah_unpack_qg.head_dim=256"]),
        *[(rope_src, "rope.hal" if c == 0 else "rope_c%d.hal" % c, [
            "yah_fused_qk_rope_batched.start_pos=%d" % (c * B),
            "yah_fused_qk_rope_batched.batch=%d" % B,
            "yah_fused_qk_rope_batched.layer_idx=0",
            "yah_fused_qk_rope_batched.max_context=%d" % T,
            "yah_fused_qk_rope_batched.num_heads=24",
            "yah_fused_qk_rope_batched.num_kv_heads=4",
            "yah_fused_qk_rope_batched.head_dim=256",
            "yah_fused_qk_rope_batched.rotary_dim=64",
            "yah_fused_qk_rope_batched.q_elems=%d" % (6144 * B),
            "yah_fused_qk_rope_batched.kv_elems=%d" % (1024 * B),
            "yah_fused_qk_rope_batched.cache32_elems=%d" % KC,
            "yah_fused_qk_rope_batched.cache16_elems=%d" % (1024 * B if kv16_scratch else KC),
            *(["yah_fused_qk_rope_batched.cache_start=%d" % (c * B)] if kv16_scratch else []),
            *(["yah_fused_qk_rope_batched.k16_elems=%d" % (1024 * T)] if paged_f16k else [])]) for c in range(NCH)],
        *([(qrot_src, "qrot.hal", ["yah_qrot.token_count=%d" % B])] if qrot_first < NCH else []),
        *[(src, "%s.hal" % stem if c == 0 else "%s_c%d.hal" % (stem, c),
           ["%s.cache_capacity=%d" % (sym, T), "%s.rows=%d" % (sym, (c + 1) * B)])
          for c in range(NCH) if kvdeq
          for src, stem, sym in ((kdeq_src, "kdeq%d" % deq_bits, "yah_kdeq%d" % deq_bits),
                                 (vdeq_src, "vdeq%d" % deq_bits, "yah_vdeq%d" % deq_bits))],
        *[(occ(attn_qr_src if c >= qrot_first else attn_src, c), "wmma.hal" if c == 0 else "wmma_c%d.hal" % c, [
            "attention_prefill.cache_capacity=%d" % T,
            "attention_prefill.token_count=%d" % B,
            "attention_prefill.start_pos=%d" % (c * B),
            "attention_prefill.num_heads=24", "attention_prefill.num_kv_heads=4",
            "attention_prefill.head_dim=256", "attention_prefill.gqa=6"]) for c in range(NCH)],
        *[(occ(attn_qr_t_src if c >= qrot_first else attn_t_src, c), "wmma_t.hal" if c == 0 else "wmma_t_c%d.hal" % c, [
            "attention_prefill.cache_capacity=%d" % T,
            "attention_prefill.token_count=%d" % B,
            "attention_prefill.start_pos=%d" % (c * B),
            "attention_prefill.num_heads=24", "attention_prefill.num_kv_heads=4",
            "attention_prefill.head_dim=256", "attention_prefill.gqa=6"]) for c in range(NCH) if af24],
        *([(vtrans_src, "vtrans.hal",
            ["yah_vtrans.token_count=%d" % T, "yah_vtrans.cache_capacity=%d" % T])]
          if vtrans_src else []),
        *([(kmean_src, "kmean.hal", ["yah_kvq.token_count=%d" % B, "yah_kvq.cache_capacity=%d" % B])]
          if kq8_on else []),
        *([(kq8_src, "kq8.hal", ["yah_kvq.token_count=%d" % B, "yah_kvq.cache_capacity=%d" % B])]
          if kq8_on and not kv_paged else []),
        *([(kq8_src, "kq8.hal" if c == 0 else "kq8_c%d.hal" % c,
            ["yah_kvq.token_count=%d" % B, "yah_kvq.cache_capacity=%d" % B,
             "yah_kvq.start_pos=%d" % (c * B), "yah_kvq.pool_rows=%d" % T]) for c in range(NCH)]
          if kq8_on and kv_paged else []),
        *([(vtpage_src, "vtpage.hal" if c == 0 else "vtpage_c%d.hal" % c,
            ["yah_kvq.token_count=%d" % B, "yah_kvq.start_pos=%d" % (c * B), "yah_kvq.pool_rows=%d" % T])
           for c in range(NCH)] if paged_f16v else []),
        *([(vq8_src, "vq8.hal" if c == 0 else "vq8_c%d.hal" % c,
            ["yah_kvq.token_count=%d" % B, "yah_kvq.cache_capacity=%d" % T, "yah_kvq.start_pos=%d" % (c * B)])
           for c in range(NCH)] if vq8_on else []),
        *([(vq4_src, "vq4.hal" if c == 0 else "vq4_c%d.hal" % c,
            ["yah_kvq.token_count=%d" % B, "yah_kvq.cache_capacity=%d" % T, "yah_kvq.start_pos=%d" % (c * B)])
           for c in range(NCH)]
          if vq4_on else []),
        *([(vseed_src, "vseed.hal", ["yah_kvq.token_count=%d" % B])] if (vq8_on or vq4_on) else []),
        ("yah_rmsnorm_f32.loom", "rmsnorm.hal",
         ["yah_rmsnorm.rows=1", "yah_rmsnorm.eps=1e-06"]),
        ("yah_gemv_q6k_f32.loom", "gemv.hal",
         ["yah_gemv_q6k.m_rows=248320", "yah_gemv_q6k.k_blocks=20"]),
        ("yah_argmax_f32.loom", "argmax.hal", ["yah_argmax.vocab=248320"]),
    ]
    # Loom low-level LICM (HRX patch 0011) for the attention kernels where it wins: it hoists the fragment loads'
    # per-lane LDS address math out of the key loop (fp16 -1..-2%; kv4 QROT -0.5% with one extra live register); the
    # kv4 staged-Q build loses 11% and the afrag GEMMs 5-20%, so they compile without it
    def attn_env(loom):
        if "_kt2" in loom:   # KT2 (fp16 sources only): hoisting capped at 6 new loop-live units (240 VGPRs)
            return {"LOOM_EXP_LICM": "1", "LOOM_EXP_LICM_MAX_LIVE": "6"}
        loom = loom.replace("_alt.loom", ".loom").replace("_deep.loom", ".loom").replace("_kt2.loom", ".loom")
        if loom in (attn_qr_src, attn_qr_t_src):   # fp16 paged under kvdeq, else the kv4 direct-Q kernel
            return {"LOOM_EXP_LICM": "1"} if kvdeq else {"LOOM_EXP_LICM": "1", "LOOM_EXP_LICM_MAX_LIVE": "1"}
        if loom in (attn_src, attn_t_src) and (not gen_attn_fa.KDEC or kv8deq):
            return {"LOOM_EXP_LICM": "1"}
        return None
    for loom, outname, configs in fixed:
        E.emit(loom, configs, outname, outdir, attn_env(loom))
        if outname.startswith("wmma") and "_kt2" in loom:   # KT2 past 240 VGPRs loses its third workgroup: use the old twin
            hal = os.path.join(outdir, outname)
            if hal_vgprs(hal) > KT2_VGPR or hal_spills(hal):
                c = int(re.search(r"_c(\d+)\.hal$", outname).group(1)) if "_c" in outname else 0
                print("attention %s: KT2 %d VGPRs, %d B spills, re-emitted from the previous twin" % (outname, hal_vgprs(hal), hal_spills(hal)))
                loom = occ(loom.replace("_kt2.loom", ".loom"), c, use_kt2=False)
                E.emit(loom, configs, outname, outdir, attn_env(loom))
        if outname.startswith("wmma"):   # attn_guard (deep twins: 3 workgroups per WGP allow 256 VGPRs, never spills)
            hal = os.path.join(outdir, outname)
            limit = 256 if "_deep" in loom else KT2_VGPR if "_kt2" in loom else ATTN_VGPR
            if hal_vgprs(hal) > limit or hal_spills(hal):
                print("attention %s: %d VGPRs, %d B spills, re-emitted without DIAG_SWAP" % (outname, hal_vgprs(hal), hal_spills(hal)))
                E.emit(alt(loom), configs, outname, outdir, attn_env(loom))
                if hal_spills(hal):
                    raise SystemExit("attention %s still spills: refusing to emit a spilling attention kernel" % outname)
        n += 1

    tables = os.path.join(E.LOOM, "tables")
    for src, dst in [("grid_iq3s.bin", "grid_iq3s.bin"),
                     ("grid_iq3xxs.bin", "grid_iq3xxs.bin"),
                     ("grid_iq2xxs.bin", "grid_iq2xxs.bin"),
                     ("grid_iq2xs.bin", "grid_iq2xs.bin"),
                     ("ksigns_iq2xs.bin", "ksigns_iq2xxs.bin")]:
        shutil.copy(os.path.join(tables, src), os.path.join(outdir, dst))
    print("emitted %d GEMM + fixed prefill HALs for B=%d (tile=%d, token_tiles=%d)"
          % (n, B, TILE, TT))


if __name__ == "__main__":
    main()
