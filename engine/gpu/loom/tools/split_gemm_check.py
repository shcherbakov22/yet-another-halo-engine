#!/usr/bin/env python3
"""Check the GPU half of an NPU column split: a kstore / kres over rows [0, N - npu_rows) at the full output stride.

usage: split_gemm_check.py <model.gguf> <workdir> <tensor> <npu_rows> <act.f16> [plain|af|afp|afto] [kstore|kres|swiglu|ffn|kqg]

Runs the set's GEMM for <tensor> over all N rows, then the split variant (gen_gemm_tile.OSTRIDE = N) into an output
filled with NaN. The split variant's rows must equal the full GEMM bit for bit and the NPU's trailing rows of every
token must stay NaN. act.f16 is a [B][K] GEMM input (e.g. a YAH_DUMP_ACT dump).
af: the afrag form with emit_prefill_pp's knobs for the full shape (input fragment-major); afp: its persistent kres
(gen_kres_persist, grid y = 1). The kstore output is f16 when the full shape is in emit_prefill_pp.O16_MT, as in the
prefill set. kres: out = resid + W x, with a random f32 residual. swiglu (tensor: ffn_up): out = f16(silu(gate) * W x)
with a random f32 gate; ffn (tensor: ffn_gate): the fused gate + up GEMM (emit_prefill_pp.AF_FFN), its weights the gate
tensor then the up tensor. afto: the afrag form with fragment-major output (the afrag down projection's input).
kqg (tensor: attn_q, rows = heads x [256 q | 256 gate]): q and gate outputs, whole heads split (npu_rows / 512 heads).
Needs PYTHONPATH with llama.cpp's gguf-py.
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dataclasses  # noqa: E402
import dq_bfp16_check as DQ  # noqa: E402
import gen_kres_persist  # noqa: E402
import emit_prefill_pp as EP  # noqa: E402
import gen_gemm_tile as TG  # noqa: E402

EMIT = DQ.EMIT
HALRUN, GPURUN = DQ.HALRUN, DQ.GPURUN


def tile(fmt, kind, kb, n, mode):
    """The prefill set's tile for an n-row matrix: the afrag form with its emit_prefill_pp.AF knobs, or the default.
    Returns (tile, persist knobs or None)."""
    if mode == "plain":
        return TG.default_tile(fmt, kind, kb), None
    if kind == "ffn":
        t = dataclasses.replace(TG.default_tile(fmt, "swiglu", kb), **EP.AF_TILE, **EP.AF_FFN[fmt], ffn=True)
        return dataclasses.replace(t, tout=mode == "afto"), None
    knobs = dict(EP.AF[(fmt, kind, n // 16, kb)])
    persist = knobs.pop("persist", None)
    t = dataclasses.replace(TG.default_tile(fmt, kind, kb), **EP.AF_TILE, **knobs, tout=mode == "afto")
    return t, (dataclasses.replace(t, **persist) if isinstance(persist, dict) else t) if persist else None


def emit(work, tag, fmt, kind, t, mt, kb, tt, ostride, out16=False, persist=None, ntiles=0):
    TG.OSTRIDE, TG.OUT16 = ostride, out16
    try:
        if persist is not None:
            text = gen_kres_persist.persist(TG.gen(fmt, "kres", persist, False),
                                            TG.gen(fmt, "kstore", dataclasses.replace(persist, respre=0), False), ntiles)
        else:
            text = TG.gen(fmt, kind, t)
    finally:
        TG.OSTRIDE, TG.OUT16 = 0, False
    sym = "yah_ffn_gemm_" + fmt + {"kres": "_kres", "swiglu": "_swiglu", "ffn": "_ffn", "kqg": "_kqg"}.get(kind, "")
    src = os.path.join(work, tag + ".loom")
    open(src, "w").write(text)
    r = subprocess.run([sys.executable, EMIT, src, os.path.join(work, tag), f"{sym}.m_tiles={mt}", f"{sym}.k_blocks={kb}",
                        f"{sym}.token_tiles={tt}"], capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"{tag}: emit failed\n{r.stdout[-3000:]}{r.stderr[-3000:]}")
    return r.stdout.strip().splitlines()[0]


def main():
    if len(sys.argv) < 6:
        sys.exit(__doc__)
    import gguf
    model, work, name, nn, act = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
    os.makedirs(work, exist_ok=True)
    rd = gguf.GGUFReader(model)
    tn = next(x for x in rd.tensors if x.name == name)
    fmt = DQ.FMT[tn.tensor_type.name]
    K, N = int(tn.shape[0]), int(tn.shape[1])
    x = np.fromfile(act, np.float16)
    B = x.size // K
    kb = K // 256 * TG.G.FMTS[fmt].get("kdiv", 1)
    mode = sys.argv[6] if len(sys.argv) > 6 else "plain"
    kind = sys.argv[7] if len(sys.argv) > 7 else "kstore"
    af = mode != "plain"
    t, pt = tile(fmt, kind, kb, N, mode)
    pt = pt if mode == "afp" else None
    out16 = (kind == "kstore" and N // 16 in EP.O16_MT) or kind in ("swiglu", "ffn")
    tout = mode == "afto"
    wfile = None
    if kind == "ffn":   # one weight binding: the gate tensor, then the up tensor
        up = next(x for x in rd.tensors if x.name == name.replace("ffn_gate", "ffn_up"))
        wfile = os.path.join(work, "w_ffn.bin")
        np.concatenate([np.asarray(tn.data).reshape(-1), np.asarray(up.data).reshape(-1)]).tofile(wfile)
    gfile = os.path.join(work, "gate.f32")
    if kind == "swiglu":
        (np.random.default_rng(6).standard_normal(B * N) * 2).astype(np.float32).tofile(gfile)
    mt, mtg = N // 16, (N - nn) // 16
    assert nn % 16 == 0 and mtg % t.rowgrp == 0 and B % t.bn == 0
    tt = B // t.bn
    full = emit(work, "full", fmt, kind, t, mt, kb, tt, 0, out16, pt, tt)
    split = emit(work, "split", fmt, kind, t, mtg, kb, tt, N, out16, pt, tt)
    gy = 1 if pt else tt
    resid = os.path.join(work, "resid.f32")
    if kind == "kres":
        np.random.default_rng(5).standard_normal(B * N).astype(np.float32).tofile(resid)
    if af:   # fragment-major input: [token / 16][k / 16][token % 16][k % 16]
        act_t = os.path.join(work, "act_t.f16")
        x.reshape(B // 16, 16, K // 16, 16).transpose(0, 2, 1, 3).tofile(act_t)
        act = act_t
    esz = 2 if out16 else 4
    tables = [DQ.table_file(fmt, e) for e in TG.G.FMTS[fmt]["extra"]]
    wst, ost = 17408 * 16 * 2, 17408 * B * 4   # the driver's wstage_ / ostage_
    env = dict(os.environ, HAL_RUN_ITERS="1")

    qg = kind == "kqg"
    obytes = N // 2 * B * 4 if qg else N * B * esz   # kqg: q and gate, f32 [B][N / 2] each

    def run(hal, gx, out_spec, gate_spec=None):
        rs = [B * N * 4] if kind in ("kres", "swiglu") else []
        mins = ",".join(str(v) for v in [0] + [os.path.getsize(f) for f in tables] + [B * K * 2] + rs + [wst, ost, obytes]
                        + ([obytes] if qg else []))
        cmd = [GPURUN, "split-gemm", "--", HALRUN, model, hal, f"{gx},{gy}", str(t.lanes), mins,
               f"f:{wfile}" if wfile else f"t:{name}"]
        cmd += [f"f:{f}" for f in tables] + [f"f:{act}"] + ([f"f:{resid}"] if kind == "kres" else [])
        cmd += [f"f:{gfile}"] if kind == "swiglu" else []
        cmd += [f"o:{wst}:/dev/null", f"o:{ost}:/dev/null", out_spec] + ([gate_spec] if qg else [])
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
        if "hal_run: ok" not in r.stdout:
            sys.exit(f"hal_run failed\n{r.stdout[-2000:]}{r.stderr[-2000:]}")

    yf, ys = os.path.join(work, "y_full.f32"), os.path.join(work, "y_split.f32")
    dt, it = (np.float16, np.uint16) if out16 else (np.float32, np.uint32)
    gf, gs = os.path.join(work, "g_full.f32"), os.path.join(work, "g_split.f32")
    sentinel = os.path.join(work, "y_nan.f32")
    np.full(obytes // np.dtype(dt).itemsize, np.nan, dt).tofile(sentinel)
    run(full, mt // t.rowgrp, f"o:{obytes}:{yf}", f"o:{obytes}:{gf}")
    run(split, mtg // t.rowgrp, f"io:{sentinel}:{ys}", f"io:{sentinel}:{gs}")
    if qg:   # q and gate side by side: [B][N / 2] each -> [B][N] with q in the first half
        pair = lambda a, b: np.concatenate([np.fromfile(a, dt).reshape(B, N // 2), np.fromfile(b, dt).reshape(B, N // 2)], 1)
        yfull, ysplit = pair(yf, gf), pair(ys, gs)
        h = (N - nn) // 2   # the GPU's columns in each of q and gate
        gpu_same = all(np.array_equal(yfull[:, o:o + h].view(it), ysplit[:, o:o + h].view(it)) for o in (0, N // 2))
        npu_kept = all(bool(np.isnan(ysplit[:, o + h:o + N // 2]).all()) for o in (0, N // 2))
        print(f"{name} ({fmt} kqg{' ' + mode if af else ''}) N={N} K={K} B={B}: GPU heads [0, {(N - nn) // 512}) "
              f"{'bit-identical' if gpu_same else 'DIFFER'}; NPU heads {'untouched' if npu_kept else 'WRITTEN'}")
        for f in (yf, ys, gf, gs, sentinel) + ((act,) if af else ()):
            os.remove(f)
        sys.exit(0 if gpu_same and npu_kept else 1)
    def rows_major(y):   # fragment-major [token / 16][row / 16][16][16] -> [token][row]
        return y.reshape(B // 16, N // 16, 16, 16).transpose(0, 2, 1, 3).reshape(B, N) if tout else y.reshape(B, N)
    yfull = rows_major(np.fromfile(yf, dt))
    ysplit = rows_major(np.fromfile(ys, dt))
    gpu_same = np.array_equal(yfull[:, :N - nn].view(it), ysplit[:, :N - nn].view(it))
    npu_kept = bool(np.isnan(ysplit[:, N - nn:]).all())
    print(f"{name} ({fmt} {kind}{' ' + mode if af else ''}{', f16 out' if out16 else ''}) N={N} K={K} B={B}: GPU rows [0, {N - nn}) {'bit-identical' if gpu_same else 'DIFFER'} to the"
          f" full GEMM; NPU rows [{N - nn}, {N}) {'untouched' if npu_kept else 'WRITTEN'}")
    for f in ((yf, ys, sentinel) + ((act,) if af else ()) + ((resid,) if kind == "kres" else ())
              + ((wfile,) if wfile else ()) + ((gfile,) if kind == "swiglu" else ())):
        os.remove(f)
    sys.exit(0 if gpu_same and npu_kept else 1)


if __name__ == "__main__":
    main()
