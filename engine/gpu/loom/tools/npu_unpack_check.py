#!/usr/bin/env python3
"""Check gen_npu_unpack.py against npu_gemm_check.unpack_c, bit for bit.

usage: npu_unpack_check.py <model.gguf> <workdir> <tokens> <cols> <stride> <off> [out16]

Random C into an output filled with NaN: columns [off, off + 64 * cols) must hold C exactly, all others stay NaN.
UNP_PARTS=n (partial Cs summed), UNP_RESID=1 (out = resid + C), UNP_SWIGLU=1 (f16(silu(gate) * up), checked to f16
rounding: the kernel's exp / reciprocal are not numpy's), UNP_TILED=1 (fragment-major f16 output), UNP_QG=1 (attention q
rows: <stride> is the q / gate row count, <off> the first NPU row of heads x [256 q | 256 gate]).
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bfp16_check as B  # noqa: E402
import gen_npu_unpack as G  # noqa: E402
import npu_gemm_check as N  # noqa: E402


def main():
    if len(sys.argv) < 7:
        sys.exit(__doc__)
    model, work = sys.argv[1], sys.argv[2]
    tokens, cols, stride, off = (int(v) for v in sys.argv[3:7])
    parts = int(os.environ.get("UNP_PARTS", "1"))
    resid, swiglu, tiled, qg = (os.environ.get(k) == "1" for k in ("UNP_RESID", "UNP_SWIGLU", "UNP_TILED", "UNP_QG"))
    out16 = (len(sys.argv) > 7 and sys.argv[7] != "0") or swiglu or tiled
    dt = np.float16 if out16 else np.float32
    nblk = parts * (2 if swiglu else 1)
    os.makedirs(work, exist_ok=True)
    c = np.random.default_rng(3).standard_normal(nblk * tokens * cols * 64).astype(np.float32) * 3
    cf, of, sf = (os.path.join(work, n) for n in ("c.f32", "out.f32", "nan.f32"))
    c.tofile(cf)
    np.full(tokens * stride, np.nan, dt).tofile(sf)
    src = os.path.join(work, "unpack.loom")
    open(src, "w").write(G.gen(tokens, cols, stride, off, out16 and not (swiglu or tiled), resid, parts, swiglu, tiled, qg))
    r = subprocess.run([sys.executable, B.EMIT, src, os.path.join(work, "unpack"), "nop=0"], capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"emit failed\n{r.stdout[-3000:]}{r.stderr[-3000:]}")
    hal = r.stdout.strip().splitlines()[0]
    rf = os.path.join(work, "resid.f32")
    r_in = np.random.default_rng(4).standard_normal(tokens * stride).astype(np.float32)
    if resid:
        r_in.tofile(rf)
    obytes = tokens * stride * np.dtype(dt).itemsize
    cmd = [B.GPURUN, "npu-unpack", "--", B.HALRUN, model, hal, str(tokens * cols * 8 // G.WG), str(G.WG),
           ",".join(str(v) for v in [c.nbytes] + ([obytes] if resid else []) + [obytes]), f"f:{cf}"]
    gfo = os.path.join(work, "gate_out.f32")
    cmd += ([f"f:{rf}"] if resid else []) + [f"io:{sf}:{of}"] + ([f"io:{sf}:{gfo}"] if qg else [])
    if qg:
        cmd[cmd.index(f"{c.nbytes},{obytes}")] = f"{c.nbytes},{obytes},{obytes}"
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=dict(os.environ, HAL_RUN_ITERS="4"))
    if "hal_run: ok" not in r.stdout:
        sys.exit(f"hal_run failed\n{r.stdout[-2000:]}{r.stderr[-2000:]}")
    print(next(ln for ln in r.stdout.splitlines() if "per dispatch" in ln))
    if qg:   # rows off .. off + 64 cols of heads x [256 q | 256 gate] -> q / gate [tokens][stride]
        if "hal_run: ok" not in r.stdout:
            sys.exit(f"hal_run failed\n{r.stdout[-2000:]}{r.stderr[-2000:]}")
        q, g = np.fromfile(of, np.float32).reshape(tokens, stride), np.fromfile(gfo, np.float32).reshape(tokens, stride)
        want = N.unpack_c(c, cols, tokens // 64)
        exp_q = np.full((tokens, stride), np.nan, np.float32)
        exp_g = exp_q.copy()
        for j in range(64 * cols):
            row = off + j
            h, w = divmod(row, 512)
            (exp_q if w < 256 else exp_g)[:, h * 256 + w % 256] = want[:, j]
        ok = all(np.array_equal(a.view(np.uint32), b.view(np.uint32)) for a, b in ((q, exp_q), (g, exp_g)))
        print(f"unpack qg tokens={tokens} cols={cols} rows {off}..{off + 64 * cols} of 2 x {stride}: "
              f"{'exact, other columns untouched' if ok else 'DIFFERS'}")
        for f in (cf, of, sf, gfo):
            os.remove(f)
        sys.exit(0 if ok else 1)
    got = np.fromfile(of, dt)
    if tiled:   # back to row-major
        got = got.reshape(tokens // 16, stride // 16, 16, 16).transpose(0, 2, 1, 3)
    got = got.reshape(tokens, stride)
    pe = tokens * cols * 64
    blk = [N.unpack_c(c[i * pe:(i + 1) * pe], cols, tokens // 64) for i in range(nblk)]
    if swiglu:
        g, u = blk[0].astype(np.float32), blk[1].astype(np.float32)
        want = (g * (np.float32(1) / (np.float32(1) + np.exp(-g))) * u).astype(np.float32)
    else:
        want = blk[0]
        for b in blk[1:]:
            want = want + b
        if resid:
            want = r_in.reshape(tokens, stride)[:, off:off + 64 * cols] + want
    want = want.astype(dt)
    sl = got[:, off:off + 64 * cols]
    if swiglu:   # within one f16 ulp
        ulp = np.abs(sl.view(np.int16).astype(np.int32) - want.view(np.int16).astype(np.int32))
        ok = bool(np.isfinite(sl).all() and ulp.max() <= 1)
        print(f"  swiglu: max {ulp.max()} f16 ulp, {np.count_nonzero(ulp)} of {ulp.size} differ by 1")
    else:
        ok = np.array_equal(sl.view(np.uint16 if out16 else np.uint32), want.view(np.uint16 if out16 else np.uint32))
    rest = np.delete(got, np.s_[off:off + 64 * cols], axis=1)
    kept = bool(np.isnan(rest).all())
    mode = "".join(f" {k}" for k, v in (("f16", out16), ("resid", resid), ("swiglu", swiglu), ("tiled", tiled)) if v)
    print(f"unpack tokens={tokens} cols={cols} stride={stride} off={off}{mode}{f' parts={parts}' if parts > 1 else ''}: slice {'exact' if ok else 'DIFFERS'}, "
          f"other columns {'untouched' if kept else 'WRITTEN'}")
    for f in (cf, of, sf) + ((rf,) if resid else ()):
        os.remove(f)
    sys.exit(0 if ok and kept else 1)


if __name__ == "__main__":
    main()
