#!/usr/bin/env python3
"""Check gen_npu_gemm.py on the NPU against a float64 oracle of the bfp16ebs8 math.

usage: npu_gemm_check.py <workdir> <cols> <m_blocks> <ks,ks,ks,ks> <passes> [act.f16 wgt.f16]

Operands are random (wide per-row dynamic range) or f16 files: activations [64 * m_blocks][K], weights [64 * cols][K].
They are packed with the encoder's numpy oracle (bfp16_check.reference), the image runs through iree-xdna-run, and C is
compared with the exact product of the bfp16 values; the error is f32 accumulation order only (~1e-7 relative).
NPU_REPEAT=N also times N invocations with the control-only continuation (perf pmode is the caller's job).
"""
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import bfp16_check as B  # noqa: E402
import gen_npu_gemm as N  # noqa: E402
import hrx_paths  # noqa: E402


def unpack_c(raw, cols, nb):
    """C binding [col][M block][slab mp][slab np][chain][8][8] f32 -> [64 * nb][64 * cols]."""
    c = raw.reshape(cols, nb, N.MP, N.NP, 2, 2, 8, 8)
    return c.transpose(1, 2, 4, 6, 0, 3, 5, 7).reshape(64 * nb, 64 * cols)


def value(x):
    E, m = B.quantize(x)
    rows, k = m.shape
    return (m.reshape(rows, k // 8, 8).astype(np.float64) * np.exp2(E.astype(np.float64) - 133)[..., None]).reshape(rows, k)


def run(cmd, env, tag):
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    if r.returncode:
        sys.exit(f"{tag} failed\n{r.stdout[-2000:]}{r.stderr[-2000:]}")
    return r


def main():
    if len(sys.argv) < 6:
        sys.exit(__doc__)
    work, cols, nb = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    ks, passes = tuple(int(v) for v in sys.argv[4].split(",")), int(sys.argv[5])
    cfg = N.Config(cols, nb, ks, passes)
    M, Nn, K = 64 * nb, 64 * cols, 8 * passes * sum(ks)
    os.makedirs(work, exist_ok=True)
    if len(sys.argv) > 7:
        a = np.fromfile(sys.argv[6], np.float16)[:M * K].reshape(M, K)
        w = np.fromfile(sys.argv[7], np.float16)[:Nn * K].reshape(Nn, K)
    else:
        rng = np.random.default_rng(11)
        a = (rng.standard_normal((M, K)) * np.exp(rng.uniform(-3, 3, (M, 1)))).astype(np.float16)
        w = (rng.standard_normal((Nn, K)) * 0.05).astype(np.float16)
    src = os.path.join(work, "npu_gemm.loom")
    img = os.path.join(work, "npu_gemm.xdna")
    open(src, "w").write(N.gen(cfg))
    env = hrx_paths.env()
    env.update(LOOM_EXP_LOCKED_PACK="1", LOOM_EXP_LATE_STORAGE="1")
    run([hrx_paths.LOOM_COMPILE, src, f"--root=@{cfg.entry}", "--target=amd.xdna.aie2p:amd.xdna.strix_halo.17f0_11",
         f"--output={img}"], env, "loom-compile")
    a_b, w_b, c_b = (os.path.join(work, n) for n in ("a.bin", "w.bin", "c.bin"))
    sa, sw, sc = N.stream_bytes(cfg)
    ra, _ = B.reference(a, "act", M, list(ks), passes, 64, True)
    rw, _ = B.reference(w, "wgt", Nn, list(ks), passes, 64, True)
    assert ra.size == sa and rw.size == sw
    ra.tofile(a_b)
    rw.tofile(w_b)
    np.zeros(sc // 4, np.float32).tofile(c_b)
    co = os.path.join(work, "c_out.bin")
    base = ["--image=" + img, "--entry=" + cfg.entry, "--binding_memory=system", "--binding=" + a_b,
            "--binding=" + w_b, "--binding=" + c_b]
    width = None
    for cand in range(cols, 9):   # memory-tile stages can need a wider context than the compute columns
        r = subprocess.run([hrx_paths.XDNA_RUN, f"--columns={cand}", f"--output=2={co}"] + base,
                           capture_output=True, text=True, env=env, timeout=600)
        if r.returncode == 0:
            width = cand
            break
        if "does not match the admitted native context" not in r.stderr:
            sys.exit(f"iree-xdna-run failed\n{r.stderr[-2000:]}")
    if width is None:
        sys.exit("no context width admitted the image")
    got = unpack_c(np.fromfile(co, np.float32), cols, nb)
    ref = value(a) @ value(w).T
    err = np.abs(got - ref).max() / np.abs(ref).max()
    exact = a.astype(np.float64) @ w.astype(np.float64).T
    rms = np.sqrt(np.mean((got - exact) ** 2) / np.mean(exact ** 2))
    print(f"npu gemm {M}x{Nn}x{K} ks={list(ks)} passes={passes} (context {width} columns): max |err| / max |ref| "
          f"{err:.2e} vs bfp16 oracle; rel RMS {rms:.2e} vs unquantized f16")
    reps = int(os.environ.get("NPU_REPEAT", "0"))
    if reps:
        def timed(n):
            t0 = time.perf_counter()
            run([hrx_paths.XDNA_RUN, f"--columns={width}", "--repeat_invocation", f"--invocation_count={n}"] + base,
                env, "timing")
            return time.perf_counter() - t0
        t1, t2 = min(timed(2) for _ in range(3)), min(timed(2 + reps) for _ in range(3))
        us = (t2 - t1) / reps * 1e6
        print(f"  {us:.1f} us per repeat invocation, {M * Nn * K / us / 1e6:.2f} TMAC/s")
    sys.exit(0 if err < 1e-5 else 1)


if __name__ == "__main__":
    main()
