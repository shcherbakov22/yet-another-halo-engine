#!/usr/bin/env python3
"""Check gen_npu_gemm.py on the NPU against a float64 oracle of the bfp16ebs8 math.

usage: npu_gemm_check.py <workdir> <cols> <m_blocks> <ks,ks,ks,ks> <passes> [act.f16 wgt.f16]

Operands are random (wide per-row dynamic range) or f16 files: activations [64 * m_blocks][K], weights [TN * cols][K] (gen_npu_gemm.TN).
They are packed with the encoder's numpy oracle (bfp16_check.reference), the image runs through iree-xdna-run, and each C
partial (one per replay group of passes, gen_npu_gemm.groups) is compared with the exact product of its K range of the
bfp16 values; C is bf16: within half an ulp of the exact product (f32 accumulation order is ~1e-7 below that).
Time the NPU with the counter profiler, not wall clock.
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import bfp16_check as B  # noqa: E402
import gen_npu_gemm as N  # noqa: E402
import hrx_paths  # noqa: E402


def unpack_c(raw, cols, nb, groups=1, group=None):
    """C binding [col][group][M block][slab mp][slab np][chain][8][8] (as f32) -> [TM * nb][TN * cols]: the sum of the
    groups' partials in order (as the GPU unpack adds them), or partial group only."""
    c = raw.reshape(cols, groups, nb, N.MP, N.NP, 2, 2, 8, 8)
    if group is not None:
        c = c[:, group]
    else:
        acc = c[:, 0]
        for g in range(1, groups):
            acc = acc + c[:, g]
        c = acc
    return c.transpose(1, 2, 4, 6, 0, 3, 5, 7).reshape(N.TM * nb, N.TN * cols)


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
    cfg = N.Config(cols, nb, ks, passes, mu=int(os.environ.get("NPU_MU", "2")))   # NPU_MU=1: one M slab per iteration
    M, Nn, K = N.TM * nb, N.TN * cols, 8 * passes * sum(ks)
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
    env.update(N.LOOM_ENV)
    run([hrx_paths.LOOM_COMPILE, src, f"--root=@{cfg.entry}", "--target=amd.xdna.aie2p:amd.xdna.strix_halo.17f0_11",
         f"--output={img}"], env, "loom-compile")
    a_b, w_b, c_b = (os.path.join(work, n) for n in ("a.bin", "w.bin", "c.bin"))
    sa, sw, sc = N.stream_bytes(cfg)
    ra, _ = B.reference(a, "act", M, list(ks), passes, 64, True)
    rw, _ = B.reference(w, "wgt", Nn, list(ks), passes, N.TN, True)
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
    raw = (np.fromfile(co, np.uint16).astype(np.uint32) << 16).view(np.float32)   # bf16 C
    grp = N.groups(cfg) or (passes,)
    err, k0 = 0.0, 0
    for g, gp in enumerate(grp):   # each group's partial: its K range
        k1 = k0 + 8 * gp * sum(ks)
        part = unpack_c(raw, cols, nb, len(grp), g)
        ref = value(a[:, k0:k1]) @ value(w[:, k0:k1]).T
        ulp = np.exp2(np.floor(np.log2(np.maximum(np.abs(ref), 1e-30))) - 7)   # of bf16 at ref
        err = max(err, (np.abs(part - ref) / ulp).max())
        k0 = k1
    got = unpack_c(raw, cols, nb, len(grp))
    exact = a.astype(np.float64) @ w.astype(np.float64).T
    rms = np.sqrt(np.mean((got - exact) ** 2) / np.mean(exact ** 2))
    print(f"npu gemm {M}x{Nn}x{K} ks={list(ks)} passes={passes} groups={list(grp)} (context {width} columns): max |err| "
          f"{err:.3f} bf16 ulp vs the bfp16 oracle; rel RMS {rms:.2e} vs unquantized f16")
    sys.exit(0 if err <= 0.501 else 1)   # each partial rounded once (RNE) from f32 accumulation


if __name__ == "__main__":
    main()
