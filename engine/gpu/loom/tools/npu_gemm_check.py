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
    # NPU_MU=1: one M slab per iteration (2-slot activation ring); NPU_FUSE=1: the GEMM cores fill their panels
    mu, fuse = int(os.environ.get("NPU_MU", "2")), int(os.environ.get("NPU_FUSE", "0"))
    if os.environ.get("NPU_GROUPS"):   # replay groups of passes, e.g. "5" (one group)
        N.GE.GROUPS = tuple(int(v) for v in os.environ["NPU_GROUPS"].split(","))
    # fuse 2, NPU_KRAW=<K> NPU_K0=<k>: rows of K columns, the call's K at column k (a site's chunk)
    kraw, k0r = int(os.environ.get("NPU_KRAW", "0")), int(os.environ.get("NPU_K0", "0"))
    cfg = N.Config(cols, nb, ks, passes, mu=mu, fuse=fuse, kraw=kraw)
    M, Nn, K = N.TM * nb, N.TN * cols, 8 * passes * sum(ks)
    os.makedirs(work, exist_ok=True)
    wq = None
    if fuse == 2:   # real IQ4_XS rows (NPU_GGUF): the panels are decoded on the cores; W's value is the decoder's
        sys.path.insert(0, os.path.expanduser("~/llama.cpp/gguf-py"))
        import gguf
        D = N.decoder("IQ4_XS")
        rd = gguf.GGUFReader(os.path.expanduser(os.environ.get(
            "NPU_GGUF", "~/Downloads/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf")))
        KR = kraw or K
        t = next(x for x in rd.tensors if x.tensor_type.name == "IQ4_XS" and len(x.shape) == 2
                 and int(x.shape[0]) == KR and int(x.shape[1]) >= Nn)
        wraw = np.asarray(t.data).view(np.uint8).reshape(-1, KR // 256, D.BLK)[:Nn].copy()
        wf = D.weights(np.ascontiguousarray(wraw[:, k0r // 256:(k0r + K) // 256])).reshape(Nn, K)   # exact f32
        frag = wf.reshape(Nn // 8, 8, K // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
        bb = D.Q.bfp_hw(frag).reshape(-1, 8, 9)                    # per row [E][8 mantissas]
        mq = bb[:, :, 1:].copy().view(np.int8).astype(np.float64)
        wq = (mq * np.exp2(bb[:, :, 0].astype(np.float64) - 133)[..., None]).reshape(
            Nn // 8, K // 8, 8, 8).transpose(0, 2, 1, 3).reshape(Nn, K)
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
    env.update(N.loom_env(cfg))
    for k in os.environ.get("NPU_ENV_DROP", "").split(","):   # e.g. LOOM_EXP_PANEL_STREAM (diagnosis)
        env.pop(k, None)
    for kv in os.environ.get("NPU_ENV_ADD", "").split(","):    # e.g. LOOM_EXP_ROUTE_RESET=1 (diagnosis)
        if "=" in kv:
            env[kv.split("=")[0]] = kv.split("=", 1)[1]
    run([hrx_paths.LOOM_COMPILE, src, f"--root=@{cfg.entry}", "--target=amd.xdna.aie2p:amd.xdna.strix_halo.17f0_11",
         f"--output={img}"], env, "loom-compile")
    a_b, w_b, c_b = (os.path.join(work, n) for n in ("a.bin", "w.bin", "c.bin"))
    sa, sw, sc = N.stream_bytes(cfg)
    a_in = a
    if fuse == 2:   # the decoders emit each pass's k-blocks in panel order (gen_npu_dec.KPERM): A follows
        kperm = np.concatenate([128 * p + D.KPERM for p in range(passes)])
        a_in = a.reshape(M, K // 8, 8)[:, kperm].reshape(M, K)
    ra, _ = B.reference(a_in, "act", M, list(ks), passes, 64, True)
    rw, _ = B.reference(w, "wgt", Nn, list(ks), passes, N.TN, True)
    assert ra.size == sa and rw.size == sw
    ra.tofile(a_b)
    rw.tofile(w_b)
    np.zeros(sc // 4, np.float32).tofile(c_b)
    co = os.path.join(work, "c_out.bin")
    base = ["--image=" + img, "--entry=" + cfg.entry, "--binding_memory=system", "--binding=" + a_b,
            "--binding=" + w_b, "--binding=" + c_b]
    if fuse:   # the egress lanes' dummy binding (never transferred), the raw stream binding
        d_b, r_b = os.path.join(work, "d.bin"), os.path.join(work, "r.bin")
        words = max(fw * nrec for c in range(cols) for fw, nrec in (N.fuse_record(cfg, cov)
                                                                  for cov in N.fuse_cover(cfg, c).values()))
        np.zeros(len(ks) * cols * words, np.int32).tofile(d_b)
        if fuse == 2:
            # the GGUF rows as they are, from the call's first column
            np.concatenate([wraw.reshape(-1)[k0r // 256 * D.BLK:], np.zeros(256, np.uint8)]).tofile(r_b)
        else:
            N.fuse_raw(np.frombuffer(rw.tobytes(), np.int32), cfg).tofile(r_b)
        base += ["--binding=" + d_b, "--binding=" + r_b]
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
        ref = value(a[:, k0:k1]) @ (value(w[:, k0:k1]) if wq is None else wq[:, k0:k1]).T
        ulp = np.exp2(np.floor(np.log2(np.maximum(np.abs(ref), 1e-30))) - 7)   # of bf16 at ref
        e = np.abs(part - ref) / ulp
        e = np.where(np.isnan(e), np.inf, e)   # (Python max drops NaN)
        err = max(err, float(e.max()))
        if err > 0.501:   # where: per column, the bad output rows (features) and tokens
            for c in range(cols):
                ec = e[:, N.TN * c:N.TN * (c + 1)]
                if ec.max() > 0.501:
                    bad = np.argwhere(ec > 0.501)
                    print(f"group {g} column {c}: {len(bad)} bad, features {np.unique(bad[:, 1] // 16)} (slabs), "
                          f"tokens {bad[:, 0].min()}..{bad[:, 0].max()}")
        k0 = k1
    got = unpack_c(raw, cols, nb, len(grp))
    exact = a.astype(np.float64) @ (w.astype(np.float64) if wq is None else wf).T
    rms = np.sqrt(np.mean((got - exact) ** 2) / np.mean(exact ** 2))
    print(f"npu gemm {M}x{Nn}x{K} ks={list(ks)} passes={passes} groups={list(grp)} (context {width} columns): max |err| "
          f"{err:.3f} bf16 ulp vs the bfp16 oracle; rel RMS {rms:.2e} vs unquantized f16")
    sys.exit(0 if err <= 0.501 else 1)   # each partial rounded once (RNE) from f32 accumulation


if __name__ == "__main__":
    main()
