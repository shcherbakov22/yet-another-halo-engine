#!/usr/bin/env python3
"""One kstore GEMM split between the GPU and the NPU, end to end through engine/build/npu_split_run.

usage: npu_split_check.py <model.gguf> <workdir> <tensor> <npu_rows> <act.f16> [rounds]

Emits the GPU kernels (activation encoder, the NPU's rows decoded to BFP16, the split kstore, the C unpack)
and the NPU image, runs them on one stream with the NPU relay, and checks the output: the GPU's rows bit-identical to
the full kstore GEMM, the NPU's rows against the float64 product of the bfp16 operands (f32 accumulation order only).
npu_rows is a multiple of 8 * gen_npu_gemm.TN (one NPU call per 640 output features). Needs PYTHONPATH with llama.cpp's gguf-py.
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import bfp16_check as B  # noqa: E402
import dq_bfp16_check as DQ  # noqa: E402
import gen_bfp16_encode as GE  # noqa: E402
import gen_gemm_tile as TG  # noqa: E402
import gen_npu_gemm as GN  # noqa: E402
import gen_npu_unpack as GU  # noqa: E402
import hrx_paths  # noqa: E402
import npu_gemm_check as NC  # noqa: E402
import split_gemm_check as SC  # noqa: E402

KS = GN.KS   # k-blocks per pass and row; K = 1024 * passes
ROWS = 8 * GN.TN   # output rows per NPU call


def emit_plain(work, tag, text):
    src = os.path.join(work, tag + ".loom")
    open(src, "w").write(text)
    r = subprocess.run([sys.executable, B.EMIT, src, os.path.join(work, tag), "nop=0"], capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"{tag}: emit failed\n{r.stdout[-3000:]}{r.stderr[-3000:]}")
    return r.stdout.strip().splitlines()[0]


def main():
    if len(sys.argv) < 6:
        sys.exit(__doc__)
    import gguf
    from gguf import quants
    model, work, name, nn, act = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
    iters = int(sys.argv[6]) if len(sys.argv) > 6 else 9
    os.makedirs(work, exist_ok=True)
    rd = gguf.GGUFReader(model)
    tn = next(x for x in rd.tensors if x.name == name)
    fmt = DQ.FMT[tn.tensor_type.name]
    K, N = int(tn.shape[0]), int(tn.shape[1])
    passes = K // 1024
    assert K == 8 * passes * sum(KS) and nn % ROWS == 0
    x = np.fromfile(act, np.float16)
    tokens = x.size // K
    x = x.reshape(tokens, K)
    panels, nb = nn // ROWS, tokens // 64
    kb = K // 256 * TG.G.FMTS[fmt].get("kdiv", 1)
    # GPU kernels
    t = TG.default_tile(fmt, "kstore", kb)
    tt = tokens // t.bn
    full = SC.emit(work, "full", fmt, "kstore", t, N // 16, kb, tt, 0)
    split = SC.emit(work, "split", fmt, "kstore", t, (N - nn) // 16, kb, tt, N)
    enc_act = emit_plain(work, "enc_act", GE.gen("act", tokens, list(KS), passes))
    unpack = emit_plain(work, "unpack", GU.gen(tokens, nn // GN.TN, N, N - nn))
    # the NPU's rows decoded straight into its BFP16 weight stream (gen_gemm_tile.DQ_BFP)
    dt = TG.dataclasses.replace(TG.default_tile(fmt, "kstore", kb), bm=64, wm=2, wn=4, decahead=False, ksub=64,
                                dbuf=False)
    sym = "yah_dequant_" + fmt + "_bfp16"
    src = os.path.join(work, "dq.loom")
    TG.DQ_BFP = (KS, passes)
    try:
        open(src, "w").write(TG.gen(fmt, "dequant", dt))
    finally:
        TG.DQ_BFP = None
    r = subprocess.run([sys.executable, B.EMIT, src, os.path.join(work, "dq"), f"{sym}.m_tiles={nn // 16}",
                        f"{sym}.k_blocks={kb}", f"{sym}.token_tiles=1"], capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"dequant emit failed\n{r.stdout[-3000:]}{r.stderr[-3000:]}")
    dq = r.stdout.strip().splitlines()[0]
    # NPU image: one ROWS-wide panel (8 columns) per call
    cfg = GN.Config(8, nb, KS, passes)
    xsrc, ximg = os.path.join(work, "npu.loom"), os.path.join(work, "npu.xdna")
    open(xsrc, "w").write(GN.gen(cfg))
    env = hrx_paths.env()
    env.update(LOOM_EXP_LOCKED_PACK="1", LOOM_EXP_LATE_STORAGE="1")
    r = subprocess.run([hrx_paths.LOOM_COMPILE, xsrc, f"--root=@{cfg.entry}",
                        "--target=amd.xdna.aie2p:amd.xdna.strix_halo.17f0_11", f"--output={ximg}"],
                       capture_output=True, text=True, env=env)
    if r.returncode:
        sys.exit(f"loom-compile failed\n{r.stderr[-3000:]}")
    a_bytes, w_panel, c_panel = GN.stream_bytes(cfg)
    tables = [DQ.table_file(fmt, e) for e in TG.G.FMTS[fmt]["extra"]]
    plan = {"tensor": name, "tokens": tokens, "n": N, "k": K, "npu_rows": nn, "panels": panels,
            "a_bytes": a_bytes, "w_panel_bytes": w_panel, "c_panel_bytes": c_panel,
            "split_hal": split, "split_gx": (N - nn) // 16 // t.rowgrp, "full_hal": full, "full_gx": N // 16 // t.rowgrp, "split_gy": tt, "split_wg": t.lanes,
            "enc_act_hal": enc_act, "enc_act_wgs": tokens // 8 * (K // 8) // GE.WG,
            "dq_hal": dq, "dq_wgs": TG.dq_wgs(nn // 16, kb, dt, fmt), "dq_wg": dt.lanes,
            "unpack_hal": unpack, "unpack_wgs": tokens * nn // 8 // GU.WG,
            "npu_xdna": ximg, "npu_entry": cfg.entry, "npu_columns": 8, "tables": ",".join(tables), "iters": iters, "warmup_ms": 4000}
    pf = os.path.join(work, "plan.txt")
    open(pf, "w").write("".join(f"{k}={v}\n" for k, v in plan.items()))
    out = os.path.join(work, "y_npu.f32")
    r = subprocess.run([B.GPURUN, "npu-split", "--", os.path.join(hrx_paths.ROOT, "engine", "build", "npu_split_run"),
                        model, pf, act, out], capture_output=True, text=True, timeout=900, env=env)
    print("\n".join(ln for ln in r.stdout.splitlines() if " us" in ln))
    if "npu_split_run: ok" not in r.stdout:
        sys.exit(f"npu_split_run failed\n{r.stdout[-2000:]}{r.stderr[-2000:]}")
    # reference: the full kstore GEMM for the GPU's rows
    yf = os.path.join(work, "y_full.f32")
    wst, ost = 17408 * 16 * 2, 17408 * tokens * 4
    mins = ",".join(str(v) for v in [0] + [os.path.getsize(f) for f in tables] + [tokens * K * 2, wst, ost, N * tokens * 4])
    cmd = [B.GPURUN, "npu-split", "--", B.HALRUN, model, full, f"{N // 16 // t.rowgrp},{tt}", str(t.lanes), mins,
           f"t:{name}"] + [f"f:{f}" for f in tables] + [f"f:{act}", f"o:{wst}:/dev/null", f"o:{ost}:/dev/null",
                                                        f"o:{N * tokens * 4}:{yf}"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=dict(os.environ, HAL_RUN_ITERS="1"))
    if "hal_run: ok" not in r.stdout:
        sys.exit(f"hal_run failed\n{r.stdout[-2000:]}{r.stderr[-2000:]}")
    y = np.fromfile(out, np.float32).reshape(tokens, N)
    yfull = np.fromfile(yf, np.float32).reshape(tokens, N)
    gpu_same = np.array_equal(y[:, :N - nn].view(np.uint32), yfull[:, :N - nn].view(np.uint32))
    w = quants.dequantize(np.ascontiguousarray(np.asarray(tn.data)[N - nn:]), tn.tensor_type).astype(np.float16)
    ref = NC.value(x) @ NC.value(w).T
    got = y[:, N - nn:]
    err = np.abs(got - ref).max() / np.abs(ref).max()
    rms = np.sqrt(np.mean((got - yfull[:, N - nn:]) ** 2) / np.mean(yfull[:, N - nn:].astype(np.float64) ** 2))
    print(f"{name} ({fmt}) {tokens}x{N}x{K}, NPU rows {nn} ({panels} calls): GPU rows "
          f"{'bit-identical' if gpu_same else 'DIFFER'}; NPU rows max |err| / max |ref| {err:.2e} vs the bfp16 oracle, "
          f"rel RMS {rms:.2e} vs the GPU GEMM")
    for f in (out, yf):
        os.remove(f)
    sys.exit(0 if gpu_same and err < 1e-5 else 1)


if __name__ == "__main__":
    main()
