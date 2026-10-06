#!/usr/bin/env python3
"""Check the GPU half of an NPU column split: kstore over rows [0, N - npu_rows) at the full output stride.

usage: split_gemm_check.py <model.gguf> <workdir> <tensor> <npu_rows> <act.f16>

Runs the set's kstore GEMM for <tensor> over all N rows, then the split variant (gen_gemm_tile.OSTRIDE = N) into an
output filled with NaN. The split variant's rows must equal the full GEMM bit for bit and the NPU's trailing rows of
every token must stay NaN. act.f16 is a [B][K] GEMM input (e.g. a YAH_DUMP_ACT dump).
Needs PYTHONPATH with llama.cpp's gguf-py.
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dq_bfp16_check as DQ  # noqa: E402
import gen_gemm_tile as TG  # noqa: E402

EMIT = DQ.EMIT
HALRUN, GPURUN = DQ.HALRUN, DQ.GPURUN


def emit(work, tag, fmt, t, mt, kb, tt, ostride):
    TG.OSTRIDE = ostride
    try:
        text = TG.gen(fmt, "kstore", t)
    finally:
        TG.OSTRIDE = 0
    sym = "yah_ffn_gemm_" + fmt
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
    t = TG.default_tile(fmt, "kstore", kb)
    mt, mtg = N // 16, (N - nn) // 16
    assert nn % 16 == 0 and mtg % t.rowgrp == 0 and B % t.bn == 0
    tt = B // t.bn
    full = emit(work, "full", fmt, t, mt, kb, tt, 0)
    split = emit(work, "split", fmt, t, mtg, kb, tt, N)
    tables = [DQ.table_file(fmt, e) for e in TG.G.FMTS[fmt]["extra"]]
    wst, ost = 17408 * 16 * 2, 17408 * B * 4   # the driver's wstage_ / ostage_
    env = dict(os.environ, HAL_RUN_ITERS="1")

    def run(hal, gx, out_spec):
        mins = ",".join(str(v) for v in [0] + [os.path.getsize(f) for f in tables] + [B * K * 2, wst, ost, N * B * 4])
        cmd = [GPURUN, "split-gemm", "--", HALRUN, model, hal, f"{gx},{tt}", str(t.lanes), mins, f"t:{name}"]
        cmd += [f"f:{f}" for f in tables] + [f"f:{act}", f"o:{wst}:/dev/null", f"o:{ost}:/dev/null", out_spec]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
        if "hal_run: ok" not in r.stdout:
            sys.exit(f"hal_run failed\n{r.stdout[-2000:]}{r.stderr[-2000:]}")

    yf, ys = os.path.join(work, "y_full.f32"), os.path.join(work, "y_split.f32")
    run(full, mt // t.rowgrp, f"o:{N * B * 4}:{yf}")
    sentinel = os.path.join(work, "y_nan.f32")
    np.full(N * B, np.nan, np.float32).tofile(sentinel)
    run(split, mtg // t.rowgrp, f"io:{sentinel}:{ys}")
    yfull = np.fromfile(yf, np.float32).reshape(B, N)
    ysplit = np.fromfile(ys, np.float32).reshape(B, N)
    gpu_same = np.array_equal(yfull[:, :N - nn].view(np.uint32), ysplit[:, :N - nn].view(np.uint32))
    npu_kept = bool(np.isnan(ysplit[:, N - nn:]).all())
    print(f"{name} ({fmt}) N={N} K={K} B={B}: GPU rows [0, {N - nn}) {'bit-identical' if gpu_same else 'DIFFER'} to the"
          f" full GEMM; NPU rows [{N - nn}, {N}) {'untouched' if npu_kept else 'WRITTEN'}")
    for f in (yf, ys, sentinel):
        os.remove(f)
    sys.exit(0 if gpu_same and npu_kept else 1)


if __name__ == "__main__":
    main()
