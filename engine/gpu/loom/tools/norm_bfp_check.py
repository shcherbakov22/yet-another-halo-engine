#!/usr/bin/env python3
"""The split RMSNorm with the NPU's BFP16 activation stream fused in (gen_half_norm.gen_split bfp=) against the plain norm.

usage: norm_bfp_check.py <model.gguf> <workdir> <norm weight tensor> [row|tiled|both] [rows]

Runs the plain norm and the bfp form on the same random hidden state: their f16 outputs must match bit for bit, and the
BFP16 stream must equal the numpy oracle of bfp16_check.py on that f16 (what yah_bfp16_encode_act writes from it).
HAL_RUN_ITERS=N times both.
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bfp16_check as B  # noqa: E402
import gen_bfp16_encode as GE  # noqa: E402
import gen_half_norm as N  # noqa: E402
from emit_prefill_pp import NORM_SPLIT, NPU_KS  # noqa: E402

DIM, WPR, PASSES = 5120, 4, 5


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    model, work, wname = sys.argv[1:4]
    mode = sys.argv[4] if len(sys.argv) > 4 else "row"
    rows = int(sys.argv[5]) if len(sys.argv) > 5 else 2048
    tiled = {"row": False, "tiled": True, "both": "both"}[mode]
    os.makedirs(work, exist_ok=True)
    rng = np.random.default_rng(5)
    x = (rng.standard_normal((rows, DIM)) * np.exp(rng.uniform(-2, 2, (rows, 1)))).astype(np.float32)
    xf = os.path.join(work, "x.f32")
    x.tofile(xf)
    total = GE.layout_bytes("act", rows, NPU_KS, PASSES)
    nout = rows * DIM * 2
    env = dict(os.environ, HAL_RUN_ITERS=os.environ.get("HAL_RUN_ITERS", "1"))
    outs = {}
    for tag, bfp in (("plain", None), ("bfp", (rows, NPU_KS, PASSES))):
        src = os.path.join(work, f"norm_{tag}.loom")
        open(src, "w").write(N.gen_split(DIM, wpr=WPR, split=NORM_SPLIT, tiled=tiled, bfp=bfp))
        r = subprocess.run([sys.executable, B.EMIT, src, os.path.join(work, tag), f"yah_half_norm.rows={rows}",
                            f"yah_half_norm.dim={DIM}", "yah_half_norm.eps=1e-06", "yah_half_norm.fused=0"],
                           capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"{tag}: emit failed\n{r.stdout[-3000:]}{r.stderr[-3000:]}")
        hal = r.stdout.strip().splitlines()[0]
        files = [os.path.join(work, f"{tag}.out{i}") for i in range(2 if tiled == "both" else 1)]
        binds = [f"f:{xf}", f"z:{x.nbytes}", f"t:{wname}", f"z:{rows * 4}"] + [f"o:{nout}:{f}" for f in files]
        mins = [x.nbytes, x.nbytes, DIM * 4, rows * 4] + [nout] * len(files)
        if bfp:
            files.append(os.path.join(work, "bfp.out"))
            binds.append(f"o:{total}:{files[-1]}")
            mins.append(total)
        cmd = [B.GPURUN, "norm-bfp", "--", B.HALRUN, model, hal, str(rows // WPR), str(32 * WPR * NORM_SPLIT),
               ",".join(map(str, mins))] + binds
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
        if "hal_run: ok" not in r.stdout:
            sys.exit(f"{tag}: hal_run failed\n{r.stdout[-2000:]}{r.stderr[-2000:]}")
        ms = [ln.strip() for ln in r.stdout.splitlines() if "ms per dispatch" in ln]
        outs[tag] = ([np.fromfile(f, np.uint8) for f in files], ms[0] if ms else "")
    pf, pms = outs["plain"]
    bf, bms = outs["bfp"]
    same = all(np.array_equal(a, b) for a, b in zip(pf, bf))
    h = pf[0].view(np.float16).reshape(rows, DIM)
    if tiled is True:   # fragment-major [row / 16][k / 16][row % 16][k % 16] back to rows
        h = h.reshape(rows // 16, DIM // 16, 16, 16).transpose(0, 2, 1, 3).reshape(rows, DIM)
    ref, off = B.reference(np.ascontiguousarray(h), "act", rows, list(NPU_KS), PASSES, 64, True)
    idx = (off[..., None] + np.arange(72)).reshape(-1)
    bad = np.count_nonzero(bf[-1][idx] != ref[idx])
    print(f"{wname} {mode} rows={rows}: f16 outputs {'identical' if same else 'DIFFER'}, BFP16 stream {bad} of {idx.size} "
          f"bytes differ | plain {pms} | bfp {bms}")
    os.remove(xf)
    sys.exit(0 if same and not bad else 1)


if __name__ == "__main__":
    main()
