#!/usr/bin/env python3
"""Check gen_npu_unpack.py against npu_gemm_check.unpack_c, bit for bit.

usage: npu_unpack_check.py <model.gguf> <workdir> <tokens> <cols> <stride> <off>

Random C into an output filled with NaN: columns [off, off + 64 * cols) must hold C exactly, all others stay NaN.
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
    os.makedirs(work, exist_ok=True)
    c = np.random.default_rng(3).standard_normal(tokens * cols * 64).astype(np.float32)
    cf, of, sf = (os.path.join(work, n) for n in ("c.f32", "out.f32", "nan.f32"))
    c.tofile(cf)
    np.full(tokens * stride, np.nan, np.float32).tofile(sf)
    src = os.path.join(work, "unpack.loom")
    open(src, "w").write(G.gen(tokens, cols, stride, off))
    r = subprocess.run([sys.executable, B.EMIT, src, os.path.join(work, "unpack"), "nop=0"], capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"emit failed\n{r.stdout[-3000:]}{r.stderr[-3000:]}")
    hal = r.stdout.strip().splitlines()[0]
    cmd = [B.GPURUN, "npu-unpack", "--", B.HALRUN, model, hal, str(tokens * cols * 8 // G.WG), str(G.WG),
           f"{c.nbytes},{tokens * stride * 4}", f"f:{cf}", f"io:{sf}:{of}"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=dict(os.environ, HAL_RUN_ITERS="1"))
    if "hal_run: ok" not in r.stdout:
        sys.exit(f"hal_run failed\n{r.stdout[-2000:]}{r.stderr[-2000:]}")
    got = np.fromfile(of, np.float32).reshape(tokens, stride)
    want = N.unpack_c(c, cols, tokens // 64)
    ok = np.array_equal(got[:, off:off + 64 * cols], want)
    rest = np.delete(got, np.s_[off:off + 64 * cols], axis=1)
    kept = bool(np.isnan(rest).all())
    print(f"unpack tokens={tokens} cols={cols} stride={stride} off={off}: slice {'exact' if ok else 'DIFFERS'}, "
          f"other columns {'untouched' if kept else 'WRITTEN'}")
    for f in (cf, of, sf):
        os.remove(f)
    sys.exit(0 if ok and kept else 1)


if __name__ == "__main__":
    main()
