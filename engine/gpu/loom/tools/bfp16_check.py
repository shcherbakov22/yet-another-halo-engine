#!/usr/bin/env python3
"""Check gen_bfp16_encode.py against a numpy oracle, byte for byte.

usage: bfp16_check.py <model.gguf> <workdir> act|wgt <rows> <ks,ks,...> <passes> [input.f16 | random] [tile] [pad]

The input is an f16 [rows][K] file (e.g. a YAH_DUMP_ACT dump) or random rows with a wide per-row dynamic range.
Every work item's store range is simulated first; the kernel is dispatched only if all of them fit the output.
Writes <workdir>/<layout>.bfp (the encoded stream) for reuse by the NPU harness.
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_bfp16_encode as G  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
EMIT = os.path.join(HERE, "..", "emit_hal.py")
HALRUN = os.path.join(ROOT, "engine", "build", "hal_run")
GPURUN = os.path.join(ROOT, "engine", "run", "gpu_run.sh")


def quantize(x):
    """bfp16ebs8 of f16 rows: E [rows][K/8] u8, m [rows][K] i8 (the NPU GEMM harness's rule)."""
    rows, k = x.shape
    b = x.astype(np.float64).reshape(rows, k // 8, 8)
    amax = np.abs(b).max(2)
    ex = np.where(amax > 0, np.floor(np.log2(np.where(amax > 0, amax, 1.0))), -127)
    E = np.clip(ex + 127, 0, 255).astype(np.uint8)
    q = np.exp2(E.astype(np.float64) - 133)[..., None]
    m = np.clip(np.round(b / q), -128, 127).astype(np.int8)
    return E, m.reshape(rows, k)


def offsets(layout, rows, ks, passes, tile, pad):
    """Byte offset of every fragment [rows/8][K/8 blocks], mirroring the kernel's index math."""
    pk = sum(ks)
    kb = passes * pk
    sb = np.array([G.slab_bytes(k, pad) for k in ks])
    sub = tile // 16
    start = np.cumsum([0] + ks[:-1])
    if layout == "act":
        nblk = rows // tile
        base = np.cumsum([0] + [nblk * passes * sub * b for b in sb[:-1]])
    else:
        base = np.cumsum([0] + [passes * sub * b for b in sb[:-1]])
        panel = int(sum(passes * sub * b for b in sb))
    g8 = np.arange(rows // 8)[:, None]
    kbi = np.arange(kb)[None, :]
    g16, h = g8 // 2, g8 % 2
    blk, slab = g16 // sub, g16 % sub
    p, o = kbi // pk, kbi % pk
    sl = np.searchsorted(start, o, side="right") - 1
    kin = o - start[sl]
    if layout == "act":
        rec = (blk * passes + p) * sub + slab
        off = base[sl] + rec * sb[sl]
    else:
        off = blk * panel + base[sl] + (p * sub + slab) * sb[sl]
    return off + kin * 144 + h * 72


def reference(x, layout, rows, ks, passes, tile, pad):
    E, m = quantize(x)
    kb = E.shape[1]
    frag = np.zeros((rows // 8, kb, 8, 9), np.uint8)
    Er = E.reshape(rows // 8, 8, kb).transpose(0, 2, 1)
    mr = m.view(np.uint8).reshape(rows // 8, 8, kb, 8).transpose(0, 2, 1, 3)
    frag[..., 0] = Er
    frag[..., 1:] = mr
    off = offsets(layout, rows, ks, passes, tile, pad)
    out = np.zeros(G.layout_bytes(layout, rows, ks, passes, tile, pad), np.uint8)
    idx = off[..., None] + np.arange(72)
    out[idx.reshape(-1)] = frag.reshape(-1)
    return out, off


def main():
    if len(sys.argv) < 7:
        sys.exit(__doc__)
    model, work, layout = sys.argv[1], sys.argv[2], sys.argv[3]
    rows, ks, passes = int(sys.argv[4]), [int(v) for v in sys.argv[5].split(",")], int(sys.argv[6])
    src_in = sys.argv[7] if len(sys.argv) > 7 else "random"
    tile = int(sys.argv[8]) if len(sys.argv) > 8 else 64
    pad = (sys.argv[9] != "0") if len(sys.argv) > 9 else True
    K = 8 * passes * sum(ks)
    os.makedirs(work, exist_ok=True)
    if src_in == "random":
        rng = np.random.default_rng(7)
        x = (rng.standard_normal((rows, K)) * np.exp(rng.uniform(-6, 4, (rows, 1)))).astype(np.float16)
        x[:, :8] = 0                                       # all-zero blocks
    else:
        x = np.fromfile(src_in, np.float16)
        assert x.size >= rows * K, f"{src_in}: {x.size} halves < {rows} x {K}"
        x = x[:rows * K].reshape(rows, K)
    ref, off = reference(x, layout, rows, ks, passes, tile, pad)
    total = ref.size
    if off.min() < 0 or off.max() + 72 > total:
        sys.exit(f"store range [{off.min()}, {off.max() + 72}) outside the {total}-byte output: not dispatching")
    tag = f"bfp16_{layout}"
    srcf = os.path.join(work, tag + ".loom")
    open(srcf, "w").write(G.gen(layout, rows, ks, passes, tile, pad))
    r = subprocess.run([sys.executable, EMIT, srcf, os.path.join(work, tag), "nop=0"], capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"emit failed\n{r.stdout[-3000:]}{r.stderr[-3000:]}")
    hal = r.stdout.strip().splitlines()[0]
    xf = os.path.join(work, tag + ".x")
    x.tofile(xf)
    of = os.path.join(work, layout + ".bfp")
    items = (rows // 8) * (K // 8)
    cmd = [GPURUN, "bfp16-check", "--", HALRUN, model, hal, str((items + G.WG - 1) // G.WG), str(G.WG),
           f"{x.nbytes},{total}", f"f:{xf}", f"o:{total}:{of}"]
    env = dict(os.environ, HAL_RUN_ITERS=os.environ.get("HAL_RUN_ITERS", "1"))
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    if "hal_run: ok" not in r.stdout:
        sys.exit(f"hal_run failed\n{r.stdout[-2000:]}{r.stderr[-2000:]}")
    got = np.fromfile(of, np.uint8)
    os.remove(xf)
    ms = [ln for ln in r.stdout.splitlines() if "ms per dispatch" in ln]
    # padding bytes are never written; compare the fragment bytes only
    idx = (off[..., None] + np.arange(72)).reshape(-1)
    bad = np.count_nonzero(got[idx] != ref[idx])
    print(f"{layout} rows={rows} K={K} ks={ks} passes={passes}: {bad} of {idx.size} fragment bytes differ"
          + (f" | {ms[0].strip()}" if ms else ""))
    if bad:
        first = idx[np.nonzero(got[idx] != ref[idx])[0][0]]
        print(f"  first mismatch at byte {first}: got {got[first]} want {ref[first]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
