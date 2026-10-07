#!/usr/bin/env python3
"""Check gen_npu_dequant's Q4_K decoder on the NPU against a numpy oracle, byte for byte, on real model blocks.

usage: npu_dequant_check.py <model.gguf> <workdir> [records]

Takes 8-row super-block groups of the first Q4_K tensor (every other one with a subnormal f16 d), runs the leaf once per
group in a 1-worker array program and compares the 32 BFP16 fragments of each group with the oracle (the leaf's f32
op order, then the hardware bfp16 rule). Prints the leaf's static bundle count (= cycles of the straight-line leaf).
Needs PYTHONPATH with llama.cpp's gguf-py.
"""
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import gen_npu_dequant as G  # noqa: E402
import hrx_paths  # noqa: E402

IN_B, OUT_B = G.IN_B, G.OUT_B


def array_program(records):
    S = "reg<aie2p.array.scalar : index>"
    return f"""aie2p.target<array> @array_target
aie2p.target<core> @core_target

low.func.def public retain target<amd.xdna.aie2p.array>(@array_target) abi(array_program) @dq_check() asm {{
  %n0 = constant.u32 0 : {S}
  %n1 = constant.u32 1 : {S}
  %n2 = constant.u32 2 : {S}
  %rec = constant.u32 {records} : {S}
  %origin = constant.u64 0 : reg<aie2p.array.offset : offset>
  %workers = group %n1
  %ib = binding 0, "read"
  %ob = binding 1, "write"
  %k = worker %workers, %n0, @dq_q4k
  constrain.location %k, %n0, %n2
  %o_all = receiver %ob, 0 : reg<aie2p.array.receiver : tile<1x{records}x{OUT_B // 4}xi32>>
  %orc = partition.receiver %o_all, %origin, %n0, %n1 : reg<aie2p.array.receiver : tile<{OUT_B // 4}xi32>>
  %so = sender %k, 1 : reg<aie2p.array.sender : tile<{OUT_B // 4}xi32>>
  %cho = channel %so, %orc, %n2, %rec : reg<aie2p.array.channel : tile<{OUT_B // 4}xi32>>
  %is = sender %ib, 0 : reg<aie2p.array.sender : tile<{records}x{IN_B // 4}xi32>>
  %iv = view.sender %is, %origin : reg<aie2p.array.sender : tile<{IN_B // 4}xi32>>
  %ri = receiver %k, 0 : reg<aie2p.array.receiver : tile<{IN_B // 4}xi32>>
  %chi = channel %iv, %ri, %n2, %rec : reg<aie2p.array.channel : tile<{IN_B // 4}xi32>>
  return
}}
"""


def q4k_fields(blk):
    """blk [..., 144] u8 -> d, dmin (f64), sc, m [..., 8], q [..., 256] (llama.cpp block_q4_K)."""
    d = blk[..., 0:2].copy().view(np.float16)[..., 0].astype(np.float64)
    dm = blk[..., 2:4].copy().view(np.float16)[..., 0].astype(np.float64)
    s = blk[..., 4:16].astype(np.int64)
    sc = np.zeros(blk.shape[:-1] + (8,), np.int64)
    mn = np.zeros_like(sc)
    for j in range(4):
        sc[..., j] = s[..., j] & 63
        mn[..., j] = s[..., j + 4] & 63
    for j in range(4, 8):
        sc[..., j] = (s[..., j + 4] & 15) | ((s[..., j - 4] >> 6) << 4)
        mn[..., j] = (s[..., j + 4] >> 4) | ((s[..., j] >> 6) << 4)
    qs = blk[..., 16:144].astype(np.int64).reshape(blk.shape[:-1] + (4, 32))
    q = np.concatenate([np.stack([qs[..., j, :] & 15, qs[..., j, :] >> 4], -2) for j in range(4)], -2)
    return d, dm, sc, mn, q.reshape(blk.shape[:-1] + (256,))


def pack_record(blk8):
    """blk8 [8 rows][144] -> the leaf's input record: headers [row][16 B], qs [chunk][row][8 B]."""
    return np.concatenate([blk8[:, :16].reshape(-1), blk8[:, 16:].reshape(8, 16, 8).transpose(1, 0, 2).reshape(-1)])


def bf16_rne(x):
    u = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).astype(np.uint32).view(np.float32)


def oracle(T, M, q):
    """The leaf's f32 op order. T, M [..., 8 sub-blocks] exact (17 bits); q [..., 256]."""
    f = np.float32
    T, M = T.astype(f), M.astype(f)
    T1 = bf16_rne(T)
    T2 = bf16_rne(T - T1)
    M1 = bf16_rne(M)
    M2 = bf16_rne(M - M1)
    M3 = (M - M1 - M2).astype(f)
    c = (f(-128) * T1).astype(f)
    for t in (f(-128) * T2, -M1, -M2, -M3):
        c = (c + t).astype(f)
    X = (128 + q).astype(f)
    w = (np.repeat(c, 32, -1) + np.repeat(T1, 32, -1) * X).astype(f)
    return (w + np.repeat(T2, 32, -1) * X).astype(f)


def bfp_hw(w):
    """w [n][8 rows][8] f32 -> [n][72] B: E = exponent of the block max, +1 if a rounded mantissa leaves int8."""
    w = np.where(np.abs(w) < np.float32(2.0 ** -126), np.float32(0), w)
    E = ((np.abs(w).max(2).view(np.uint32) >> 23) & 255).astype(np.int64)
    m0 = np.round(w.astype(np.float64) * np.exp2(133 - E[..., None]))
    E = E + np.any((m0 > 127) | (m0 < -128), axis=2)
    m = np.clip(np.round(w.astype(np.float64) * np.exp2(133 - E[..., None])), -128, 127).astype(np.int8)
    out = np.zeros(w.shape[:2] + (9,), np.uint8)
    out[..., 0] = E
    out[..., 1:] = m.view(np.uint8)
    return out.reshape(w.shape[0], 72)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    import gguf
    model, work = sys.argv[1], sys.argv[2]
    R = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    os.makedirs(work, exist_ok=True)
    src = os.path.join(work, "dq.loom")
    open(src, "w").write(array_program(R) + "\n" + G.gen_q4k() + "\n")
    env = dict(hrx_paths.env(), LOOM_EXP_LOCKED_PACK="1")
    report = os.path.join(work, "report.txt")
    r = subprocess.run([hrx_paths.LOOM_COMPILE, src, "--root=@dq_check", "--target=amd.xdna.aie2p:amd.xdna.strix_halo.17f0_11",
                        f"--output={work}/dq.xdna", "--compile-report=text-details", f"--compile-report-output={report}"],
                       capture_output=True, text=True, env=env)
    if r.returncode:
        sys.exit("loom-compile failed\n" + r.stderr[-4000:])
    bundles = next(re.search(r" instructions=(\d+)", l).group(1) for l in open(report) if l.startswith("COMPILE-REPORT: emission"))

    t = next(x for x in gguf.GGUFReader(model).tensors if x.tensor_type.name == "Q4_K")
    nsb = int(t.shape[0]) // 256
    raw = np.asarray(t.data).view(np.uint8).reshape(-1, nsb, 144)
    rng = np.random.default_rng(3)
    sub = np.argwhere(((raw[:, :, 1] >> 2) & 31) == 0)
    groups = []
    for i in range(R):
        if i % 2 and len(sub):
            row, sb = sub[rng.integers(len(sub))]
            row = min(row - row % 8, raw.shape[0] - 8)
        else:
            row, sb = 8 * rng.integers(raw.shape[0] // 8), rng.integers(nsb)
        groups.append(raw[row:row + 8, sb])
    blks = np.stack(groups)
    np.stack([pack_record(g) for g in blks]).tofile(f"{work}/in.bin")
    np.zeros(R * OUT_B, np.uint8).tofile(f"{work}/out0.bin")
    r = subprocess.run([hrx_paths.XDNA_RUN, "--columns=1", f"--image={work}/dq.xdna", "--entry=dq_check",
                        "--binding_memory=system", f"--binding={work}/in.bin", f"--binding={work}/out0.bin",
                        f"--output=1={work}/out.bin"], capture_output=True, text=True, env=env, timeout=60)
    if r.returncode:
        sys.exit("xdna run failed\n" + r.stdout[-1500:] + r.stderr[-1500:])
    got = np.fromfile(f"{work}/out.bin", np.uint8).reshape(R, 32, 72)
    d, dm, sc, mn, q = q4k_fields(blks)
    w = oracle(d[..., None] * sc, dm[..., None] * mn, q)
    ref = bfp_hw(w.reshape(R, 8, 32, 8).transpose(0, 2, 1, 3).reshape(R * 32, 8, 8)).reshape(R, 32, 72)
    bad = np.any(got != ref, axis=2)
    print(f"q4k decode: {int(bad.sum())} of {bad.size} fragments differ ({R} groups, half with subnormal d); "
          f"leaf {bundles} bundles per group")
    if bad.any():
        i, f = np.argwhere(bad)[0]
        sys.exit(f"group {i} fragment {f}\n got {got[i, f].reshape(8, 9).tolist()}\n ref {ref[i, f].reshape(8, 9).tolist()}")


if __name__ == "__main__":
    main()
