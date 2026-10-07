#!/usr/bin/env python3
"""The NPU's weight share through the GPU: the tile GEMM's dequant kind (f16 [rows][K]) then the BFP16 encoder.

usage: dq_bfp16_check.py <model.gguf> <workdir> <tensor> <npu_rows> <ks,ks,...> <passes>

Takes the last <npu_rows> output features of <tensor> (the NPU computes the trailing columns).
Dequantizes them with yah_dequant_<fmt> (the GPU GEMMs' decode) and checks the f16 against gguf-py.
Encodes that f16 into the cascade GEMM's weight stream with yah_bfp16_encode_wgt and checks the bytes against the numpy oracle.
Runs the fused form (yah_dequant_<fmt>_bfp16, gen_gemm_tile.DQ_BFP: the dequant writes the stream itself) against the same oracle.
DQ_KCHUNK=<k_off>,<K>: a K chunk (ks / passes describe the chunk; the fused form decodes it from the full rows).
Writes <workdir>/dq.f16 and <workdir>/wgt.bfp. Needs PYTHONPATH with llama.cpp's gguf-py.
"""
import dataclasses
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bfp16_check as B  # noqa: E402
import gen_bfp16_encode as GE  # noqa: E402
import gen_gemm_tile as TG  # noqa: E402
import gen_npu_gemm as GN  # noqa: E402

EMIT = B.EMIT
HALRUN, GPURUN = B.HALRUN, B.GPURUN
TABLE_DIR = os.path.join(HERE, "..", "tables")


def table_file(fmt, extra):
    """The tile GEMM's LDS table bindings: the format's grid, and the IQ2 sign table."""
    return os.path.join(TABLE_DIR, "ksigns_iq2xs.bin" if extra == "ksigns" else f"grid_{fmt}.bin")
FMT = {"Q3_K": "q3k", "Q4_K": "q4k", "Q5_K": "q5k", "Q6_K": "q6k", "IQ4_XS": "iq4xs", "IQ3_XXS": "iq3xxs",
       "IQ3_S": "iq3s", "IQ2_XXS": "iq2xxs", "IQ2_XS": "iq2xs", "Q8_0": "q8_0"}


def run(tag, cmd, env):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    if "hal_run: ok" not in r.stdout:
        sys.exit(f"{tag}: hal_run failed\n{r.stdout[-2000:]}{r.stderr[-2000:]}")
    return r.stdout


def main():
    if len(sys.argv) < 7:
        sys.exit(__doc__)
    import gguf
    from gguf import quants
    model, work, name = sys.argv[1], sys.argv[2], sys.argv[3]
    nn, ks, passes = int(sys.argv[4]), [int(v) for v in sys.argv[5].split(",")], int(sys.argv[6])
    os.makedirs(work, exist_ok=True)
    rd = gguf.GGUFReader(model)
    t = next(x for x in rd.tensors if x.name == name)
    fmt = FMT[t.tensor_type.name]
    K, N = int(t.shape[0]), int(t.shape[1])
    k_off, Kc = (int(v) for v in os.environ.get("DQ_KCHUNK", f"0,{K}").split(","))
    chunk = Kc != K
    assert Kc == 8 * passes * sum(ks) and k_off + Kc <= K and k_off % 1024 == 0, f"{name}: chunk {k_off}+{Kc} of {K}"
    raw = np.asarray(t.data)                               # [N][row bytes]
    share = np.ascontiguousarray(raw[N - nn:])
    wf = os.path.join(work, "dq.w")
    share.tofile(wf)
    ref = quants.dequantize(share, t.tensor_type).astype(np.float16)   # [nn][K]
    # the dequant kind as emit_prefill_pp.decode_free builds it
    dt = TG.default_tile(fmt, "kstore", K // 256)
    dt = dataclasses.replace(dt, bm=64, wm=2, wn=4, decahead=False, ksub=64, dbuf=False)
    kb = K // 256 * TG.G.FMTS[fmt].get("kdiv", 1)          # k_blocks counts the format's blocks (q8_0: 32 wide)
    mt = nn // 16
    assert mt % dt.rowgrp == 0 and (kb // TG.G.FMTS[fmt].get("kdiv", 1)) % TG.DQ_BLOCKS == 0
    sym = "yah_dequant_" + fmt
    src = os.path.join(work, "dq.loom")
    open(src, "w").write(TG.gen(fmt, "dequant", dt))
    r = subprocess.run([sys.executable, EMIT, src, os.path.join(work, "dq"), f"{sym}.m_tiles={mt}",
                        f"{sym}.k_blocks={kb}", f"{sym}.token_tiles=1"], capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"dequant emit failed\n{r.stdout[-3000:]}{r.stderr[-3000:]}")
    hal = r.stdout.strip().splitlines()[0]
    tables = [table_file(fmt, x) for x in TG.G.FMTS[fmt]["extra"]]
    df = os.path.join(work, "dq.f16")
    mins = [share.nbytes] + [os.path.getsize(x) for x in tables] + [nn * K * 2]
    env = dict(os.environ, HAL_RUN_ITERS=os.environ.get("HAL_RUN_ITERS", "1"))
    out = run("dequant", [GPURUN, "dq-bfp16", "--", HALRUN, model, hal, str(TG.DQ_WGS), str(dt.lanes),
                          ",".join(str(v) for v in mins), f"f:{wf}"] + [f"f:{x}" for x in tables]
              + [f"o:{nn * K * 2}:{df}"], env)
    got = np.fromfile(df, np.float16).reshape(nn, K)
    d = np.abs(got.astype(np.float32) - ref.astype(np.float32))
    exact = np.count_nonzero(got.view(np.uint16) != ref.view(np.uint16))
    ms = [ln.strip() for ln in out.splitlines() if "ms per dispatch" in ln]
    print(f"{name} ({fmt}) rows {N - nn}..{N} K={K}: dequant f16 vs gguf-py: {exact} of {got.size} differ, "
          f"max |d| {d.max():.3e} (max |w| {np.abs(ref.astype(np.float32)).max():.3e})" + (f" | {ms[0]}" if ms else ""))
    # encode the GPU's f16 (what the NPU will multiply) and check it byte for byte
    eref, off = B.reference(np.ascontiguousarray(got[:, k_off:k_off + Kc]), "wgt", nn, ks, passes, GN.TN, True)
    total = eref.size
    assert off.min() >= 0 and off.max() + 72 <= total
    esrc = os.path.join(work, "enc.loom")
    open(esrc, "w").write(GE.gen("wgt", nn, ks, passes, GN.TN, k_off=k_off, k_src=K if chunk else None))
    r = subprocess.run([sys.executable, EMIT, esrc, os.path.join(work, "enc"), "nop=0"], capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"encoder emit failed\n{r.stdout[-3000:]}{r.stderr[-3000:]}")
    ehal = r.stdout.strip().splitlines()[0]
    of = os.path.join(work, "wgt.bfp")
    items = (nn // 8) * (Kc // 8)
    out = run("encode", [GPURUN, "dq-bfp16", "--", HALRUN, model, ehal, str((items + GE.WG - 1) // GE.WG), str(GE.WG),
                         f"{nn * K * 2},{total}", f"f:{df}", f"o:{total}:{of}"], env)
    gb = np.fromfile(of, np.uint8)
    idx = (off[..., None] + np.arange(72)).reshape(-1)
    bad = np.count_nonzero(gb[idx] != eref[idx])
    ms = [ln.strip() for ln in out.splitlines() if "ms per dispatch" in ln]
    print(f"  weight stream: {bad} of {idx.size} fragment bytes differ" + (f" | {ms[0]}" if ms else ""))
    # fused: dequant straight into the stream
    TG.DQ_BFP = (tuple(ks), passes) + ((k_off // 256, K // 256) if chunk else ())
    try:
        text = TG.gen(fmt, "dequant", dataclasses.replace(dt, ksub=TG.DQ_BFP_KSUB))
    finally:
        TG.DQ_BFP = None
    fsym = sym + "_bfp16"
    fsrc = os.path.join(work, "dqbfp.loom")
    open(fsrc, "w").write(text)
    kbc = Kc // 256 * TG.G.FMTS[fmt].get("kdiv", 1)
    r = subprocess.run([sys.executable, EMIT, fsrc, os.path.join(work, "dqbfp"), f"{fsym}.m_tiles={mt}",
                        f"{fsym}.k_blocks={kbc}", f"{fsym}.token_tiles=1"], capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"fused emit failed\n{r.stdout[-3000:]}{r.stderr[-3000:]}")
    fhal = r.stdout.strip().splitlines()[0]
    ff = os.path.join(work, "wgt_fused.bfp")
    mins = [share.nbytes] + [os.path.getsize(x) for x in tables] + [total]
    out = run("fused", [GPURUN, "dq-bfp16", "--", HALRUN, model, fhal, str(TG.dq_wgs(mt, kbc, dt, fmt)), str(dt.lanes),
                        ",".join(str(v) for v in mins), f"f:{wf}"] + [f"f:{x}" for x in tables] + [f"o:{total}:{ff}"], env)
    fb = np.fromfile(ff, np.uint8)
    fbad = np.count_nonzero(fb[idx] != eref[idx])
    ms = [ln.strip() for ln in out.splitlines() if "ms per dispatch" in ln]
    print(f"  fused dequant -> stream: {fbad} of {idx.size} fragment bytes differ" + (f" | {ms[0]}" if ms else ""))
    os.remove(wf)
    os.remove(ff)
    sys.exit(1 if bad or fbad else 0)


if __name__ == "__main__":
    main()
