#!/usr/bin/env python3
"""Check the NPU split's K remainder on the GPU: the afrag kstore over a K window (gen_gemm_tile.KWIN) of the NPU's rows.

usage: npu_rem_check.py <model.gguf> <workdir> <tensor> <npu_rows> <kb_start> [act.f16]

The NPU computes K columns [0, 256 * kb_start) of its trailing <npu_rows> rows; this kernel computes the rest,
[256 * kb_start, K), into f32 [B][npu_rows]. Its weight binding is those rows (as the driver binds them, at their byte
offset), its input the whole fragment-major [B][K] activation. Compared with the float64 product of the f16 dequant
(gguf-py, the GPU decode's values) and the f16 input over the window: the error is f32 accumulation order only.
act.f16: a [B][K] input (default: random, B = 2048). HAL_RUN_ITERS=N times it. Needs PYTHONPATH with llama.cpp's gguf-py.
REM_LEAD=1: the leading window [0, 256 * kb_start) instead (the FFN block's down over the GPU's features, npuffnrem_<fmt>).
"""
import dataclasses
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dq_bfp16_check as DQ  # noqa: E402
import emit_prefill_pp as EP  # noqa: E402
import gen_gemm_tile as TG  # noqa: E402


def main():
    if len(sys.argv) < 6:
        sys.exit(__doc__)
    import gguf
    from gguf import quants
    model, work, name, nn, kb0 = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    os.makedirs(work, exist_ok=True)
    tn = next(x for x in gguf.GGUFReader(model).tensors if x.name == name)
    fmt = DQ.FMT[tn.tensor_type.name]
    K, N = int(tn.shape[0]), int(tn.shape[1])
    kbt = K // 256
    lead = os.environ.get("REM_LEAD") == "1"
    kbw = kb0 if lead else kbt - kb0
    assert TG.G.FMTS[fmt].get("kdiv", 1) == 1 and 0 < kb0 < kbt and nn % 128 == 0
    if len(sys.argv) > 6:
        x = np.fromfile(sys.argv[6], np.float16).reshape(-1, K)
    else:
        x = (np.random.default_rng(9).standard_normal((2048, K)) * 0.5).astype(np.float16)
    B = x.shape[0]
    share = np.ascontiguousarray(np.asarray(tn.data)[N - nn:])
    wf = os.path.join(work, "rem.w")
    share.tofile(wf)
    w16 = quants.dequantize(share, tn.tensor_type).astype(np.float16)   # [nn][K]
    act_t = os.path.join(work, "act_t.f16")
    x.reshape(B // 16, 16, K // 16, 16).transpose(0, 2, 1, 3).tofile(act_t)
    t = EP.npu_rem_tile(fmt, kbw, kbt)
    TG.KWIN = (0 if lead else kb0, kbt)
    try:
        text = TG.gen(fmt, "kstore", t, False)
    finally:
        TG.KWIN = None
    sym = "yah_ffn_gemm_" + fmt
    src = os.path.join(work, "rem.loom")
    open(src, "w").write(text)
    mt, tt = nn // 16, B // t.bn
    gate = subprocess.run([sys.executable, os.path.join(HERE, "footprint_gate.py"), src, sym, fmt, "kstore", str(mt),
                           str(kbw), str(tt), str(B), f"kfull={kbt}"], capture_output=True, text=True)
    if gate.returncode:
        sys.exit(f"footprint gate refused: {(gate.stdout + gate.stderr)[-600:]}")
    r = subprocess.run([sys.executable, DQ.EMIT, src, os.path.join(work, "rem"), f"{sym}.m_tiles={mt}",
                        f"{sym}.k_blocks={kbw}", f"{sym}.token_tiles={tt}"], capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"emit failed\n{r.stdout[-3000:]}{r.stderr[-3000:]}")
    hal = r.stdout.strip().splitlines()[0]
    tables = [DQ.table_file(fmt, e) for e in TG.G.FMTS[fmt]["extra"]]
    wst, ost = 17408 * 16 * 2, 17408 * B * 4   # the driver's wstage_ / ostage_
    out = os.path.join(work, "rem.f32")
    obytes = B * nn * 4
    mins = ",".join(str(v) for v in [share.nbytes] + [os.path.getsize(f) for f in tables] + [B * K * 2, wst, ost, obytes])
    cmd = [DQ.GPURUN, "npu-rem", "--", DQ.HALRUN, model, hal, f"{mt // t.rowgrp},{tt}", str(t.lanes), mins, f"f:{wf}"]
    cmd += [f"f:{f}" for f in tables] + [f"f:{act_t}", f"o:{wst}:/dev/null", f"o:{ost}:/dev/null", f"o:{obytes}:{out}"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                       env=dict(os.environ, HAL_RUN_ITERS=os.environ.get("HAL_RUN_ITERS", "1")))
    if "hal_run: ok" not in r.stdout:
        sys.exit(f"hal_run failed\n{r.stdout[-2000:]}{r.stderr[-2000:]}")
    ms = [ln.strip() for ln in r.stdout.splitlines() if "ms per dispatch" in ln]
    got = np.fromfile(out, np.float32).reshape(B, nn).astype(np.float64)
    k0 = 256 * kb0
    kw = slice(0, k0) if lead else slice(k0, K)
    ref = x[:, kw].astype(np.float64) @ w16[:, kw].astype(np.float64).T
    err = np.abs(got - ref).max() / np.abs(ref).max()
    print(f"{name} ({fmt}) rows [{N - nn}, {N}) K window [{kw.start}, {kw.stop}) B={B}: max |err| / max |ref| {err:.2e}"
          + (f" | {ms[0]}" if ms else ""))
    for f in (wf, act_t, out):
        os.remove(f)
    sys.exit(0 if err < 1e-5 else 1)


if __name__ == "__main__":
    main()
