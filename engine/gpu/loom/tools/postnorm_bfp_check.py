#!/usr/bin/env python3
"""The DeltaNet postnorm over heads 0..39 with ssm_out's NPU input fused in (gen_postnorm_bfp) against the production part.

usage: postnorm_bfp_check.py <model.gguf> <workdir> [row|tiled]

Runs emit_prefill_pp's postnorm part a (postnorm_heads of yah_ssm_postnorm_gate_f16 with the f16 gate, fragment-major
with tiled) and the fused kernel on the same random input: their f16 outputs must match bit for bit, and the BFP16 stream
must equal the numpy oracle of bfp16_check.py on that f16 (what yah_bfp16_encode_act writes for ssm_out's first K chunk).
HAL_RUN_ITERS=N times both.
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bfp16_check as B  # noqa: E402
import emit_prefill as E  # noqa: E402
import emit_prefill_pp as EP  # noqa: E402
import gen_postnorm_bfp as PB  # noqa: E402

T, H0, NH, PASSES = 2048, 0, 40, 5


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    model, work = sys.argv[1:3]
    tiled = len(sys.argv) > 3 and sys.argv[3] == "tiled"
    os.makedirs(work, exist_ok=True)
    rng = np.random.default_rng(7)
    raw = (rng.standard_normal((T * 48, 128)) * np.exp(rng.uniform(-2, 2, (T * 48, 1)))).astype(np.float32)
    gate = (rng.standard_normal((T * 48, 128)) * 2).astype(np.float16)
    rf, gf, nf = (os.path.join(work, n) for n in ("raw.f32", "gate.f16", "nan.f16"))
    raw.tofile(rf)
    gate.tofile(gf)
    nout = T * 6144 * 2
    np.full(T * 6144, np.nan, np.float16).tofile(nf)
    ks = list(EP.NPU_KS)
    nab = B.G.layout_bytes("act", T, ks, PASSES)
    base = open(os.path.join(E.LOOM, "yah_ssm_postnorm_gate_f16.loom")).read()
    plain = EP.postnorm_heads(EP.postnorm_g16(base), H0, NH, T)
    if tiled:
        plain = EP.postnorm_tiled(plain)
    env = dict(os.environ, HAL_RUN_ITERS=os.environ.get("HAL_RUN_ITERS", "1"))
    outs = {}
    for tag, text in (("plain", plain), ("bfp", PB.gen(T, H0, NH, ks, PASSES, tiled))):
        src = os.path.join(work, f"pn_{tag}.loom")
        open(src, "w").write(text)
        r = subprocess.run([sys.executable, B.EMIT, src, os.path.join(work, tag),
                            f"yah_ssm_postnorm_fp16.head_count={NH * T}"], capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"{tag}: emit failed\n{r.stdout[-3000:]}{r.stderr[-3000:]}")
        hal = r.stdout.strip().splitlines()[0]
        of = os.path.join(work, f"{tag}.out")
        binds = [f"f:{rf}", "t:blk.0.ssm_norm.weight", f"f:{gf}", f"io:{nf}:{of}"]
        mins = [raw.nbytes, 128 * 4, gate.nbytes, nout]
        files = [of]
        if tag == "bfp":
            files.append(os.path.join(work, "bfp.ab"))
            binds.append(f"o:{nab}:{files[-1]}")
            mins.append(nab)
        cmd = [B.GPURUN, "postnorm-bfp", "--", B.HALRUN, model, hal, str(NH * T // 8), "256",
               ",".join(map(str, mins))] + binds
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
        if "hal_run: ok" not in r.stdout:
            sys.exit(f"{tag}: hal_run failed\n{r.stdout[-2000:]}{r.stderr[-2000:]}")
        ms = [ln.strip() for ln in r.stdout.splitlines() if "ms per dispatch" in ln]
        outs[tag] = ([np.fromfile(f, np.uint8) for f in files], ms[0] if ms else "")
    (po,), pms = outs["plain"]
    (bo, ab), bms = outs["bfp"]
    same = np.array_equal(po, bo)
    h = po.view(np.float16).reshape(T, 6144) if not tiled else \
        po.view(np.float16).reshape(T // 16, 384, 16, 16).transpose(0, 2, 1, 3).reshape(T, 6144)
    ref, off = B.reference(np.ascontiguousarray(h[:, :1024 * PASSES]), "act", T, ks, PASSES, 64, True)
    kb = (H0 + NH) * 16   # the k-blocks the part writes
    idx = (off[:, :kb, None] + np.arange(72)).reshape(-1)
    bad = np.count_nonzero(ab[idx] != ref[idx])
    print(f"postnorm heads {H0}..{H0 + NH - 1} {'tiled' if tiled else 'row'}: f16 outputs {'identical' if same else 'DIFFER'}, "
          f"BFP16 stream {bad} of {idx.size} bytes differ | plain {pms} | bfp {bms}")
    for f in (rf, gf, nf):
        os.remove(f)
    sys.exit(0 if same and not bad else 1)


if __name__ == "__main__":
    main()
