#!/usr/bin/env python3
"""footprint_gate.py <file.loom> <sym> <fmt> <kind> <m_tiles> <k_blocks> <token_tiles> [B=2048] [masked] [ostride=<rows>]
[kfull=<k_blocks of the whole row>]

Compile the source under the exact config and read each root's declared byte envelope from the compile report.
Refuse (exit 3) unless every envelope fits the buffer loom_forward_pp binds for that root.
An envelope past its buffer does not fault on this GPU: the shader reads unmapped VA and the gfx ring times out.
emit_prefill_pp runs this on every generated GEMM before it emits the HAL.
"""
import atexit, os, shutil, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import loom_preflight as lp
QB = {"q4k": (256, 144), "q5k": (256, 176), "q6k": (256, 210), "q3k": (256, 110), "q2k": (256, 84),
      "iq4xs": (256, 136), "iq3s": (256, 110), "iq3xxs": (256, 98), "iq2xxs": (256, 66), "iq2xs": (256, 74),
      "q8_0": (32, 34), "iq4nl": (32, 18), "f16": (256, 512)}
src, sym, fmt, kind, mt, kb, tt = sys.argv[1:8]
mt, kb, tt = int(mt), int(kb), int(tt)
B = int(sys.argv[8]) if len(sys.argv) > 8 else 2048
# masked: the kernel declares its token count (config "tokens") and the last token tile is partial
# ostride: a column-split kstore / kres (gen_gemm_tile.OSTRIDE) on the full <rows>-row output (and residual)
# mixed ffn "<gate fmt>:<up fmt>": the weight binding holds both tensors
fmts = fmt.split(":")
qk, bpb = QB[fmts[0]]
M, K = mt * 16, kb * qk
MO = next((int(a.split("=")[1]) for a in sys.argv[9:] if a.startswith("ostride=")), M)
# kfull: a K-chunk dequant (gen_gemm_tile.DQ_BFP) reading kb of each <kfull>-block weight row
KBW = next((int(a.split("=")[1]) for a in sys.argv[9:] if a.startswith("kfull=")), kb)
wbytes = M * kb * sum(QB[f][1] for f in fmts) if len(fmts) > 1 else M * KBW * bpb * (2 if kind == "ffn" else 1)
bound = {"weight": wbytes, "input": B * K * 2, "resid": B * MO * 4, "gate": B * M * 4,
         "output": M * K * 2 if kind == "dequant" else B * MO * (2 if kind in ("swiglu", "kqg", "ffn") else 4),
         "gate_out": B * M * 2,
         # the driver's grid buffers (loom_forward_pp.cc): 512 / 256 / 512 / 1024 words
         "grid": max({"iq3s": 2048, "iq3xxs": 1024, "iq2xxs": 2048, "iq2xs": 4096}.get(f, 0) for f in fmts),
         "ksigns": 128,
         "wstage": 17408 * 16 * 2, "ostage": 20480 * B * 4}
sys.path.insert(0, os.path.dirname(HERE))
import hrx_paths  # noqa: E402
e = hrx_paths.env()
# per-call temporaries: emits may run in parallel (engine/tune)
tmp = tempfile.mkdtemp(prefix="footprint_gate_")
atexit.register(shutil.rmtree, tmp, True)
rep = os.path.join(tmp, "report.json")
root = "@" + sym
cmd = [hrx_paths.LOOM_COMPILE, src, "--root=" + root,
       "--target=amdgpu:gfx1151", "--format=amdgpu-hsaco", "--output=" + os.path.join(tmp, "k.hsaco"),
       "--compile-report=details", "--compile-report-output=" + rep,
       f"--config={sym}.m_tiles={mt}", f"--config={sym}.k_blocks={kb}", f"--config={sym}.token_tiles={tt}"]
if "masked" in sys.argv[9:]:
    cmd.append(f"--config={sym}.tokens={B}")
r = subprocess.run(cmd, env=e, capture_output=True, text=True)
if r.returncode:
    print("GATE: compile failed", r.stderr[-800:]); sys.exit(2)
got = lp.declared_envelopes(rep, with_names=True)
if not got:
    print("GATE: no envelopes in report; refusing"); sys.exit(3)
rows, names = got
bad = []
for arg, nbytes in sorted(rows.items()):
    nm = names.get(arg, "?")
    lim = bound.get(nm)
    ok = lim is not None and nbytes <= lim
    if not ok:
        bad.append(f"{nm}: declares {nbytes} > bound {lim}")
if bad:
    print("GATE REFUSED " + os.path.basename(src) + f" m_tiles={mt} k_blocks={kb}: " + "; ".join(bad)); sys.exit(3)
print("gate ok " + ", ".join(f"{names.get(a,'?')}={b}" for a, b in sorted(rows.items())))
