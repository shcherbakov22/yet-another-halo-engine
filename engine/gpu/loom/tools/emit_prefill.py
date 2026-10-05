#!/usr/bin/env python3
"""Helpers for the HAL emitters (emit_prefill_pp.py, emit_decode.py): the GGUF tensor table, the format map and emit().

Naming convention, so the C++ driver needs no manifest:
  <outdir>/gemm_<kind>_<fmt>_<m_tiles>_<k_blocks>.hal
  <outdir>/<fixed>.hal   for the non-GEMM kernels
"""
import os, re, struct, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
LOOM = os.path.abspath(os.path.join(HERE, ".."))
EMIT = os.path.join(LOOM, "emit_hal.py")

# ggml type -> (short name, short name, qk)
FMT = {
    12: ("q4k", "q4k", 256), 13: ("q5k", "q5k", 256), 14: ("q6k", "q6k", 256),
    11: ("q3k", "q3k", 256), 23: ("iq4xs", "iq4xs", 256), 21: ("iq3s", "iq3s", 256),
    18: ("iq3xxs", "iq3xxs", 256), 20: ("iq4nl", "iq4nl", 32),
    17: ("iq2xs", "iq2xs", 256), 8: ("q8_0", "q8_0", 32),
    16: ("iq2xxs", "iq2xxs", 256), 10: ("q2k", "q2k", 256),
}
KSTORE = {"attn_qkv.weight", "attn_gate.weight", "ssm_alpha.weight",
          "ssm_beta.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
          "ffn_gate.weight"}
RESIDUAL = {"attn_output.weight", "ssm_out.weight", "ffn_down.weight"}
SWIGLU = {"ffn_up.weight"}


def parse(model):
    """Return [(name, dims, ggml type)] for every tensor of a GGUF file."""
    f = open(model, "rb"); f.read(8)
    nt, nkv = struct.unpack("<QQ", f.read(16))
    SIZ = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    def rd():
        n = struct.unpack("<Q", f.read(8))[0]; return f.read(n).decode("utf-8", "replace")
    def skip(t):
        if t == 8: rd()
        elif t == 9:
            et = struct.unpack("<I", f.read(4))[0]; n = struct.unpack("<Q", f.read(8))[0]
            if et == 8:
                for _ in range(n): rd()
            else: f.seek(SIZ[et] * n, 1)
        else: f.seek(SIZ[t], 1)
    for _ in range(nkv):
        rd(); t = struct.unpack("<I", f.read(4))[0]; skip(t)
    rows = []
    for _ in range(nt):
        nm = rd(); nd = struct.unpack("<I", f.read(4))[0]
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
        ty = struct.unpack("<I", f.read(4))[0]
        f.read(8)
        rows.append((nm, dims, ty))
    return rows


def sym_of(loomfile):
    text = open(os.path.join(LOOM, loomfile)).read()
    return re.search(r"config\.decl @([A-Za-z0-9_]+)\.m_tiles", text).group(1)


def emit(loomfile, configs, outname, outdir, env=None):
    """Compile one Loom source (relative to engine/gpu/loom, or absolute) at one config to <outdir>/<outname>.
    env: extra environment for this compile (per-kernel compiler options, e.g. LOOM_EXP_LICM)."""
    tmp = os.path.join(outdir, ".emit_tmp")
    os.makedirs(tmp, exist_ok=True)
    src = os.path.join(LOOM, loomfile)
    r = subprocess.run([sys.executable, EMIT, src, tmp] + configs,
                       capture_output=True, text=True, env={**os.environ, **env} if env else None)
    if r.returncode != 0:
        # stderr first: it holds the compiler diagnostics; stdout is progress output.
        log = os.path.join("/tmp", "emit_fail_" + os.path.basename(outname) + ".log")
        with open(log, "w") as fh:
            fh.write("cmd: %s" % " ".join([sys.executable, EMIT, src, tmp] + list(configs)))
            fh.write(chr(10) + "--- stderr ---" + chr(10) + r.stderr
                     + chr(10) + "--- stdout ---" + chr(10) + r.stdout)
        head = (r.stderr.strip() + chr(10) + r.stdout.strip()).strip().splitlines()[:24]
        raise SystemExit("emit failed for " + loomfile + " (full log: " + log + "):"
                         + (chr(10) + "  ").join(head))
    hal = r.stdout.strip().splitlines()[-1]
    dst = os.path.join(outdir, outname)
    subprocess.run(["cp", hal, dst], check=True)
    return dst
