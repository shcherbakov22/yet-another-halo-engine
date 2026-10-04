#!/usr/bin/env python3
"""Autotune the prefill tile GEMMs for one model on this GPU (gfx1151); writes a YAH_TILES table.

usage: engine/tune/tune.py <model.gguf> [--out tiles.json] [--kernels REGEX] [--top N] [--keep DIR]

For every tile GEMM the driver runs (inventory from emit_prefill_pp.gemm_combos), at each token bucket:
  1. enumerate the tile space: token tile BN, waves along tokens, KSUB, decode-ahead; check() drops illegal tiles
  2. compile every candidate in parallel through the emitter's own path (footprint gate included); drop failures and
     tiles that spill
  3. measure all of them on the GPU with real weights (engine/build/gemm_bench), clock-free cycles from HRX counters,
     few repetitions; then the best few again with more (successive halving)
  4. every candidate's output (real token rows) must hash the same as the default tile's, else it is rejected
The table: "tiles" (the full-chunk tile per GEMM), "variants" (up to two narrower tiles per GEMM) and "pick" (per
token bucket, the measured best of those), consumed by emit_prefill_pp.py and LoomPrefill::PickGemm.
Every knob keeps the per-accumulator MMA order, so tuned sets compute the same values as untuned ones.
"""
import argparse
import dataclasses
import glob
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
TOOLS = os.path.join(ROOT, "engine", "gpu", "loom", "tools")
sys.path.insert(0, TOOLS)
sys.path.insert(0, os.path.dirname(TOOLS))
import emit_prefill as E  # noqa: E402
import emit_prefill_pp as EP  # noqa: E402
import gen_gemm_tile as TG  # noqa: E402
import hrx_paths  # noqa: E402

CHUNK = 2048
BUCKETS = (64, 128, 256, 512, 2048)
# aqlprofile (HRX counters mode) comes from a ROCm TheRock build; without it the tuner falls back to wall time.
AQL = os.environ.get("YAH_AQLPROFILE_DIR", "/var/lib/lemonade/.cache/lemonade/bin/therock/gfx1151-7.13.0/lib")
IREE_PROFILE = os.path.join(hrx_paths.BUILD, "runtime/src/iree/tools/iree-profile/iree-profile")
READELF = "/opt/rocm/llvm/bin/llvm-readelf"


@dataclasses.dataclass
class Kernel:
    hal: str          # e.g. gemm_kres_iq3s_320_68
    fmt: str
    kind: str         # kstore / kres / kqg / swiglu
    mt: int
    kb: int
    tensors: list     # this shape's tensors: the bench rotates through up to 6 (as the layers do), so weights come from DRAM
    uses: int         # tensors running this HAL per pass

    @property
    def export(self):
        return "yah_ffn_gemm_" + self.fmt + ("" if self.kind == "kstore" else "_" + self.kind)

    @property
    def weight(self):  # WMMAs per pass at a full chunk: its share of GEMM time
        return self.mt * 16 * self.kb * 256 * CHUNK / 4096 * self.uses


def inventory(model):
    out = []
    for (kind, fmt, port, mt, kb), names in sorted(EP.gemm_combos(E.parse(model)).items()):
        if fmt not in EP.TILE_FMTS or mt < 4:
            continue
        k = EP.gemm_kinds(kind, mt)[-1]  # the one the driver runs
        out.append(Kernel("gemm_%s_%s_%d_%d" % (k, fmt, mt, kb), fmt, k, mt, kb, names, len(names)))
    return out


def space(k, bucket):
    """Legal tiles for kernel k at a token bucket, the default tile first."""
    base = TG.default_tile(k.fmt, k.kind, k.kb)
    seen, out = set(), []

    def add(t):
        key = dataclasses.astuple(t)
        if key in seen:
            return
        try:
            TG.check(t)
        except ValueError:
            return
        if k.mt % t.rowgrp or t.nwave > 16:
            return
        seen.add(key)
        out.append(t)
    add(base)
    bns = [bn for bn in range(32, 385, 32) if bn <= max(2 * bucket, 128)]
    for bn in bns:
        for tn in (16, 32, 64, 96, 128):
            if bn % tn or not 1 <= bn // tn <= 8:
                continue
            for ksub in (32, 64):
                for decahead in ({base.decahead, False} if base.decahead else {False}):
                    add(dataclasses.replace(base, bn=bn, wn=bn // tn, ksub=ksub, decahead=decahead))
    return out


def menu(k, base, exclude=()):
    """The prefill calibration menu of kernel k (engine/model/prefill_calib.hpp picks among these and base while serving):
    one knob of base changed at a time, plus narrow token tiles for short chunks. Legal tiles only, none in exclude."""
    seen = {dataclasses.astuple(t) for t in (base,) + tuple(exclude)}
    out = []
    for kw in (dict(decahead=not base.decahead, ksub=64 if base.decahead else 32),
               dict(ksub=32 if base.ksub == 64 else 64),
               dict(wn=2 if base.wn != 2 else 4),
               dict(bn=128, wn=2), dict(bn=128, wn=4), dict(bn=384, wn=4),
               dict(bn=64, wn=2), dict(bn=32, wn=1, ksub=32)):
        t = dataclasses.replace(base, **kw)
        try:
            TG.check(t)
        except ValueError:
            continue
        if t.decahead and k.fmt not in TG.DECAHEAD_FMTS:
            continue
        if k.mt % t.rowgrp or t.nwave > 16 or dataclasses.astuple(t) in seen:
            continue
        seen.add(dataclasses.astuple(t))
        out.append(t)
    return out


def tables(work):
    """The lookup tables (IQ grids, sign masks) gemm_bench reads, as a prefill set ships them, in work/tables."""
    table_dir = os.path.join(work, "tables")
    os.makedirs(table_dir, exist_ok=True)
    for src, dst in (("grid_iq3s.bin", "grid_iq3s.bin"), ("grid_iq3xxs.bin", "grid_iq3xxs.bin"),
                     ("grid_iq2xxs.bin", "grid_iq2xxs.bin"), ("grid_iq2xs.bin", "grid_iq2xs.bin"),
                     ("ksigns_iq2xs.bin", "ksigns_iq2xxs.bin")):
        shutil.copy(os.path.join(E.LOOM, "tables", src), os.path.join(table_dir, dst))
    return table_dir


def knobs(t, base):
    """The fields of t that differ from the default tile (the YAH_TILES form)."""
    return {f.name: getattr(t, f.name) for f in dataclasses.fields(t) if getattr(t, f.name) != getattr(base, f.name)}


def compile_one(args):
    """Emit one candidate HAL into its own directory; return (hal path, roles) or (None, reason)."""
    k, t, d = args
    masked = CHUNK % t.bn != 0
    try:
        src = TG.gen(k.fmt, k.kind, t, masked)
        os.makedirs(d, exist_ok=True)
        with open(os.devnull, "w") as null:
            stdout = sys.stdout
            sys.stdout = null
            try:
                EP._emit_gen(lambda f, kk: src, t.bn, k.fmt, k.mt, k.kb, CHUNK, k.hal + ".hal", d, k.kind, t.rowgrp, masked)
            finally:
                sys.stdout = stdout
    except BaseException as e:  # the footprint gate exits with SystemExit
        return None, ("gate/emit: " + str(e))[:160]
    hal = os.path.join(d, k.hal + ".hal")
    if not os.path.exists(hal):
        return None, "no HAL"
    notes = subprocess.run([READELF, "--notes", hal], capture_output=True, text=True).stdout
    m = re.search(r"\.private_segment_fixed_size:\s*(\d+)", notes)
    if m and int(m.group(1)) > 0:
        return None, "spills (%s B private)" % m.group(1)
    return hal, roles_of(src)


def roles_of(src):
    """The bindings of a tile GEMM source in order (its launch signature), as gemm_bench names them."""
    return [a.split(":")[0].strip().lstrip("%") for a in re.search(r"\} launch\(([^)]*)\)", src).group(1).split(",")]


def bench(model, table_dir, jobs, work, tag, counters=True):
    """Run jobs [(kernel, tile, hal, roles, tokens, reps)] in one gemm_bench process; return [(cycles, hash)] per job
    (wall ms instead of cycles without counters)."""
    # Refuse dispatches the hardware rejects: the workgroup's waves per SIMD x VGPRs must fit the 1536-entry wave32 file
    # and LDS 64 KB. gfx1151 answers an oversized dispatch with a queue teardown that faults the CP and wedges the GPU.
    for k, t, hal, *_ in jobs:
        notes = subprocess.run([READELF, "--notes", hal], capture_output=True, text=True).stdout
        vgpr = int(re.search(r"\.vgpr_count:\s*(\d+)", notes).group(1))
        lds = int(re.search(r"\.group_segment_fixed_size:\s*(\d+)", notes).group(1))
        per_simd = -(-(t.wm * t.wn) // 4)
        if per_simd * vgpr > 1536 or lds > 65536:
            raise SystemExit(f"refused {hal}: {t.wm * t.wn} waves x {vgpr} VGPRs, LDS {lds} exceed a WGP")
    jf = os.path.join(work, tag + ".jobs")
    with open(jf, "w") as f:
        for k, t, hal, roles, n, reps in jobs:
            f.write("%s %s %s %s %s %d %d %d %d %s\n" % (hal, k.export, ",".join(k.tensors[:6]), k.fmt, k.kind, t.rowgrp, t.bn,
                                                          n, reps,
                                                          ",".join(roles)))
    env = hrx_paths.env()
    counters = counters and os.path.isdir(AQL)
    prof = os.path.join(work, tag + ".irpf")
    if counters:
        env.update(HRX_PROFILE_FILE=prof, HRX_PROFILE_MODE="counters", HRX_PROFILE_COUNTERS="SQ_BUSY_CYCLES",
                   LD_LIBRARY_PATH=AQL + ":" + env["LD_LIBRARY_PATH"])
    # The compiles open the GPU device (iree-run-loom --device=amdgpu): let it settle, or gpu_run.sh refuses a busy GPU.
    for _ in range(120):
        busy = [int(open(p).read()) for p in glob.glob("/sys/class/drm/card*/device/gpu_busy_percent")]
        if max(busy or [0]) <= 10:
            break
        time.sleep(0.5)
    cmd = [os.path.join(ROOT, "engine/run/gpu_run.sh"), "tune", "--", os.path.join(ROOT, "engine/build/gemm_bench"), model,
           table_dir, jf]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    lines = [l.split() for l in r.stdout.splitlines() if l.startswith("job ")]
    if r.returncode or len(lines) != len(jobs):
        raise SystemExit("gemm_bench failed (%s):\n%s" % (tag, (r.stdout + r.stderr)[-1500:]))
    walls = [float(l[3]) for l in lines]
    hashes = [l[5] for l in lines]
    if not counters:
        return list(zip(walls, hashes))
    cyc = {}
    for l in subprocess.run([IREE_PROFILE, "counter", "--format=jsonl", "--counter_samples", prof],
                            capture_output=True, text=True).stdout.splitlines():
        d = json.loads(l)
        if d.get("type") == "counter_sample" and d["counter"] == "SQ_BUSY_CYCLES":
            cyc[d["dispatch_event_id"]] = cyc.get(d["dispatch_event_id"], 0.0) + d["value"] / 20
    ev = []
    for l in subprocess.run([IREE_PROFILE, "dispatch", "--dispatch_events", "--format=jsonl", prof],
                            capture_output=True, text=True).stdout.splitlines():
        d = json.loads(l)
        if d.get("type") == "dispatch_event" and d.get("key", "").startswith("yah_ffn_gemm"):
            ev.append(d)
    ev.sort(key=lambda d: d["event_id"])
    os.remove(prof)
    out, i = [], 0
    for (k, t, hal, roles, n, reps), h in zip(jobs, hashes):
        mine = ev[i + 3:i + 3 + reps]  # after the 2 warmups and the hashed dispatch
        i += 3 + reps
        out.append((statistics.median(cyc.get(d["event_id"], 0.0) for d in mine) if mine else float("inf"), h))
    if i != len(ev):
        raise SystemExit("profile has %d GEMM dispatches, expected %d (%s)" % (len(ev), i, tag))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--out", default="", help="default: engine/tune/tables/<model>.json, which the emitters then use")
    ap.add_argument("--kernels", default="", help="regex over HAL names (default: all)")
    ap.add_argument("--top", type=int, default=4, help="candidates per kernel and bucket kept for the second round")
    ap.add_argument("--keep", default="", help="keep candidate HALs here instead of a temporary directory")
    a = ap.parse_args()

    if not a.out:
        os.makedirs(os.path.join(ROOT, "engine/tune/tables"), exist_ok=True)
        a.out = os.path.join(ROOT, "engine/tune/tables", os.path.splitext(os.path.basename(a.model))[0] + ".json")
    kernels = [k for k in inventory(a.model) if re.search(a.kernels, k.hal)]
    kernels.sort(key=lambda k: -k.weight)
    total = sum(k.weight for k in kernels)
    work = a.keep or tempfile.mkdtemp(prefix="yah-tune-")
    os.makedirs(work, exist_ok=True)
    table_dir = tables(work)
    print("tune: %d GEMMs, work dir %s, %s" % (len(kernels), work,
          "cycles from HRX counters" if os.path.isdir(AQL) else "wall time (no aqlprofile)"), flush=True)

    # 1. candidates per (kernel, bucket); compile each distinct tile once per kernel
    cands, todo = {}, {}
    for ki, k in enumerate(kernels):
        for b in BUCKETS:
            for t in space(k, b):
                key = (ki, dataclasses.astuple(t))
                if key not in todo:
                    todo[key] = (k, t, os.path.join(work, "k%02d" % ki, "c%03d" % len(todo)))
                cands.setdefault((ki, b), []).append(key)
    print("tune: compiling %d candidates" % len(todo), flush=True)
    with ProcessPoolExecutor(os.cpu_count()) as pool:
        built = dict(zip(todo, pool.map(compile_one, todo.values())))
    ok = {key: v for key, v in built.items() if v[0]}
    reasons = {}
    for v in built.values():
        if not v[0]:
            reasons[v[1].split(":")[0]] = reasons.get(v[1].split(":")[0], 0) + 1
    print("tune: %d compiled, %d dropped %s" % (len(ok), len(built) - len(ok), reasons or ""), flush=True)

    # 2. round one: every compiled candidate, few reps; round two: the best few, more reps
    result = {}
    for rnd, reps, keep in ((1, 3, None), (2, 12, a.top)):
        jobs, idx = [], []
        for (ki, b), keys in cands.items():
            keys = [key for key in keys if key in ok]
            if keep:
                keys = sorted(keys, key=lambda key: result[(key, b)][0])[:keep]
            for key in keys:
                k, t, _ = todo[key]
                jobs.append((k, t, ok[key][0], ok[key][1], b, reps))
                idx.append((key, b))
        print("tune: round %d, %d measurements" % (rnd, len(jobs)), flush=True)
        for (key, b), r in zip(idx, bench(a.model, table_dir, jobs, work, "r%d" % rnd)):
            result[(key, b)] = r

    # 3. per kernel: the tiles, checked against the default tile's output
    table = {"tiles": {}, "variants": {}, "pick": {}}
    gain = 0.0
    for ki, k in enumerate(kernels):
        base = TG.default_tile(k.fmt, k.kind, k.kb)
        best = {}
        for b in BUCKETS:
            keys = [key for key in cands[(ki, b)] if (key, b) in result]
            ref = result.get(((ki, dataclasses.astuple(base)), b))
            good = []
            for key in keys:
                cyc, h = result[(key, b)]
                if ref and h != ref[1]:
                    print("tune: REJECT %s %s: output differs from the default tile" % (k.hal, todo[key][1]))
                    continue
                good.append((cyc, key))
            if good:
                best[b] = min(good)
        full = todo[best[CHUNK][1]][1]
        if knobs(full, base):
            table["tiles"][k.hal] = knobs(full, base)
        base_cyc = result[((ki, dataclasses.astuple(base)), CHUNK)][0]
        gain += k.weight / total * (1 - best[CHUNK][0] / base_cyc)
        # narrow variants: the best tiles of the small buckets that are narrower than the full-chunk tile
        variants = []
        for b in BUCKETS[:-1]:  # smallest bucket first: it decides a token tile's variant when two share it
            t = todo[best[b][1]][1]
            if t.bn < full.bn and t.bn not in [v.bn for v in variants]:
                variants.append(t)
        variants = sorted(variants, key=lambda t: t.bn)[:2]
        if variants:
            table["variants"][k.hal] = [knobs(t, base) | {"bn": t.bn} for t in variants]
            # per bucket, the measured best among the shipped tiles
            shipped = [full] + variants
            picks = {}
            for b in BUCKETS[:-1]:
                opts = [(result[((ki, dataclasses.astuple(t)), b)][0], t.bn) for t in shipped
                        if ((ki, dataclasses.astuple(t)), b) in result]
                if opts:
                    picks[str(b)] = min(opts)[1]
            table["pick"][k.hal] = picks
        print("tune: %-28s %5.1f%% of GEMM work  full chunk %s%s" % (
            k.hal, 100 * k.weight / total, ("BN %d (%+.1f%% cycles)" % (full.bn, 100 * (best[CHUNK][0] / base_cyc - 1)))
            if knobs(full, base) else "default", ("  narrow " + "/".join(str(t.bn) for t in variants)) if variants else ""),
            flush=True)
    with open(a.out, "w") as f:
        json.dump(table, f, indent=1, sort_keys=True)
    print("tune: full-chunk GEMM cycles %+.2f%% (work-weighted); table -> %s" % (-100 * gain, a.out))
    if not a.keep:
        shutil.rmtree(work)


if __name__ == "__main__":
    main()
