"""List scheduling of independent low-asm op streams for an AIE2P leaf with a locked (in-order) schedule.

A locked leaf issues ops in the authored order; with LOOM_EXP_LOCKED_PACK=1 adjacent independent ops share a bundle.
schedule() merges streams into one order: per cycle at most one op per issue slot, an op only once its inputs are
ready (descriptor latencies), and new registers only while every older stream can still finish (banker's rule over
vec256, accumulator and pointer units). Streams keep their own order; tails (ordered side effects such as FIFO pushes)
run after their stream's body and after the previous stream's tail.
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "external", "hrx",
                                "loom", "py"))

FREE = ("concat", "slice", "copy", "return")
NAME = re.compile(r"%[\w.]+")
# Dependent accumulate through the vec slot (MAC, f32 add) measures ~9 cycles, not the descriptor's bypass stage.
VEC_LATENCY = 9
VEC = {"vldb.unpack": 2, "vlda.512": 2, "vldb.512": 2, "vshuffle": 2, "vbcst": 2, "vbroadcast": 2, "vadd.16": 2,
       "vsub.16": 2, "vband": 2, "vbor": 2, "vmov.512": 2, "vlda.256": 1, "vldb.256": 1, "vconv.bf16": 2, "max.u16": 2}
ACC = {"vlda.acc": 1, "mma.": 4, "mmul.": 4, "vadd.f32x64": 4, "vconv.fp32": 2}
CLASSES = ("v", "a", "p")
_MODEL = None


def model():
    """mnemonic -> (issue slot, result ready stage)."""
    global _MODEL
    if _MODEL is None:
        from loom.target.arch.amd.xdna.aie2p.core_descriptors import AIE2P_CORE_DESCRIPTOR_SET as S
        ds = getattr(S, "descriptors", None) or S
        _MODEL = {}
        for d in ds:
            mn = d.asm_forms[0].mnemonic
            if mn in _MODEL or "volatile" in d.key:
                continue
            parts = str(d.schedule_class or "").split(".")
            ready = max([o.ready_stage for o in d.operands if o.role.name == "RESULT"] or [1])
            _MODEL[mn] = (parts[5] if len(parts) > 5 else "none", ready)
    return _MODEL


def parse(op):
    """(mnemonic, defined values, used values) of one low-asm line."""
    s = op.strip()
    if re.match(r"^%[\w.]+(, %[\w.]+)* = ", s):
        lhs, rhs = s.split(" = ", 1)
        defs = NAME.findall(lhs)
    else:
        rhs, defs = s, []
    m = re.match(r"[\w.\-]+", rhs)
    mn = m.group(0) if m else ""
    args = rhs.split(" : ")[0] if mn in FREE else rhs
    return mn, defs, NAME.findall(args[len(mn):]) if mn else []


def units(mn, op):
    """(register class, units) an op defines; a concat defines its result and frees its inputs."""
    if mn in ("padda", "padda.modifier") or (mn == "copy" and "-> reg<aie2p.ep>" in op):
        return "p", 1
    for k, v in VEC.items():
        if mn.startswith(k):
            return "v", v
    for k, v in ACC.items():
        if mn.startswith(k):
            return "a", v
    if mn == "concat":
        res = op.split("->")[-1]
        n = re.search(r"x(\d)>\s*$", res)
        if "vec256" in res:
            return "v", int(n.group(1)) if n else 1
        if "mbms" in res:
            return "a", int(n.group(1)) if n else 1
    return None, 0


def schedule(streams, window=4, tails=None, ext=(0, 0, 0), hard=(24, 20, 3)):
    """Merge streams (lists of op lines) into one issue order, at most `window` streams in flight.
    ext: units of each class live outside the streams; hard: the stream budget of each class (pointers: 8 less in,
    out, scratch and allocator slack)."""
    M = model()
    n = len(streams)
    tails = tails or [[] for _ in range(n)]
    seqs = [list(s) + list(t) for s, t in zip(streams, tails)]
    nbody = [len(s) for s in streams]
    hard = tuple(h - x for h, x in zip(hard, ext))
    defined = set()
    nuse, size = {}, {}
    for sq in seqs:
        for op in sq:
            mn, defs, uses = parse(op)
            defined.update(defs)
            for u in uses:
                nuse[u] = nuse.get(u, 0) + 1
            if defs:
                size[defs[0]] = units(mn, op)

    def need(sq):
        """need[i][c]: units of class c the stream needs beyond its own live set at position i to finish."""
        cnt, lv, traj = {}, dict.fromkeys(CLASSES, 0), []
        for op in sq:
            for u in parse(op)[2]:
                cnt[u] = cnt.get(u, 0) + 1
        for op in sq:
            before = dict(lv)
            mn, defs, uses = parse(op)
            c, z = units(mn, op)
            if c and defs:
                lv[c] += z
            peak = dict(lv)
            for u in uses:
                if u in cnt:
                    cnt[u] -= 1
                    if cnt[u] == 0 and size.get(u, (None, 0))[0]:
                        lv[size[u][0]] -= size[u][1]
            traj.append((before, peak))
        out = [{c: max(0, max(t[1][c] for t in traj[i:]) - traj[i][0][c]) for c in CLASSES} for i in range(len(traj))]
        return out + [dict.fromkeys(CLASSES, 0)]

    needs = [need(sq) for sq in seqs]
    avail, live = {}, dict.fromkeys(CLASSES, 0)
    pos = [0] * n
    out, active = [], []
    nxt = turn = cyc = 0
    while any(pos[i] < len(seqs[i]) for i in range(n)):
        while len(active) < window and nxt < n:
            active.append(nxt)
            nxt += 1
        used = set()
        issued = True
        while issued:
            issued = False
            for i in list(active):
                if pos[i] >= len(seqs[i]) or (pos[i] >= nbody[i] and turn != i):
                    continue
                op = seqs[i][pos[i]]
                mn, defs, uses = parse(op)
                free = mn.startswith(FREE) or mn == ""
                slot, ready = M.get(mn, ("none", 1))
                if not free and slot in used:
                    continue
                if any(u in defined and avail.get(u, 1 << 30) > cyc for u in uses):
                    continue
                cls, sz = units(mn, op)
                freed = dict.fromkeys(CLASSES, 0)
                for u in set(uses):
                    if size.get(u, (None, 0))[0] and nuse.get(u, 0) == uses.count(u):
                        freed[size[u][0]] += size[u][1]
                if cls and sz > freed[cls]:
                    older = sum(needs[o][pos[o]][cls] for o in active if o < i)
                    if live[cls] - freed[cls] + sz + older > hard[CLASSES.index(cls)]:
                        continue
                for u in uses:
                    nuse[u] -= 1
                    if nuse[u] == 0 and size.get(u, (None, 0))[0]:
                        live[size[u][0]] -= size[u][1]
                if cls and defs:
                    live[cls] += sz
                out.append(op)
                if not free:
                    used.add(slot)
                lat = 0 if free else (VEC_LATENCY if slot == "vec" else ready - 1)
                for d in defs:
                    avail[d] = cyc + lat
                pos[i] += 1
                if pos[i] == len(seqs[i]):
                    active.remove(i)
                issued = True
            while turn < n and pos[turn] >= len(seqs[turn]):
                turn += 1
        cyc += 1
        if cyc > 1 << 20:
            raise RuntimeError("xdna_sched: no stream can progress")
    schedule.cycles = cyc
    return out
