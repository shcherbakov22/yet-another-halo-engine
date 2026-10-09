"""Latency / slot-aware list scheduling of independent op streams for the in-order (locked) AIE2P emitter."""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import hrx_paths
sys.path.insert(0, os.path.join(hrx_paths.HRX, "loom", "py"))
_MODEL = None
_RES = None


def resources():
    """mnemonic -> [(resource, cycle offset, cycles, required)] from the AIE2P itineraries (LLVM scoreboard rule:
    required stages conflict with required and reserved bookings, reserved stages only with required ones)."""
    global _RES
    if _RES is None:
        import loom.target.arch.amd.xdna.aie2p.core_schedule_data as SD
        from loom.target.arch.amd.xdna.aie2p.core_descriptors import AIE2P_CORE_DESCRIPTOR_SET as S
        its = {it.name.lower(): it for it in SD.CORE_SCHEDULE_TABLE.itineraries}
        ds = getattr(S, "descriptors", None) or S
        r = {}
        for d in ds:
            mn = d.asm_forms[0].mnemonic
            if mn in r or "volatile" in d.key:
                continue
            parts = str(getattr(d, "schedule_class", "") or "").split(".")
            it = its.get(parts[4]) if len(parts) > 4 else None
            out, t = [], 0
            for st in (it.stages if it else ()):
                for res in st.resources:
                    if res != "EMPTY_FU" and st.cycles:
                        out.append((res, t, st.cycles, st.kind.name == "REQUIRED"))
                t += st.cycles if st.time_increment == -1 else st.time_increment
            r[mn] = out
        _RES = r
    return _RES
BLOCK = None   # set to a Counter to count why active streams' head ops wait
TRACE = None   # set to [] to record (cycle, stream, op) of the next schedule() calls


def model():
    """mnemonic -> (slot, result ready stage, [read stage per operand in asm order])."""
    global _MODEL
    if _MODEL is None:
        from loom.target.arch.amd.xdna.aie2p.core_descriptors import AIE2P_CORE_DESCRIPTOR_SET as S
        ds = getattr(S, "descriptors", None) or S
        m = {}
        for d in ds:
            f = d.asm_forms[0]
            if f.mnemonic in m or "volatile" in d.key:
                continue
            sc = str(getattr(d, "schedule_class", "") or "")
            parts = sc.split(".")
            slot = parts[5] if len(parts) > 5 else "none"
            ops = {o.field_name: o for o in d.operands}
            ready = max([o.ready_stage for o in d.operands if o.role.name == "RESULT"] or [1])
            reads = [ops[n].read_stage if n in ops else 1 for n in f.operands]
            m[f.mnemonic] = (slot, ready, reads)
        # LS_LAT="mnemonic=ready,...": override result-ready stages (latency experiments)
        import os
        for item in filter(None, os.environ.get("LS_LAT", "").split(",")):
            k_, v_ = item.split("=")
            for mn in list(m):
                if mn == k_ or mn.startswith(k_ + "*") or (k_.endswith("*") and mn.startswith(k_[:-1])):
                    m[mn] = (m[mn][0], int(v_), m[mn][2])
        _MODEL = m
    return _MODEL


FREE = ("concat", "slice", "copy", "low.schedule", "return")
# register units defined per op (vec256 / mbms); concat frees its inputs and defines the sum
VEC = {"vldb.unpack": 2, "vlda.512": 2, "vldb.512": 2, "vshuffle": 2, "vbcst": 2, "vbroadcast": 2, "vadd.16": 2,
       "vsub.16": 2, "vband": 2, "vbor": 2, "vmov.512": 2, "vlda.256": 1, "vldb.256": 1, "vconv.bf16": 2, "max.u16": 2,
       "vunpack": 2}
if os.environ.get("LS_UNITS2", "0") == "1":       # ops the table missed (0 units); off: measured worse (hot 361 -> 411)
    VEC.update({"vshift": 2, "vadd.8": 2, "vadd.32": 2, "vldb.4x": 1})
ACC = {"vlda.acc": 1, "mma.": 4, "mmul.": 4, "vadd.f32x64": 4, "vconv.fp32": 2}
if os.environ.get("LS_UNITS2", "0") == "1":
    ACC.update({"vups.4x": 2, "vadd.acc.integer": 4})


def units(mn, rhs):
    if mn in ("padda", "padda.modifier", "paddb", "padds", "mov.scalar-to-address") or (mn == "copy" and "-> reg<aie2p.ep>" in rhs):
        return ("p", 1)
    for k, v in VEC.items():
        if mn.startswith(k):
            return ("v", v)
    for k, v in ACC.items():
        if mn.startswith(k):
            return ("a", v)
    if mn == "concat":
        if "vec256" in rhs.split("->")[-1]:
            return ("v", int(re.search(r"x(\d)>\s*$", rhs.split("->")[-1]).group(1)) if re.search(r"x(\d)>\s*$", rhs.split("->")[-1]) else 1)
        if "mbms" in rhs.split("->")[-1]:
            return ("a", int(re.search(r"x(\d)>\s*$", rhs.split("->")[-1]).group(1)) if re.search(r"x(\d)>\s*$", rhs.split("->")[-1]) else 1)
    return (None, 0)
NAME = re.compile(r"%[\w.]+")


def parse(op):
    s = op.strip()
    if "=" in s.split("(")[0] or re.match(r"^%[\w.]+(, %[\w.]+)* = ", s):
        lhs, rhs = s.split(" = ", 1)
        defs = NAME.findall(lhs)
    else:
        lhs, rhs, defs = "", s, []
    mm = re.match(r"[\w.\-]+", rhs)
    mn = mm.group(0) if mm else ""
    args = rhs.split(" : ")[0] if mn in ("concat", "slice", "copy") else rhs
    uses = NAME.findall(args[len(mn):]) if mn else []
    return mn, defs, uses


def schedule(streams, window=4, tails=None, budget=None, hard=(24, 20, 3), ext=(0, 0, 0)):
    """Merge streams (lists of op strings; each in its own dependency order) into one issue order.
    tails[i] (list) run after stream i's body and after tails[i - 1] (ordered side effects, e.g. FIFO pushes).
    At most `window` streams are active at a time (bounds register pressure). Returns the op list."""
    M = model()
    RES = resources() if os.environ.get("LS_RES", "0") == "1" else {}
    req, rsv = set(), set()     # scoreboard: (cycle, resource) booked by required / reserved stages
    # Loom orders a load after an earlier store (may-alias local memory): >= ST2LD cycles later (measured 4)
    st2ld = int(os.environ.get("LS_ST2LD", "0"))
    last_store = [-100]
    n = len(streams)
    tails = tails or [[] for _ in range(n)]
    seqs = [list(s) + list(t) for s, t in zip(streams, tails)]
    nbody = [len(s) for s in streams]
    defs_in = set()
    for s in seqs:
        for op in s:
            defs_in.update(parse(op)[1])
    avail = {}          # value -> cycle at which it is ready (at read stage 1)
    # remaining uses and register footprint of every stream-defined value
    nuse, size = {}, {}
    for sq in seqs:
        for op in sq:
            mn, defs, uses = parse(op)
            for u in uses:
                nuse[u] = nuse.get(u, 0) + 1
            cls, sz = units(mn, op)
            for d in defs[:1]:
                size[d] = (cls, sz)
    live = {"v": 0, "a": 0, "p": 0}
    livev = {}

    def peak(sq):
        """Register peak of one stream run alone in program order."""
        cnt, lv, pk = {}, {"v": 0, "a": 0, "p": 0}, {"v": 0, "a": 0, "p": 0}
        for op in sq:
            for u in parse(op)[2]:
                cnt[u] = cnt.get(u, 0) + 1
        for op in sq:
            mn, defs, uses = parse(op)
            c, z = units(mn, op)
            if c and defs:
                lv[c] += z
                pk[c] = max(pk[c], lv[c])
            for u in uses:
                if u in cnt:
                    cnt[u] -= 1
                    if cnt[u] == 0 and u in size and size[u][0]:
                        lv[size[u][0]] -= size[u][1]
        return pk

    def need(sq):
        """need[i][c]: extra units stream sq still requires (beyond its own live at i) from position i on."""
        cnt, lv, traj = {}, {"v": 0, "a": 0, "p": 0}, []
        for op in sq:
            for u in parse(op)[2]:
                cnt[u] = cnt.get(u, 0) + 1
        for op in sq:
            traj.append(dict(lv))
            mn, defs, uses = parse(op)
            c, z = units(mn, op)
            if c and defs:
                lv[c] += z
            snap = dict(lv)
            for u in uses:
                if u in cnt:
                    cnt[u] -= 1
                    if cnt[u] == 0 and u in size and size[u][0]:
                        lv[size[u][0]] -= size[u][1]
            traj[-1] = (traj[-1], snap)
        res = []
        for i in range(len(traj)):
            base = traj[i][0]
            res.append({c: max(0, max(t[1][c] for t in traj[i:]) - base[c]) for c in ("v", "a", "p")})
        res.append({"v": 0, "a": 0, "p": 0})
        return res

    needs = [need(sq) for sq in seqs]
    pks = [peak(sq) for sq in seqs]
    CL = ("v", "a", "p")
    hard = tuple(h - x for h, x in zip(hard, ext))
    worst = {c: max([p[c] for p in pks] or [0]) for c in CL}
    budget = tuple(max(0, hard[j] - worst[c]) for j, c in enumerate(CL))
    pos = [0] * n
    out = []
    cyc = 0
    active = []
    nxt = 0
    tail_turn = 0       # next stream whose tail may run
    while any(pos[i] < len(seqs[i]) for i in range(n)):
        while len(active) < window and nxt < n:
            active.append(nxt)
            nxt += 1
        used = {}
        issued = True
        while issued:
            issued = False
            for i in list(active):
                if pos[i] >= len(seqs[i]):
                    continue
                op = seqs[i][pos[i]]
                if pos[i] >= nbody[i] and tail_turn != i:
                    if BLOCK is not None: BLOCK["tail order"] += 1
                    continue
                mn, defs, uses = parse(op)
                free = mn.startswith(FREE) or mn == ""
                info = M.get(mn, ("none", 1, []))
                slot = "free" if free else info[0]
                # gathers (vldb.4x*) issue in the ldb slot but book load unit A's address port (DM_ADA, itinerary
                # II_VLDB_4x32_*): no vlda in the same bundle
                extra = ("lda",) if mn.startswith("vldb.4x") and os.environ.get("LS_GX", "0") == "1" else ()
                if not free and (used.get(slot, 0) >= 1 or any(used.get(x_, 0) for x_ in extra)):
                    if BLOCK is not None: BLOCK[f"slot {slot}"] += 1
                    continue
                is_load = mn.startswith(("vlda", "vldb", "lda"))
                is_store = mn.startswith(("vst", "st"))
                if is_load and cyc < last_store[0] + st2ld:
                    if BLOCK is not None: BLOCK["store->load"] += 1
                    continue
                book = [] if free else [(cyc + o + j, r_, need) for r_, o, c_, need in RES.get(mn, ()) for j in range(c_)]
                if any(((c_, r_) in req or (need and (c_, r_) in rsv)) for c_, r_, need in book):
                    if BLOCK is not None: BLOCK[f"res {mn}"] += 1
                    continue
                ok = True
                for k, u in enumerate(uses):
                    if u in defs_in:
                        if u not in avail:
                            ok = False
                            break
                        rs = 1
                        if avail[u] + 1 - rs > cyc:
                            ok = False
                            break
                if not ok:
                    if BLOCK is not None: BLOCK[f"operand {mn}"] += 1
                    continue
                cls, sz = units(mn, op)
                freed = {"v": 0, "a": 0, "p": 0}
                for u in set(uses):
                    if u in size and size[u][0] and nuse.get(u, 0) == uses.count(u):
                        freed[size[u][0]] += size[u][1]
                if cls and sz > freed[cls]:
                    # banker's rule: leave room for every older active stream to finish (they push first)
                    older = sum(needs[o][pos[o]][cls] for o in active if o < i)
                    if live[cls] - freed[cls] + sz + older > hard[CL.index(cls)]:
                        if BLOCK is not None: BLOCK[f"budget {cls}"] += 1
                        continue
                for u in uses:
                    nuse[u] -= 1
                    if nuse[u] == 0 and u in size and size[u][0]:
                        live[size[u][0]] -= size[u][1]
                if cls and defs and nuse.get(defs[0], 0):            # a value without uses is dead at its def
                    live[cls] += sz
                    livev[defs[0]] = (cls, sz)
                for u in uses:
                    if nuse.get(u, 1) == 0:
                        livev.pop(u, None)
                out.append(op)
                if TRACE is not None:
                    TRACE.append((cyc, i, op.strip()[:60]))
                if not free:
                    used[slot] = used.get(slot, 0) + 1
                    for x_ in extra:
                        used[x_] = used.get(x_, 0) + 1
                    for c_, r_, need in book:
                        (req if need else rsv).add((c_, r_))
                if is_store:
                    last_store[0] = cyc
                for d in defs:
                    # calibrated: accumulating vec-slot ops (MAC / f32 add) feed any consumer ~9 cycles later
                    lat = 9 + 0 if (not free and info[0] == "vec") else (0 if free else info[1] - 1)
                    avail[d] = cyc + lat
                pos[i] += 1
                if pos[i] == len(seqs[i]):
                    if i == tail_turn:
                        tail_turn += 1
                    active.remove(i)
                elif pos[i] >= nbody[i] and len(tails[i]) and i == tail_turn and pos[i] == len(seqs[i]):
                    tail_turn += 1
                issued = True
            # streams with an empty tail hand the turn on once their body is done
            while tail_turn < n and pos[tail_turn] >= len(seqs[tail_turn]):
                tail_turn += 1
        cyc += 1
        if cyc > 200000:
            info = [(i, pos[i], len(seqs[i]), nbody[i], seqs[i][pos[i]].strip()[:90] if pos[i] < len(seqs[i]) else "-") for i in active]
            raise RuntimeError(f"schedule stuck: live {live} {[(k, v, nuse.get(k)) for k, v in livev.items() if v[0] == 'v']} tail_turn {tail_turn} active {info}")
    schedule.cycles = cyc
    return out
