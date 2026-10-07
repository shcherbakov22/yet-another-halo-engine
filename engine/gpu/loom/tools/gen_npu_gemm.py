#!/usr/bin/env python3
"""Emit the NPU cascade GEMM (XDNA2 array program + core leaves, Loom low asm) for loom-compile.

C[M x N] = A[M x K] . W[N x K]^T in bfp16ebs8 with f32 accumulation, N = TN = 80 per NPU column (cols <= 8).
Each column is a cascade of 4 compute rows: the head (top row) and two mids compute K-slice partials of every 16 x 16
sub-tile and pass them down the accumulator cascade; the tail (bottom row) adds its slice and the running C.
K is split per pass over the rows (ks[r] 8-wide k-blocks each, ks = [head, mid, mid, tail]); passes repeat until K.
Operand streams are the layouts of gen_bfp16_encode.py (act: [slice][M block][pass][slab], wgt: [col][slice][pass][slab]).
C: per column [M block][4 segments][sub-tile][chain][8 x 8 bf16] (see npu_gemm_check.unpack_c); the tail accumulates f32 and
packs each segment to bf16 in place on its last pass, and the C ring sends the packed half (LOOM_EXP_LS_SEND_PITCH=2).
The kernel is bound by the operand streams into the compute tiles (36 * (1 / MP + 1 / NP) bytes per MMA over two
S2MM streams of ~7.7 B/cycle): NP = 5 needs 20% less activation stream than 4; a column's whole weight panel must fit
its 512 KB memory tile, which caps NP at 5 for K = 5120.

Dataflow: weights stage once per column in its memory tile and replay per M block; each row's activations are one
stream, staged in a memory tile and multicast along the row. Activations arrive one 16-row slab per record into a
leaf-synchronized ring (the leaf acquires and releases slabs, constrain.leaf_sync); the tail holds C as 4
leaf-synchronized segments, acquired on a segment's first pass and released after its last.
A 5-pass call replays its passes in groups (gen_bfp16_encode.GROUPS: every M block over passes 0-1, then over 2-4), each
group a C partial ([col][group][M block]..., summed by the GPU unpack): a group's memory-tile slots refill with the next
call's weights while the other group computes, so calls stream back to back (Loom LOOM_EXP_PANEL_GROUPS /
LOOM_EXP_PANEL_STREAM: counting locks, a looping fill, invocations [setup | even push | odd push | waits] and fills
paced a few columns at a time so they do not starve the activation streams of DRAM; LoomNpu submits the pushes one call
ahead of the waits).
Needs HRX patch 0013; compile with LOOM_ENV.
"""
import dataclasses
import sys

import gen_bfp16_encode as GE

MBMS4 = "(reg<aie2p.mbms>, reg<aie2p.mbms>, reg<aie2p.mbms>, reg<aie2p.mbms>) -> reg<aie2p.mbms x4>"
ORDER_C = (0, 2, 1, 3)   # MMA order of the 4 chains inside a k step (a-row, w-col: c >> 1, c & 1)
LAGU = 9                 # pops issue 9 MMAs ahead of their first use (the pop -> operand latency)
WB = 9                   # pops of the next sub-tile that ride in the cascade-write block (head, mids)
MP, NP = 4, 5            # 16-row slabs per M block / per NPU column
TM, TN = 16 * MP, 16 * NP
KS = (35, 35, 35, 23)    # production k-blocks per pass and K-slice row: K = 1024 per pass
# loom-compile environment for the NPU image
LOOM_ENV = dict(LOOM_EXP_LOCKED_PACK="1", LOOM_EXP_LATE_STORAGE="1", LOOM_EXP_LS_SEND_PITCH="2",
                LOOM_EXP_PANEL_GROUPS=",".join(map(str, GE.GROUPS)), LOOM_EXP_PANEL_STREAM="1")


@dataclasses.dataclass(frozen=True)
class Config:
    cols: int           # NPU columns, N = 64 * cols
    nb: int             # 64-row M blocks, M = 64 * nb
    ks: tuple           # k-blocks per pass for (head, mid, mid, tail)
    passes: int         # K = 8 * passes * sum(ks)
    mu: int = 2         # M slabs per loop iteration of a leaf
    acap: int = 2       # activation ring slabs, head and mids
    acap_t: int = 2     # activation ring slabs, tail (its memory also holds C)
    entry: str = "npu_gemm"


def groups(cfg):
    """The call's replay groups of passes, () for a single group."""
    g = GE.pass_groups(cfg.passes)
    assert len(g) <= 2
    return g if len(g) > 1 else ()


def slab(ks):
    return (144 * ks + 63) // 64 * 64


def stream_bytes(cfg):
    """(activation, weight, C) binding sizes."""
    a = sum(cfg.nb * cfg.passes * MP * slab(k) for k in cfg.ks)
    w = cfg.cols * sum(cfg.passes * NP * slab(k) for k in cfg.ks)
    return a, w, max(1, len(groups(cfg))) * cfg.cols * cfg.nb * MP * NP * 2 * 256


def bump(L, dst, src, off):
    cur, i = src, 0
    while off > 0:
        step = min(off, 448)
        off -= step
        nxt = dst if off == 0 else f"{dst}_b{i}"
        i += 1
        L.append(f"  {nxt} = padda {cur}, {step}")
        cur = nxt
    if cur != dst:
        L.append(f"  {dst} = copy {cur} : reg<aie2p.ep> -> reg<aie2p.ep>")


def array_program(L, cfg):
    e = L.append
    rows, roles = len(cfg.ks), ("head",) + ("mid",) * (len(cfg.ks) - 2) + ("tail",)
    cols, nb, P = cfg.cols, cfg.nb, cfg.passes
    e("aie2p.target<array> @array_target\naie2p.target<core> @core_target\n")
    e(f"low.func.def public retain target<amd.xdna.aie2p.array>(@array_target) abi(array_program) @{cfg.entry}() asm {{")
    for k, v in (("zero", 0), ("one", 1), ("two", 2), ("nw", rows * cols), ("rec", nb * P), ("orec", nb), ("fold", P)):
        e(f"  %{k} = constant.u32 {v} : reg<aie2p.array.scalar : index>")
    e("  %origin = constant.u64 0 : reg<aie2p.array.offset : offset>")
    e("  %workers = group %nw")
    e('  %ab = binding 0, "read"')
    e('  %wb = binding 1, "read"')
    e('  %cb = binding 2, "write"')
    for i in range(max(cols, 2, MP) + 6):
        e(f"  %n{i} = constant.u32 {i} : reg<aie2p.array.scalar : index>")
    panel = sum(P * NP * slab(k) for k in cfg.ks)
    for c in range(cols):
        for r, role in enumerate(roles):
            e(f"  %lane{c}_{r} = constant.u32 {c * rows + r} : reg<aie2p.array.scalar : index>")
            e(f"  %k{c}_{r} = worker %workers, %lane{c}_{r}, @{role}")
            e(f"  constrain.location %k{c}_{r}, %n{c}, %n{2 + rows - 1 - r}")
    # C first: rings are placed in channel order, so the tail's C claims whole banks before its input rings
    e(f"  %ncols = constant.u32 {cols} : reg<aie2p.array.scalar : index>")
    seg = MP * NP * 4 * 256 // MP
    ng = max(1, len(groups(cfg)))   # C segments per M block and group
    e(f"  %nseg = constant.u32 {ng * nb * MP} : reg<aie2p.array.scalar : index>")
    e(f"  %c_all = receiver %cb, 0 : reg<aie2p.array.receiver : tile<{cols}x{ng * nb * MP}x{seg // 8}xi32>>")
    for c in range(cols):
        e(f"  %cr{c} = partition.receiver %c_all, %origin, %n{c}, %ncols : reg<aie2p.array.receiver : tile<{seg // 8}xi32>>")
        e(f"  %sc{c} = sender %k{c}_{rows - 1}, 2 : reg<aie2p.array.sender : tile<{seg // 8}xi32>>")
        e(f"  %chc{c} = channel %sc{c}, %cr{c}, %n{MP}, %nseg : reg<aie2p.array.channel : tile<{seg // 8}xi32>>")
        e(f"  constrain.leaf_sync %chc{c}")
    # weights: per-column panels [slice][pass][record], staged in that column's memory tile, replayed per M block
    for c in range(cols):
        for r in range(rows):
            wrec = NP * slab(cfg.ks[r])
            w_off = c * panel + sum(P * NP * slab(k) for k in cfg.ks[:r])
            e(f"  %wo{c}_{r} = constant.u64 {w_off} : reg<aie2p.array.offset : offset>")
            e(f"  %ws{c}_{r} = sender %wb, 0 : reg<aie2p.array.sender : tile<{nb}x{P}x{wrec // 4}xi32, "
              f"#encoding.layout.strided<strides=[0, {wrec // 4}, 1]>>>")
            e(f"  %wv{c}_{r} = view.sender %ws{c}_{r}, %wo{c}_{r} : reg<aie2p.array.sender : tile<{wrec // 4}xi32>>")
            e(f"  %rw{c}_{r} = receiver %k{c}_{r}, 1 : reg<aie2p.array.receiver : tile<{wrec // 4}xi32>>")
            e(f"  %chw{c}_{r} = channel %wv{c}_{r}, %rw{c}_{r}, %two, %rec : reg<aie2p.array.channel : tile<{wrec // 4}xi32>>")
            e(f"  constrain.stage %chw{c}_{r}, %n{c if cols > 1 else 1}")
    # activations: one stream per row, one slab per record, multicast along the row (first branch rotates)
    for r in range(rows):
        fa = slab(cfg.ks[r])
        a_off = sum(nb * P * MP * slab(k) for k in cfg.ks[:r])
        e(f"  %ao{r} = constant.u64 {a_off} : reg<aie2p.array.offset : offset>")
        e(f"  %as{r} = sender %ab, 0 : reg<aie2p.array.sender : tile<{nb * P * MP}x{fa // 4}xi32>>")
        e(f"  %a{r} = view.sender %as{r}, %ao{r} : reg<aie2p.array.sender : tile<{fa // 4}xi32>>")
        e(f"  %arec{r} = constant.u32 {nb * P * MP} : reg<aie2p.array.scalar : index>")
        for j in range(cols):
            c = (2 * r + j) % cols
            acap = cfg.acap_t if roles[r] == "tail" else cfg.acap
            e(f"  %ra{c}_{r} = receiver %k{c}_{r}, 0 : reg<aie2p.array.receiver : tile<{fa // 4}xi32>>")
            e(f"  %cha{c}_{r} = channel %a{r}, %ra{c}_{r}, %n{acap}, %arec{r} : reg<aie2p.array.channel : tile<{fa // 4}xi32>>")
            e(f"  constrain.leaf_sync %cha{c}_{r}")
            e(f"  constrain.stage %cha{c}_{r}, %n{(2 * r) % cols if cols > 1 else 2}")
    e("  return\n}\n")


def leaf(L, cfg, role, ks):
    e = L.append
    tail = role == "tail"
    fa = slab(ks)
    acap = cfg.acap_t if tail else cfg.acap
    a_adv = acap == MP   # a ring of MP slabs is addressed like a whole record: the base advances per iteration
    assert MP % cfg.mu == 0 and (a_adv or cfg.mu % acap == 0), "ring slot of a row must be static per iteration"
    e(f"low.func.def schedule(locked) target<amd.xdna.aie2p.core>(@core_target) abi(object_function) @{role}() asm {{")
    e("  %a = resource<native_pointer> {index = 0, source_type = buffer} : reg<aie2p.ep>")
    e("  %w = resource<native_pointer> {index = 1, source_type = buffer} : reg<aie2p.ep>")
    if tail:
        e("  %o = resource<native_pointer> {index = 2, source_type = buffer} : reg<aie2p.ep>")
    if role != "head":
        e("  set.scd-enable 1")
    if role != "tail":
        e("  set.mcd-enable 1")
    else:
        e("  set.rounding 12")   # round to nearest even (the bf16 C)
    e("  %conf = mova.i32 780")
    e("  %am1 = mova.i32 -1")
    e("  %addmode = mova.i32 60")
    if role != "head":
        e("  %qsel0 = mova.i32 0")
    e("  %one = mova.i32 1")
    e("  %zero = mova.i32 0")
    e(f"  %nmp = mova.i32 {MP // cfg.mu}")
    e(f"  %nnp = mova.i32 {NP}")
    e("  %k144 = mova.i32 16")
    e(f"  %kks = mova.i32 {fa // 16}")
    e("  %fstep = mul %k144, %kks")   # slab stride
    e("  %mf = mov.modifier %fstep")
    e("  %k512 = mova.i32 512")
    e("  %k2 = mova.i32 2")
    e("  %cstep = mul %k512, %k2")
    e("  %mcs = mov.modifier %cstep")
    e("  %mz = mov.modifier %zero")
    if tail:
        # pass counter in private storage; C segments are acquired on the first pass and released after the last
        e("  %cst = storage {byte_alignment = 64, byte_length = 64} : low.storage<private>")
        e("  %cnp = storage_address %cst : low.storage<private> -> reg<aie2p.ep>")
        grp = groups(cfg)
        if grp:
            # %cnt: the pass within the group; %gs: the group (0 / 1); %gd: the passes done in the group's phase
            ga, gb = grp
            e("  %cnt = lda %cnp, 0")
            e("  %gs = lda %cnp, 4")
            e("  %gd = lda %cnp, 8")
            e(f"  %gdiff = mova.i32 {gb - ga}")
            e(f"  %ga = mova.i32 {ga}")
            e("  %gsd = mul %gs, %gdiff")
            e("  %gsz = add.rr %ga, %gsd")
            e("  %first = lt %cnt, %one")
            e(f"  %gam2 = mova.i32 {ga - 2}")
            e("  %pm2 = add.rr %gam2, %gsd")
            e("  %clast = lt %pm2, %cnt")
        else:
            e(f"  %npass = mova.i32 {cfg.passes}")
            e("  %cnt = lda %cnp, 0")
            e("  %first = lt %cnt, %one")
            e(f"  %pm2 = mova.i32 {cfg.passes - 2}")
            e("  %clast = lt %pm2, %cnt")
        # first pass: C loads read a 1 KB zero buffer with step 0
        e("  %zst = storage {byte_alignment = 64, byte_length = 1024} : low.storage<private>")
        e("  %zbp = storage_address %zst : low.storage<private> -> reg<aie2p.ep>")
        e("  low.cond_br %first, ^zfill, ^nofill : reg<aie2p.er>")
        e("^nofill:")
        e("  %olc = copy %o : reg<aie2p.ep> -> reg<aie2p.ep>")
        bump(L, "%olb", "%olc", 512)
        e("  low.br ^start(%olb: reg<aie2p.ep>)")
        e("^zfill:")
        e("  %zacc = acc.clear.f32x64")
        e("  %zp = slice %zacc[0] : reg<aie2p.mbms x4> -> reg<aie2p.mbms>")
        e("  %zbc = copy %zbp : reg<aie2p.ep> -> reg<aie2p.ep>")
        bump(L, "%zbb", "%zbc", 512)
        for q in range(16):
            e(f"  vst.acc %zp, %zbb, {64 * q - 512}")
        e("  low.br ^start(%zbb: reg<aie2p.ep>)")
        e("^start(%plb: reg<aie2p.ep>):")
        if grp:
            e("  %cn1 = add.rr %cnt, %one")
            e("  %cwrap = lt %cn1, %gsz")
            e("  %cnn = mul %cn1, %cwrap")
            e("  st %cnn, %cnp, 0")
            e(f"  %gnb = mova.i32 {cfg.nb}")
            e("  %gplen = mul %gnb, %gsz")
            e("  %gd1 = add.rr %gd, %one")
            e("  %gmore = lt %gd1, %gplen")   # 1 while the group's phase continues
            e("  %gdn = mul %gd1, %gmore")
            e("  st %gdn, %cnp, 8")
            e("  %gsw = lt %gmore, %one")     # 1 at the phase switch: s <- 1 - s
            e("  %gs2 = add.rr %gs, %gs")
            e("  %gt = mul %gs2, %gsw")
            e("  %gsa = add.rr %gs, %gsw")
            e("  %gtn = mul %gt, %am1")
            e("  %gsn = add.rr %gsa, %gtn")
            e("  st %gsn, %cnp, 4")
        else:
            e("  %cn1 = add.rr %cnt, %one")
            e("  %cwrap = lt %cn1, %npass")
            e("  %cnn = mul %cn1, %cwrap")
            e("  st %cnn, %cnp, 0")
    body(L, cfg, role, ks, a_adv, acap)


def body(L, cfg, role, ks, a_adv, acap):
    """The locked stream: per sub-tile MMA + pop bundles, then the cascade boundary.
    head: chains start with mmul; after each sub-tile, 16 cascade writes (the next sub-tile's pops ride with them)
    mid:  mmul at k 0, then k 1-4 add the incoming partial quarter by quarter (fused cascade MMA); writes like the head
    tail: chains start from C, k 0-3 add the incoming quarters; after each sub-tile C is stored and the next loaded"""
    e = L.append
    tail = role == "tail"
    mid = role == "mid"
    NT = NP * cfg.mu      # sub-tiles per iteration, M-slab major
    e("  %lo_a = vlda.load-fifo.low512 %a, 0")
    e("  %lo_w = vlda.load-fifo.low512 %w, 0")
    e("  %fa0 = vlda.load-fifo.high512 %a, %lo_a, 64")
    e("  %fw0 = vlda.load-fifo.high512 %w, %lo_w, 64")
    if tail:
        e("  %oc = copy %o : reg<aie2p.ep> -> reg<aie2p.ep>")
        bump(L, "%ob", "%oc", 512)
        po0, pct = "%ob", "reg<aie2p.ep>"
        # C load pointer: the zero buffer with step 0 on an M block's first pass, the C slot otherwise
        e("  %nf = lt %zero, %cnt")
        e("  %lstep = mul %cstep, %nf")
        e("  %mls = mov.modifier %lstep")
        pla, plp = ", %plb: reg<aie2p.ep>", ", %pl: reg<aie2p.ep>"
    else:
        po0, pct = "%zero", "reg<aie2p.er>"
        pla = plp = ""
    e(f"  low.br ^outer(%zero: reg<aie2p.er>, %a: reg<aie2p.ep>, {po0}: {pct}, %fa0: reg<aie2p.eldfiforeg>, %fw0: reg<aie2p.eldfiforeg>{pla})")
    e(f"^outer(%mp: reg<aie2p.er>, %pa: reg<aie2p.ep>, %po: {pct}, %fa: reg<aie2p.eldfiforeg>, %fw: reg<aie2p.eldfiforeg>{plp}):")
    e("  %omore = lt %mp, %nmp")
    e("  low.cond_br %omore, ^obody, ^exit : reg<aie2p.er>")
    e("^obody:")
    pc = ["%po"] + [f"%pc{t}" for t in range(1, NT)]
    pl = (["%pl"] + [f"%pl{t}" for t in range(1, NT)]) if tail else pc
    fifo = {"a": "%fa", "w": "%fw"}

    def units_for(t):
        x = f"s{t}"
        tw, tm = t % NP, t // NP
        if tw == 0:
            wp, wsrc = [], "%w"
        else:
            wp = [f"  %{x}w0 = copy %w : reg<aie2p.ep> -> reg<aie2p.ep>"]
            for j in range(tw):
                wp.append(f"  %{x}w{j + 1} = padds.modifier %{x}w{j}, %mf")
            wsrc = f"%{x}w{tw}"
        asrc, ap = "%pa", []
        sm = tm if a_adv else tm % acap
        if sm:
            ap = [f"  %{x}am0 = copy %pa : reg<aie2p.ep> -> reg<aie2p.ep>"]
            for j in range(sm):
                ap.append(f"  %{x}am{j + 1} = padds.modifier %{x}am{j}, %mf")
            asrc = f"%{x}am{sm}"
        if tw == 0:   # the previous row's slab is fully popped; wait for this row's slab
            ap = (["  rel %one, 0"] if t > 0 else []) + ["  acq %am1, 0"] + ap
        positions = [f"  %{x}qa = mova.fifo.position 0", f"  %{x}qw = mova.fifo.position 0"]
        # the tail is one pointer short for early W derivation (it has slack)
        early = (wp + positions) if not tail and t > 0 else []
        init = ap + [f"  %{x}pa = copy {asrc} : reg<aie2p.ep> -> reg<aie2p.eps>"] + ([] if early else wp) + [
                f"  %{x}pw = copy {wsrc} : reg<aie2p.ep> -> reg<aie2p.eps>"] + ([] if early else positions) + [
                f"  %{x}pa1, %{x}fa1, %{x}qa1 = vlda.fill.512 %{x}pa, {fifo['a']}, %{x}qa",
                f"  %{x}pw1, %{x}fw1, %{x}qw1 = vldb.fill.512 %{x}pw, {fifo['w']}, %{x}qw"]
        st = {"pa": f"%{x}pa1", "fa": f"%{x}fa1", "qa": f"%{x}qa1", "pw": f"%{x}pw1", "fw": f"%{x}fw1", "qw": f"%{x}qw1"}
        units, carry, n = [], [], 0
        for k in range(ks):
            for i, (ln, h) in enumerate([("a", 0), ("w", 0), ("a", 1), ("w", 1)]):
                kk, nm = ("a", "a") if ln == "a" else ("b", "w")
                line = (f"  %{x}{nm}{k}_{h}, %{x}p{ln}{k}_{h}, %{x}f{ln}{k}_{h}, %{x}q{ln}{k}_{h} = "
                        f"vld{kk}.pop.bfp16ebs8 {st['p' + ln]}, {st['f' + ln]}, {st['q' + ln]}")
                st["p" + ln], st["f" + ln], st["q" + ln] = f"%{x}p{ln}{k}_{h}", f"%{x}f{ln}{k}_{h}", f"%{x}q{ln}{k}_{h}"
                units.append(carry + [line])
                carry = []
                if i % 2 == 0:
                    continue
                n += 1
                if n % 8 == 0 and n < 2 * ks:   # FIFO refill: pops fetch 512 bits and consume 576
                    fills = []
                    for ln2, kk2 in (("a", "a"), ("w", "b")):
                        fills.append(f"  %{x}rp{ln2}{n}, %{x}rf{ln2}{n}, %{x}rq{ln2}{n} = vld{kk2}.fill.512 {st['p' + ln2]}, {st['f' + ln2]}, {st['q' + ln2]}")
                        st["p" + ln2], st["f" + ln2], st["q" + ln2] = f"%{x}rp{ln2}{n}", f"%{x}rf{ln2}{n}", f"%{x}rq{ln2}{n}"
                    units[-1].append(fills[0])
                    carry = [fills[1]]
        if carry:
            units.append(carry)
        fifo["a"], fifo["w"] = st["fa"], st["fw"]
        return early, init, units

    # each unit (a pop, maybe a fill) rides with one MMA; at a sub-tile boundary of the head and mids the next
    # sub-tile's FIFO init fills the MMA -> cascade-write latency window and its first WB pops pair with the writes
    slots, prologue, wblock = {}, [], {}
    wb = 0 if tail else WB
    for t in range(NT):
        base = t * ks * 4 - LAGU
        early, init, units = units_for(t)
        if wb and t > 0:
            s0 = t * ks * 4
            for i, line in enumerate(early):
                slots.setdefault(base - len(early) + i, []).append(line)
            wblock[t - 1] = [init] + units[:wb]
            for u in range(wb, len(units)):
                slots.setdefault(s0 + u - wb, []).extend(units[u])
            continue
        units[0] = init + units[0]
        for i, line in enumerate(early):
            g = base - (len(early) - i)
            (prologue if g < 0 else slots.setdefault(g, [])).append(line)
        for u, lines in enumerate(units):
            g = base + u
            (prologue if g < 0 else slots.setdefault(g, [])).extend(lines)

    def emit_writes(t):
        blk = wblock.get(t, [[]])
        for line in blk[0]:
            e(line)
        units = blk[1:]
        wire = [(c, q) for q in range(4) for c in ORDER_C]   # the order the consumer's fused MMAs read
        for i, (c, q) in enumerate(wire):
            e(f"  %o{t}_{c}_{q} = slice {cur[c]}[{q}] : reg<aie2p.mbms x4> -> reg<aie2p.mbms>")
            e(f"  vmov.mcd.acc %o{t}_{c}_{q}")
            for line in (units[i] if i < len(units) else []):
                e(line)
        for unit in units[len(wire):]:
            for line in unit:
                e(line)

    n_init = 6
    for line in prologue[:n_init]:
        e(line)
    rest = prologue[n_init:]
    cur = [None] * 4
    if tail:
        e("  %cqa0 = copy %first : reg<aie2p.er> -> reg<aie2p.mr26_lock>")
        e("  acq.cond %am1, %cqa0, 2")
        lds = [f"  %l0_{c}_{q} = vlda.acc {pl[0]}, {256 * c + 64 * q - 512}" for c in range(4) for q in range(4)]
        for i, line in enumerate(lds):
            e(line)
            if i < len(rest):
                e(rest[i])
        for line in rest[len(lds):]:
            e(line)
        for c in range(4):
            e(f"  %z0_{c} = concat(%l0_{c}_0, %l0_{c}_1, %l0_{c}_2, %l0_{c}_3) : {MBMS4}")
            cur[c] = f"%z0_{c}"
    else:
        for line in rest:
            e(line)
    for t in range(NT):
        for k in range(ks):
            fq = k if tail else (k - 1 if mid else -1)   # incoming quarter added at this k step
            if fq == 0:
                e(f"  %q{t}_{k} = copy %qsel0 : reg<aie2p.er> -> reg<aie2p.mr31_scd>")
                sel = f"%q{t}_{k}"
            for ci, c in enumerate(ORDER_C):
                g = (t * ks + k) * 4 + ci
                a_op, w_op = f"%s{t}a{k}_{c >> 1}", f"%s{t}w{k}_{c & 1}"
                if 0 <= fq < 4:
                    # the k step's last fused MMA advances r31 to the next quarter
                    if ci == 3 and fq < 3:
                        e(f"  %m{t}_{k}_{c}, %q{t}_{k}n = maddmac.bfp.ex-ex.scd.increment {cur[c]}, {sel}, {a_op}, {w_op}, %conf")
                        sel = f"%q{t}_{k}n"
                    else:
                        e(f"  %m{t}_{k}_{c} = maddmac.bfp.ex-ex.scd {cur[c]}, {sel}, {a_op}, {w_op}, %conf")
                elif not tail and k == 0:
                    e(f"  %m{t}_{k}_{c} = mmul.bfp16ebs8.m8n8k8 {a_op}, {w_op}, %conf")
                else:
                    e(f"  %m{t}_{k}_{c} = mma.bfp16ebs8.m8n8k8 {cur[c]}, {a_op}, {w_op}, %conf")
                cur[c] = f"%m{t}_{k}_{c}"
                for line in slots.get(g, []):
                    e(line)
        nxt = t + 1 < NT
        if not tail:
            emit_writes(t)
            continue
        # store sub-tile t, load the next sub-tile's C (or zeros on the first pass)
        row_end = (t + 1) % NP == 0
        if row_end and nxt:
            e(f"  %cqa{t + 1} = copy %first : reg<aie2p.er> -> reg<aie2p.mr26_lock>")
            e(f"  acq.cond %am1, %cqa{t + 1}, 2")
        if nxt:   # loads go through pl, so pc steps after the stores (one fewer live pointer)
            e(f"  %plc{t + 1} = copy {pl[t]} : reg<aie2p.ep> -> reg<aie2p.ep>")
            e(f"  %pl{t + 1} = padds.modifier %plc{t + 1}, %mls")
        for c in range(4):
            for q in range(4):
                if nxt:
                    e(f"  %l{t + 1}_{c}_{q} = vlda.acc {pl[t + 1]}, {256 * c + 64 * q - 512}")
                e(f"  %o{t}_{c}_{q} = slice {cur[c]}[{q}] : reg<aie2p.mbms x4> -> reg<aie2p.mbms>")
                e(f"  vst.acc %o{t}_{c}_{q}, {pc[t]}, {256 * c + 64 * q - 512}")
        def convert(base, back, group):
            """Last pass: pack the segment's f32 C (NP sub-tiles) in place to bf16 in its first half, which the C ring sends
            (LOOM_EXP_LS_SEND_PITCH=2). base is a store pointer back bytes past the segment's slot + 512. No pointer
            register is free here, so base itself walks the destination and is rebuilt for the join. group: quarters in
            flight (the free accumulator registers). Returns the join's pointer, which replaces base."""
            e(f"  low.cond_br %clast, ^cv{t}, ^nc{t} : reg<aie2p.er>")
            e(f"^nc{t}:")
            e(f"  low.br ^cj{t}({base}: reg<aie2p.ep>)")
            e(f"^cv{t}:")
            # per sub-tile b the destination pointer is fixed at slot + 512 b + 256 (stores at 32 j - 256), so the 16 loads
            # (index 512 b + 64 j - 256) do not wait on the stores; it advances once per sub-tile
            cur_p, k, fwd, back = base, 0, back + 256 - 512 * (NP - 1), back + 256
            while back > 0:
                step = min(back, 448)
                back -= step
                e(f"  %cvb{t}_{k} = padda {cur_p}, {-step}")
                cur_p, k = f"%cvb{t}_{k}", k + 1
            for b_ in range(NP):
                for g0 in range(0, 16, group):   # loads in flight: the free accumulator registers
                    for j in range(g0, g0 + group):
                        e(f"  %cvo{t}_{b_}_{j} = mov.static-byte-offset {512 * b_ + 64 * j - 256}")
                        e(f"  %cvx{t}_{b_}_{j} = mov.address-index %cvo{t}_{b_}_{j}")
                        e(f"  %cvl{t}_{b_}_{j} = vlda.acc.index {cur_p}, %cvx{t}_{b_}_{j}")
                    for j in range(g0, g0 + group):
                        e(f"  vst.convert.f32x16.to.bf16x16 %cvl{t}_{b_}_{j}, {cur_p}, {32 * j - 256}")
                if b_ + 1 < NP:
                    bump(L, f"%cvp{t}_{b_}", cur_p, 512)
                    cur_p = f"%cvp{t}_{b_}"
            bump(L, f"%cvr{t}", cur_p, fwd)
            e(f"  low.br ^cj{t}(%cvr{t}: reg<aie2p.ep>)")
            e(f"^cj{t}(%cj{t}p: reg<aie2p.ep>):")
            e(f"  %cqr{t} = copy %clast : reg<aie2p.er> -> reg<aie2p.mr26_lock>")
            e(f"  rel.cond %one, %cqr{t}, 2")
            return f"%cj{t}p"
        if row_end and not nxt:
            pc[t] = convert(pc[t], 1024 * (NP - 1), 16)
        if nxt:
            e(f"  %pcc{t + 1} = copy {pc[t]} : reg<aie2p.ep> -> reg<aie2p.ep>")
            e(f"  %pc{t + 1} = padds.modifier %pcc{t + 1}, %mcs")
            for c in range(4):
                e(f"  %z{t + 1}_{c} = concat(%l{t + 1}_{c}_0, %l{t + 1}_{c}_1, %l{t + 1}_{c}_2, %l{t + 1}_{c}_3) : {MBMS4}")
                cur[c] = f"%z{t + 1}_{c}"
            if row_end:   # from the next store pointer, so this one is dead
                pc[t + 1] = convert(pc[t + 1], 1024 * NP, 4)   # the next sub-tile's C holds 16 of 20
    e("  rel %one, 0")
    e("  %pac = copy %pa : reg<aie2p.ep> -> reg<aie2p.ep>")
    e(f"  %pan0 = padds.modifier %pac, {'%mf' if a_adv else '%mz'}")
    steps = cfg.mu if a_adv else 1
    for j in range(1, steps):
        e(f"  %pan{j}c = copy %pan{j - 1} : reg<aie2p.ep> -> reg<aie2p.ep>")
        e(f"  %pan{j} = padds.modifier %pan{j}c, %mf")
    e(f"  %pan = copy %pan{steps - 1} : reg<aie2p.ep> -> reg<aie2p.ep>")
    if tail:
        e(f"  %pcl = copy {pc[-1]} : reg<aie2p.ep> -> reg<aie2p.ep>")
        e("  %pon = padds.modifier %pcl, %mcs")
        e(f"  %plx = copy {pl[-1]} : reg<aie2p.ep> -> reg<aie2p.ep>")
        e("  %pln = padds.modifier %plx, %mls")
    else:
        e("  %pon = copy %po : reg<aie2p.er> -> reg<aie2p.er>")
    e("  %mpn = add.rr %mp, %one")
    pln = ", %pln: reg<aie2p.ep>" if tail else ""
    e(f"  low.br ^outer(%mpn: reg<aie2p.er>, %pan: reg<aie2p.ep>, %pon: {pct}, {fifo['a']}: reg<aie2p.eldfiforeg>, {fifo['w']}: reg<aie2p.eldfiforeg>{pln})")
    e("^exit:")
    e("  return\n}\n")


def gen(cfg):
    assert len(cfg.ks) == 4 and cfg.passes >= 2 and 1 <= cfg.cols <= 8
    L = []
    array_program(L, cfg)
    leaf(L, cfg, "head", cfg.ks[0])
    assert len(set(cfg.ks[1:-1])) == 1, "the mids share one leaf"
    leaf(L, cfg, "mid", cfg.ks[1])
    leaf(L, cfg, "tail", cfg.ks[-1])
    return "\n".join(L)


def main():
    if len(sys.argv) < 5:
        sys.exit("usage: gen_npu_gemm.py <cols> <m_blocks> <ks,ks,ks,ks> <passes> [entry]")
    cfg = Config(int(sys.argv[1]), int(sys.argv[2]), tuple(int(v) for v in sys.argv[3].split(",")), int(sys.argv[4]),
                 entry=sys.argv[5] if len(sys.argv) > 5 else "npu_gemm")
    sys.stdout.write(gen(cfg))


if __name__ == "__main__":
    main()
