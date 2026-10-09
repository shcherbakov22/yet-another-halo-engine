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
LOOM_EXP_PANEL_STREAM: counting locks, a looping fill, invocations [setup | even push | odd push | waits | lead push]
and fills paced a few columns at a time so they do not starve the activation streams of DRAM; the lead push, a job's
first call, fills unpaced; LoomNpu submits the pushes one call ahead of the waits).
Needs HRX patch 0013; compile with LOOM_ENV. Config.gate > 0 adds the NPU-side gate (GATE below; HRX patch 0016).
"""
import contextlib
import dataclasses
import importlib
import math
import os
import re
import struct
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
    gate: int = 0       # NPU-side gate: poll supply per job (0: ungated); see GATE below
    dcol: int = 0       # 2: the fill column decodes fmt from a raw binding (gen_npu_dec); 1: a fill column (column cols, rows 2..5) streams each GEMM column's weight panel into its memory
                        # tile (constrain.fill; worker f fills columns 2 f, 2 f + 1); the panel binding is not read
    frec: int = 2304    # fill record bytes
    fmt: str = "IQ4_XS"  # dcol 2: the raw weights' GGUF format
    kraw: int = 0       # dcol 2: the raw rows' K (row stride; the call reads K = 1024 * passes of it), 0: the call's K
    swiglu: bool = False  # one replay group: a column's first NP / 2 slabs are gate rows, the rest the up rows of the
                          # same features; the tail writes silu(gate) * up as bfp16ebs8 fragments (convert_swiglu) into
                          # H (hpass), the next GEMM's activations (ain = "h")
    ain: str = ""         # "h": the activations are read from H (a swiglu call's output) instead of the encoder's stream
    dgu: int = 0          # swiglu dcol 2: bytes from a gate row to its up row (the raw stream per pass: [gate / up][32 rows])
    hpass: int = 0        # H as [M block][pass hpass][16-token slab][128 k-blocks] (the FFN block's swiglu output, down's
                          # input): swiglu (cols = hpass): column c writes pass c; ain "h": records [M block][pass, slab]
    swcol: int = 0        # swiglu dcol 2: raw rows between columns' first gate rows (0: TN / 2)
    gord: str = "P4"      # dcol 2: the grid decoders' k-block order (gen_npu_dec DQ_GORD; "N": natural)
    hord: str = ""        # swiglu: H in the down decoder's k-block order: "P4" ((0, 2, 1, 3) per 4 k-blocks: IQ4_XS's
                          # and the P4 grid decoders'), "Q4K" ((0, 4, 1, 5, 2, 6, 3, 7) per 8: Q4_K's; a call's 4 gate
                          # k-blocks (one format) are then slabs 32 rows apart, the caller steps call j's rows by
                          # 64 (j / 2) + 16 (j % 2)), "": natural
    ufmt: str = ""        # swiglu dcol 2 column pairs (cols 6): columns 0-2 compute gate rows (fmt), 3-5 the same features'
                          # up rows (ufmt); gate tail u streams its C segment (bf16) to up tail u + 3 (core stream)
    fuse: int = 0       # 1: every GEMM core fills its own weight panel slice at each replay group's first firing
                        # (pass-through: its column's raw stream multicast to its core stream, its slice through a
                        # leaf-synced egress ring into its memory-tile panel, constrain.fill; HRX patch 0020); W is
                        # leaf-synchronized. 2: the raw stream is IQ4_XS (fuse_raw_dec): every filling core decodes
                        # the super-blocks covering its slice (gen_npu_dec.leaf_inline, in its idle W ring)
    ofeat: int = 0      # final f32 output, fragment-major into [token / 16][ofeat / 16][16][16]: column c's 80 features are
                        # fragments 5c .. 5c + 4; needs one replay group (GE.GROUPS = (passes,)) and
                        # LOOM_EXP_LS_SEND_PITCH=1 (the whole f32 slot is sent)


# NPU-side gate (cfg.gate = supply): column 0's head waits for each job's ready word itself and its neighbor mid relays.
# Extra bindings: 3 flag (read; 16-byte record, word 0 ready = the job's gate value), 4 tick (write; scratch), 5 signal
# (write; go at byte 0, done at byte 16). After its last firing of a job the head polls request-driven
# (constrain.request: one fresh flag read per tick) until ready >= (its job count + 1) * GATE_CALLS (jobs count from 1 after the
# setup call; jobs sharing a ready word store theirs in order, after the GPU joined the earlier ones), backing off (pace =
# GP0 + GPS * max(0, polls - 64) delay iterations; a supply of 1024 polls lasts ~400 ms: decoder-column sets leave layers
# without NPU work; the tick stream takes one record per poll, so the supply cannot grow). It hands [seq, status, polls,
# ncalls] to the mid over a leaf-synchronized neighbor channel and ticks / reads out the rest of the supply GB records per
# firing of the next job. The mid emits go and done (constrain.signal + constrain.gate): the control program's gate
# invocation waits for go before any data moves; its done invocation queues done after the job's egress, so done lands
# after C. done = the job's ready value, with GATE_FAILED set if the head gave up (status 2). Job state (firings done, firings per job, leftover
# supply, pending reads, sequence) lives in private storage, which the array setup of a core stream plan zeroes; a job of
# N calls is N * nb * passes firings, and before the first gated job (state zero) it is one call: the setup call's.
GP0, GPS, GB = 650, 104, 8
# The protocol the host follows (dispatch.txt "npugate <GATE_SUPPLY> <GATE_CALLS> <GATE_RECORD>"): a job's gate value is
# sequence * GATE_CALLS + its calls (jobs of 1 .. GATE_CALLS - 1 calls); done lands GATE_RECORD bytes into the signal
# binding. GATE_FAILED in done: the head gave up (the host sets it too for a failed command; gen_npu_unpack.GATED_LOOP).
GATE_SUPPLY, GATE_CALLS, GATE_RECORD, GATE_FAILED = 1024, 64, 16, 1 << 31


_LEAVES = {}


def decoder(fmt, gord="P4"):
    """gen_npu_dec for fmt (its constants follow DQ_FMT and DQ_GORD at import)."""
    import gen_npu_dec
    if gen_npu_dec.FMT != fmt or os.environ.get("DQ_GORD", "P4") != gord:
        os.environ["DQ_FMT"] = fmt
        os.environ["DQ_GORD"] = gord
        gen_npu_dec = importlib.reload(gen_npu_dec)
    return gen_npu_dec


def loom_env(cfg, swap=False):
    """LOOM_ENV for cfg. The decoder column (dcol 2): records interleaved per panel slab, the send pitch on the C ring
    only (the decoders' fill rings keep pitch 1). swap: the image's setup alone, only the decoder tiles' records (an
    image switch between decoder formats of otherwise identical images)."""
    env = dict(LOOM_ENV)
    if cfg.dcol == 2:
        env["LOOM_EXP_PANEL_INTERLEAVE"] = str(NP)
    if cfg.dcol == 2 or cfg.fuse:   # the fill egress rings are contiguous (fuse_prologue steps them by a record)
        env["LOOM_EXP_LS_SEND_PITCH_RECORD"] = str(NP * 4 * 256 // 8 * 4)
    if cfg.fuse:   # the shared last-two-rows panel interleaved per unit (its filler emits both rows' slabs of a unit
        env["LOOM_EXP_PANEL_INTERLEAVE"] = str(NP)   # together; uniform slots of KS (36, 36, 36, 20) would not fit)
    if not groups(cfg):
        env["LOOM_EXP_PANEL_GROUPS"] = str(cfg.passes)
        if cfg.dcol == 2:   # one group, still streamed from the decoder column (HRX patch 0019)
            env["LOOM_EXP_PANEL_STREAM_SINGLE"] = "1"
    if cfg.swiglu:
        env["LOOM_EXP_LS_SEND_PITCH"] = str(SW_PITCH)
        env["LOOM_EXP_LS_SEND_PITCH_RECORD"] = str(SW_REC)
        env["LOOM_EXP_PANEL_GROUPS"] = str(cfg.passes)
    if cfg.ufmt:   # 6 GEMM columns + the fill column: admitted on the engine's 8-column context
        env["LOOM_EXP_PARTITION_COLUMNS"] = "8"
    if cfg.dcol == 2 and not swap:   # images of other layouts share the context: their routes must not outlive them
        env["LOOM_EXP_ROUTE_RESET"] = "1"
    if swap:
        env["LOOM_EXP_SETUP_ONLY"] = "1"
        env["LOOM_EXP_SETUP_TILES"] = ",".join(f"{fill_col(cfg)}:{2 + f}" for f in range(fill_workers(cfg)))
    return env


def fill_col(cfg):
    """The fill workers' column: column cols, column pairs column 7 like every other image of the set (a context that
    ran the pair image left its column 6 tiles' state unusable for a later image's GEMM column there)."""
    return 7 if cfg.ufmt else cfg.cols


def groups(cfg):
    """The call's replay groups of passes, () for a single group."""
    if cfg.swiglu or cfg.ain:
        return ()
    g = GE.pass_groups(cfg.passes)
    assert len(g) <= 2
    return g if len(g) > 1 else ()


def bf16_round(x):
    """x rounded to bf16 (nearest even), as a float."""
    u = struct.unpack("<I", struct.pack("<f", x))[0]
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", u))[0]


def bf16_bits(x):
    return struct.unpack("<I", struct.pack("<f", bf16_round(x)))[0] >> 16


def f32_bits(x):
    v = struct.unpack("<i", struct.pack("<f", x))[0]
    return v


SW_REC = NP * 2 * 72      # swiglu: a segment's output record, NP k-blocks x 2 token halves of bfp16 fragments (720 B)
SW_PITCH = 8              # its send pitch: a C slot of 8 records (5760 B) holds the segment's f32 C (5120 B)
SW_SLOT = SW_PITCH * SW_REC


@contextlib.contextmanager
def np_override(n, pair=False):
    """Generate with n 16-row slabs per column (TN = 16 n) instead of NP (e.g. 4: a swiglu call's 32 features per
    column, so its k-block units line up with 1024-feature passes). pair: column pairs (Config.ufmt), whose up columns
    write all n slabs' features: twice the record, so a slot of 4 records holds the f32 segment."""
    global NP, TN, SW_REC, SW_PITCH, SW_SLOT
    saved = NP, TN, SW_REC, SW_PITCH, SW_SLOT
    NP, TN = n, 16 * n
    SW_REC = NP * 2 * 72 * (2 if pair else 1)
    SW_PITCH = 4 if pair else 8
    SW_SLOT = SW_PITCH * SW_REC
    assert SW_SLOT >= 1024 * NP
    try:
        yield
    finally:
        NP, TN, SW_REC, SW_PITCH, SW_SLOT = saved


def slab(ks):
    return (144 * ks + 63) // 64 * 64


def stream_bytes(cfg):
    """(activation, weight, C) binding sizes."""
    a = sum(cfg.nb * cfg.passes * MP * slab(k) for k in cfg.ks)
    if cfg.ain == "h":   # the view's extent in H ([M block][pass][slab][128 k-blocks])
        a = (144 * sum(cfg.ks[:-1]) + (cfg.nb - 1) * cfg.hpass * MP * 128 * 144 + (cfg.passes * MP - 1) * 128 * 144
             + slab(cfg.ks[-1]))
    w = cfg.cols * sum(cfg.passes * NP * slab(k) for k in cfg.ks)
    if cfg.swiglu:   # the receiver's extent in H: the last 16-token row's columns
        # the last column's (pass's) last record in H (column pairs: up column u writes passes 2 u, 2 u + 1)
        last = 2 * (cfg.cols // 2 - 1) if cfg.ufmt else cfg.cols - 1
        return a, w, ((cfg.nb - 1) * cfg.hpass * MP * 128 * 144 + (MP - 1) * 128 * 144
                      + last * MP * 128 * 144 + SW_REC)
    return a, w, max(1, len(groups(cfg))) * cfg.cols * cfg.nb * MP * NP * 2 * 256


def shuffle16_cycles():
    """The non-trivial cycles of piece i -> 2 (i % 8) + i // 8 on 16 pieces."""
    seen, out = set(), []
    for i in range(16):
        c, j = [], i
        while j not in seen:
            seen.add(j)
            c.append(j)
            j = 2 * (j % 8) + j // 8
        if len(c) > 1:
            out.append(c)
    return out


SHUFFLE16_CYCLES = shuffle16_cycles()


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
    for k, v in (("zero", 0), ("one", 1), ("two", 2), ("nw", rows * cols + (fill_workers(cfg) if cfg.dcol else 0)), ("rec", nb * P), ("orec", nb), ("fold", P)):
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
            grole = role + "_g" if cfg.gate and c == 0 and r < 2 else role
            if cfg.fuse and r == rows - 2 and r > 1:
                grole = "mid_l"   # the last mid also fills the tail's slice
            if cfg.ufmt and role == "tail" and c < cols // 2:
                grole = "tail_s"   # a gate column's tail: it streams its C to the up column's tail
            e(f"  %k{c}_{r} = worker %workers, %lane{c}_{r}, @{grole}")
            e(f"  constrain.location %k{c}_{r}, %n{c}, %n{2 + rows - 1 - r}")
    # C first: rings are placed in channel order, so the tail's C claims whole banks before its input rings
    e(f"  %ncols = constant.u32 {cols} : reg<aie2p.array.scalar : index>")
    seg = MP * NP * 4 * 256 // MP
    ng = max(1, len(groups(cfg)))   # C segments per M block and group
    e(f"  %nseg = constant.u32 {ng * nb * MP} : reg<aie2p.array.scalar : index>")
    if cfg.swiglu:
        # column c's record (NP k-blocks of 16 tokens) at k-block NP c of its 16-token row in H
        rec = f"{SW_REC // 4}"   # column c: pass c of H [M block][pass][slab][128 k-blocks]
        if cfg.ufmt:   # up column u: passes 2 u, 2 u + 1 (j // 16 of the call); gate columns: a junk binding
            h = cols // 2
            e(f"  %c_all = receiver %cb, 0 : reg<aie2p.array.receiver : tile<{h}x{nb}x{MP}x{rec}xi32, "
              f"#encoding.layout.strided<strides=[{2 * MP * 128 * 36}, {cfg.hpass * MP * 128 * 36}, {128 * 36}, 1]>>>")
            e(f"  %jb = binding {9 if cfg.gate else 6}, \"write\"")
            e(f"  %j_all = receiver %jb, 0 : reg<aie2p.array.receiver : tile<{h}x{nb * MP}x{rec}xi32>>")
            e(f"  %nhalf = constant.u32 {h} : reg<aie2p.array.scalar : index>")
        else:
            assert cols == cfg.hpass
            e(f"  %c_all = receiver %cb, 0 : reg<aie2p.array.receiver : tile<{cols}x{nb}x{MP}x{rec}xi32, "
              f"#encoding.layout.strided<strides=[{MP * 128 * 36}, {cfg.hpass * MP * 128 * 36}, {128 * 36}, 1]>>>")
        stype = f"tile<{rec}xi32>"
    elif cfg.ofeat:
        # a segment is [sub-tile 5][16 tokens][16 f32] (the tail's convert_f32)
        # the tail's slot is [sub-tile][16 tokens][16 f32] (convert_f32): NP consecutive fragments of one 16-token row
        # (the 1280 contiguous words as 5 x 256: a shim DMA wrap holds at most 1023)
        rec = f"{NP}x256"
        # with two replay groups each group's partial is its own tensor, stacked: [group][token / 16][ofeat / 16][16][16]
        e(f"  %c_all = receiver %cb, 0 : reg<aie2p.array.receiver : tile<{cols}x{ng * nb * MP}x{rec}xi32, "
          f"#encoding.layout.strided<strides=[{NP * 256}, {cfg.ofeat * 16}, 256, 1]>>>")
        stype = f"tile<{rec}xi32>"
    else:
        rec = f"{seg // 8}"
        e(f"  %c_all = receiver %cb, 0 : reg<aie2p.array.receiver : tile<{cols}x{ng * nb * MP}x{rec}xi32>>")
        stype = f"tile<{rec}xi32>"
    for c in range(cols):
        if cfg.ufmt:
            u = c % (cols // 2)
            src = "%c_all" if c >= cols // 2 else "%j_all"
            e(f"  %cr{c} = partition.receiver {src}, %origin, %n{u}, %nhalf : reg<aie2p.array.receiver : tile<{rec}xi32>>")
        else:
            e(f"  %cr{c} = partition.receiver %c_all, %origin, %n{c}, %ncols : reg<aie2p.array.receiver : tile<{rec}xi32>>")
        e(f"  %sc{c} = sender %k{c}_{rows - 1}, 2 : reg<aie2p.array.sender : {stype}>")
        e(f"  %chc{c} = channel %sc{c}, %cr{c}, %n{MP}, %nseg : reg<aie2p.array.channel : tile<{rec}xi32>>")
        e(f"  constrain.leaf_sync %chc{c}")
    if cfg.ufmt:   # gate tail u -> up tail u + 3: its C segment as bf16 (core streams, 512 words a segment)
        e(f"  %nxs = constant.u32 {nb * MP * 512} : reg<aie2p.array.scalar : index>")
        for u in range(cols // 2):
            e(f"  %xs{u} = sender %k{u}_{rows - 1}, 3 : reg<aie2p.array.sender : tile<1xi32>>")
            e(f"  %xr{u} = receiver %k{u + cols // 2}_{rows - 1}, 3 : reg<aie2p.array.receiver : tile<1xi32>>")
            e(f"  %chx{u} = channel %xs{u}, %xr{u}, %two, %nxs : reg<aie2p.array.channel : tile<1xi32>>")
            e(f"  constrain.core_stream %chx{u}")
    if cfg.dcol:
        fill_channels(e, cfg, panel)
    if cfg.fuse:
        fuse_channels(e, cfg, panel)
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
            if cfg.fuse:   # rows 0 .. rows - 2 fill their own panels; the tail's shares row rows - 2's fill
                if r < rows - 1:
                    e(f"  constrain.leaf_sync %chw{c}_{r}")
                e(f"  constrain.fill %chw{c}_{r}, %chd{c}_{min(r, rows - 2)}")
        if cfg.dcol:
            e(f"  constrain.fill %chw{c}_0, %chf{c}")
    if cfg.dcol:
        fill_inputs(e, cfg, panel)
    # activations: one stream per row, one slab per record, multicast along the row (first branch rotates)
    for r in range(rows):
        fa = slab(cfg.ks[r])
        a_off = sum(nb * P * MP * slab(k) for k in cfg.ks[:r])
        arec = f"{fa // 4}xi32"
        if cfg.ain == "h":   # H [M block][pass][slab][128 k-blocks]: records [M block][pass, slab]
            a_off = 144 * sum(cfg.ks[:r])
            e(f"  %ao{r} = constant.u64 {a_off} : reg<aie2p.array.offset : offset>")
            rw = fa // 4
            sp = next(k for k in range(1, 9) if rw % k == 0 and rw // k <= 1023)
            arec = f"{sp}x{rw // sp}xi32"
            e(f"  %as{r} = sender %ab, 0 : reg<aie2p.array.sender : tile<{nb}x{P * MP}x{sp}x{rw // sp}xi32, "
              f"#encoding.layout.strided<strides=[{cfg.hpass * MP * 128 * 36}, {128 * 36}, {rw // sp}, 1]>>>")
        else:
            e(f"  %ao{r} = constant.u64 {a_off} : reg<aie2p.array.offset : offset>")
            e(f"  %as{r} = sender %ab, 0 : reg<aie2p.array.sender : tile<{nb * P * MP}x{fa // 4}xi32>>")
        e(f"  %a{r} = view.sender %as{r}, %ao{r} : reg<aie2p.array.sender : tile<{arec}>>")
        e(f"  %arec{r} = constant.u32 {nb * P * MP} : reg<aie2p.array.scalar : index>")
        for j in range(cols):
            c = (2 * r + j) % cols
            acap = cfg.acap_t if roles[r] == "tail" else cfg.acap
            e(f"  %ra{c}_{r} = receiver %k{c}_{r}, 0 : reg<aie2p.array.receiver : tile<{arec}>>")
            e(f"  %cha{c}_{r} = channel %a{r}, %ra{c}_{r}, %n{acap}, %arec{r} : reg<aie2p.array.channel : tile<{arec}>>")
            e(f"  constrain.leaf_sync %cha{c}_{r}")
            e(f"  constrain.stage %cha{c}_{r}, %n{astage(r, cols) if cols > 1 else 2}")
    if cfg.gate:
        S = cfg.gate
        e('  %fb = binding 3, "read"')
        e('  %tb = binding 4, "write"')
        e('  %sb = binding 5, "write"')
        e(f"  %gsup = constant.u32 {S} : reg<aie2p.array.scalar : index>")
        e(f"  %gf_all = sender %fb, 0 : reg<aie2p.array.sender : tile<1x{S}x1x4xi32, #encoding.layout.strided<strides=[4, 0, 0, 1]>>>")
        e("  %gf = partition.sender %gf_all, %origin, %n0, %one : reg<aie2p.array.sender : tile<4xi32>>")
        e("  %gpoll = receiver %k0_0, 2 : reg<aie2p.array.receiver : tile<4xi32>>")
        e("  %chgp = channel %gf, %gpoll, %two, %gsup : reg<aie2p.array.channel : tile<4xi32>>")
        e("  constrain.core_stream %chgp")
        e(f"  %gt_all = receiver %tb, 0 : reg<aie2p.array.receiver : tile<1x{S}x1x4xi32, #encoding.layout.strided<strides=[4, 0, 0, 1]>>>")
        e("  %gt = partition.receiver %gt_all, %origin, %n0, %one : reg<aie2p.array.receiver : tile<4xi32>>")
        e("  %gtick = sender %k0_0, 3 : reg<aie2p.array.sender : tile<4xi32>>")
        e("  %chgt = channel %gtick, %gt, %two, %gsup : reg<aie2p.array.channel : tile<4xi32>>")
        e("  constrain.core_stream %chgt")
        e("  constrain.request %chgp, %chgt")
        e("  %gn1s = sender %k0_0, 4 : reg<aie2p.array.sender : tile<4xi32>>")
        e("  %gn1r = receiver %k0_1, 2 : reg<aie2p.array.receiver : tile<4xi32>>")
        e("  %chgn = channel %gn1s, %gn1r, %one, %one : reg<aie2p.array.channel : tile<4xi32>>")
        e("  constrain.leaf_sync %chgn")
        e("  %gs_all = receiver %sb, 0 : reg<aie2p.array.receiver : tile<1x2x4xi32>>")
        e("  %gs = partition.receiver %gs_all, %origin, %n0, %one : reg<aie2p.array.receiver : tile<4xi32>>")
        e("  %gsig = sender %k0_1, 3 : reg<aie2p.array.sender : tile<4xi32>>")
        e("  %chgs = channel %gsig, %gs, %two, %two : reg<aie2p.array.channel : tile<4xi32>>")
        e("  constrain.core_stream %chgs")
        e("  constrain.gate %chgs")
        e("  constrain.signal %chgs")
    e("  return\n}\n")


FILL_PER = 2   # GEMM columns per fill worker
FILL_DIV = 1   # timing emulation: input records frec / FILL_DIV, written FILL_DIV times (raw-sized DRAM reads)


def astage(r, cols):
    """The memory-tile column K-slice row r's activations stage in: one row per column (0, 2, 4, 6; 6 columns: 0, 2, 4, 1)."""
    return 2 * r if 2 * r < cols else (2 * r + 1) % cols


def fill_map(cfg):
    """Per GEMM column its (fill worker, port, decoder format); per fill worker its (ports, format). Column pairs give
    the gate columns and the up columns their own workers (one decoder format each)."""
    if cfg.ufmt:
        assert cfg.cols == 6
        cols = [(0, 0, cfg.fmt), (0, 1, cfg.fmt), (1, 0, cfg.fmt), (2, 0, cfg.ufmt), (2, 1, cfg.ufmt), (3, 0, cfg.ufmt)]
    else:
        cols = [(c // FILL_PER, c % FILL_PER, cfg.fmt) for c in range(cfg.cols)]
    workers = {}
    for f, j, fmt in cols:
        workers[f] = (max(workers.get(f, (0, fmt))[0], j + 1), fmt)
    return cols, [workers[f] for f in sorted(workers)]


def fill_name(cfg, n, fmt):
    return f"fill{n}_{fmt}" if cfg.ufmt else f"fill{n}"


def fill_workers(cfg):
    return len(fill_map(cfg)[1])


def fill_channels(e, cfg, panel):
    """Fill workers in column cols (rows 2..): worker f takes columns 2 f, 2 f + 1, each a contiguous panel of %wb in
    frec-byte records (in port j, out port 2 + j), into the dummy write binding %fsb (lane c)."""
    cols, rows = cfg.cols, len(cfg.ks)
    frec = 2304 if cfg.dcol == 2 else cfg.frec
    assert cols <= 7 and panel % frec == 0
    nrec = panel // frec
    fw = frec // 4
    e(f"  %fcol = constant.u32 {cols} : reg<aie2p.array.scalar : index>")
    e(f"  %floc = constant.u32 {fill_col(cfg)} : reg<aie2p.array.scalar : index>")
    e(f"  %fnrec = constant.u32 {nrec} : reg<aie2p.array.scalar : index>")
    e(f"  %fsb = binding {6 if cfg.gate else 3}, \"write\"")
    e(f"  %fs_all = receiver %fsb, 0 : reg<aie2p.array.receiver : tile<{cols}x{nrec}x{fw}xi32>>")
    cmap, wmap = fill_map(cfg)
    for f, (n, fmt) in enumerate(wmap):
        e(f"  %flane{f} = constant.u32 {rows * cols + f} : reg<aie2p.array.scalar : index>")
        e(f"  %frow{f} = constant.u32 {2 + f} : reg<aie2p.array.scalar : index>")
        e(f"  %kf{f} = worker %workers, %flane{f}, @{fill_name(cfg, n, fmt)}")
        e(f"  constrain.location %kf{f}, %floc, %frow{f}")
    for c in range(cols):
        f, j, _ = cmap[c]
        nports = wmap[f][0]
        e(f"  %fsr{c} = partition.receiver %fs_all, %origin, %n{c}, %fcol : reg<aie2p.array.receiver : tile<{fw}xi32>>")
        e(f"  %fss{c} = sender %kf{f}, {nports + j} : reg<aie2p.array.sender : tile<{fw}xi32>>")
        e(f"  %chf{c} = channel %fss{c}, %fsr{c}, %two, %fnrec : reg<aie2p.array.channel : tile<{fw}xi32>>")
        if cfg.dcol == 2:
            e(f"  constrain.leaf_sync %chf{c}")


def fill_inputs(e, cfg, panel):
    """The fill workers' input channels (after the panels: each stages in its column's memory tile, behind the panel)."""
    nrec, fw = panel // cfg.frec, cfg.frec // 4 // FILL_DIV
    if cfg.dcol == 2:
        # raw rows [N][kraw / 256][BLK]: per column 25 (pass, slab) units of [16 rows][4 super-blocks]
        P_, NS_ = cfg.passes, NP
        assert sum(cfg.ks) == 128
        e(f"  %rwb = binding {7 if cfg.gate else 4}, \"read\"")
        if cfg.ufmt:   # the up columns' rows: their own binding (another format, another row size)
            e(f"  %rub = binding {8 if cfg.gate else 5}, \"read\"")
        e(f"  %funits = constant.u32 {P_ * NS_} : reg<aie2p.array.scalar : index>")
        cmap = fill_map(cfg)[0]
        for c in range(cfg.cols):
            f, j, fmt = cmap[c]
            D = decoder(fmt, cfg.gord)
            RS = (cfg.kraw or 8 * cfg.passes * sum(cfg.ks)) // 256 * D.BLK // 4   # row stride in words
            RW = D.ROWB // 4                                  # record row words (fixed for every format)
            if cfg.ufmt:   # gate column u / up column u: rows swcol u .. of its own binding
                up = c >= cfg.cols // 2
                e(f"  %fo{c} = constant.u64 {(c % (cfg.cols // 2)) * cfg.swcol * RS * 4} : reg<aie2p.array.offset : offset>")
                e(f"  %fi_all{c} = sender {'%rub' if up else '%rwb'}, 0 : reg<aie2p.array.sender : tile<{P_}x{NS_}x16x{RW}xi32, "
                  f"#encoding.layout.strided<strides=[{D.BLK}, {16 * RS}, {RS}, 1]>>>")
            elif cfg.swiglu:   # column c: gate rows swcol c .. (its features), then the same up rows dgu bytes on
                assert cfg.dgu % 4 == 0 and NS_ % 2 == 0
                e(f"  %fo{c} = constant.u64 {c * (cfg.swcol or TN // 2) * RS * 4} : reg<aie2p.array.offset : offset>")
                e(f"  %fi_all{c} = sender %rwb, 0 : reg<aie2p.array.sender : tile<{P_}x2x{NS_ // 2}x16x{RW}xi32, "
                  f"#encoding.layout.strided<strides=[{D.BLK}, {cfg.dgu // 4}, {(32 if cfg.hord == 'Q4K' else 16) * RS}, "
                  f"{RS}, 1]>>>")
            else:
                e(f"  %fo{c} = constant.u64 {c * TN * RS * 4} : reg<aie2p.array.offset : offset>")
                e(f"  %fi_all{c} = sender %rwb, 0 : reg<aie2p.array.sender : tile<{P_}x{NS_}x16x{RW}xi32, "
                  f"#encoding.layout.strided<strides=[{D.BLK}, {16 * RS}, {RS}, 1]>>>")
            e(f"  %fiv{c} = view.sender %fi_all{c}, %fo{c} : reg<aie2p.array.sender : tile<16x{RW}xi32>>")
            e(f"  %fir{c} = receiver %kf{f}, {j} : reg<aie2p.array.receiver : tile<16x{RW}xi32>>")
            e(f"  %chfi{c} = channel %fiv{c}, %fir{c}, %one, %funits : reg<aie2p.array.channel : tile<16x{RW}xi32>>")
            a_cols = {astage(r, cfg.cols) for r in range(len(cfg.ks))}
            e(f"  constrain.stage %chfi{c}, {'%fcol' if c in a_cols else f'%n{c}'}")
        return
    for c in range(cfg.cols):
        f, j, _ = fill_map(cfg)[0][c]
        e(f"  %fo{c} = constant.u64 {c * panel} : reg<aie2p.array.offset : offset>")
        e(f"  %fi_all{c} = sender %wb, 0 : reg<aie2p.array.sender : tile<{nrec}x{fw}xi32>>")
        e(f"  %fiv{c} = view.sender %fi_all{c}, %fo{c} : reg<aie2p.array.sender : tile<{fw}xi32>>")
        e(f"  %fir{c} = receiver %kf{f}, {j} : reg<aie2p.array.receiver : tile<{fw}xi32>>")
        e(f"  %chfi{c} = channel %fiv{c}, %fir{c}, %two, %fnrec : reg<aie2p.array.channel : tile<{fw}xi32>>")
        # the activation rows stage in memory tiles (2 r) % cols: those columns' fills stage in the fill column's
        a_cols = {astage(r, cfg.cols) for r in range(len(cfg.ks))}
        e(f"  constrain.stage %chfi{c}, {'%fcol' if c in a_cols else f'%n{c}'}")


FUSE_REC = 200   # fused fill egress record words, at most (each replay group's fill must end on a whole record)


def fuse_ports(role):
    """(egress port, core stream port) of a filling worker: after its A and W ports."""
    return 2, 3


def fuse_groups(cfg):
    """The panels' replay groups as pass ranges."""
    g = groups(cfg) or (cfg.passes,)
    out, p = [], 0
    for n in g:
        out.append(range(p, p + n))
        p += n
    return out


def fuse_record(cfg, r):
    """(egress record words, records per call) of filling row r (rows 0 .. rows - 2; the second-to-last row also fills
    the last row's slice: the tail's panel shares its fill). Records are whole 4-word steps dividing every replay
    group's fill."""
    rows = len(cfg.ks)
    unit = NP * sum(slab(k) for k in (cfg.ks[r:] if r == rows - 2 else cfg.ks[r:r + 1])) // 4
    if cfg.fuse == 2:   # whole 64-byte copies, whole records per slab
        fw = max(w for w in range(16, FUSE_REC + 1, 16) if all(sw % w == 0 for _, sw in fuse_pieces(cfg, r)))
        return fw, cfg.passes * unit // fw
    g = math.gcd(*(len(p) * unit for p in fuse_groups(cfg)))
    fw = max(w for w in range(4, FUSE_REC + 1, 4) if g % w == 0)
    return fw, cfg.passes * unit // fw


FUSE_SB, FUSE_BLK = 4, 136       # fuse 2: IQ4_XS super-blocks per row of a unit (a pass: 1024 k), bytes per super-block
FUSE_UNIT_RAW = 16 * FUSE_SB * FUSE_BLK   # a (pass, slab) unit's raw bytes: [16 rows][4 super-blocks]


def fuse_span(cfg, r):
    """fuse 2, filling row r: (k-block range k0, k1 of a pass; first super-block, super-block count)."""
    rows = len(cfg.ks)
    k0 = sum(cfg.ks[:r])
    k1 = sum(cfg.ks) if r == rows - 2 else k0 + cfg.ks[r]
    s0, s1 = k0 // 32, -(-k1 // 32)
    return k0, k1, s0, s1 - s0


def fuse_pieces(cfg, r):
    """fuse 2, filling row r: per slice of its panel unit, (first k-block, padded slab words)."""
    rows = len(cfg.ks)
    out = [(sum(cfg.ks[:j]), slab(cfg.ks[j]) // 4) for j in ((r, rows - 1) if r == rows - 2 else (r,))]
    assert all(sw % 16 == 0 for _, sw in out), "fuse 2: slabs of whole 64-byte copies"
    return out


def fuse_raw_dec(raw, cfg):
    """fuse 2: the raw binding from IQ4_XS blocks raw [cols 80 rows][K / 256 super-blocks][136] (u8): per column, per
    (pass, slab) unit, [16 rows][4 super-blocks of the pass]."""
    r = raw.reshape(cfg.cols, NP, 16, cfg.passes, FUSE_SB, FUSE_BLK)
    return r.transpose(0, 3, 1, 2, 4, 5).reshape(-1)


def fuse_raw(w, cfg):
    """The fused raw binding from the weight binding (int32 words, [column][group][slice][pass in group][NP slabs]):
    per column, per replay group, per unit (pass in group, slab): the rows' slabs ([unit][row][slab])."""
    import numpy as np
    P = cfg.passes
    panel = sum(P * NP * slab(k) for k in cfg.ks) // 4
    out = []
    for c in range(cfg.cols):
        off = c * panel
        for g in fuse_groups(cfg):
            sl = []
            for k in cfg.ks:
                n = len(g) * NP * slab(k) // 4
                sl.append(w[off:off + n].reshape(len(g) * NP, slab(k) // 4))
                off += n
            for u in range(len(g) * NP):
                out.extend(s_[u] for s_ in sl)
    return np.concatenate(out)


def fuse_channels(e, cfg, panel):
    """Per filling worker (rows 0 .. rows - 2): its leaf-synced egress ring (the source of its panel's fill; the dummy
    write binding, a lane per worker, is never transferred), and its column's raw stream, multicast over the column's
    filling core streams."""
    rows, cols = len(cfg.ks), cfg.cols
    uw = FUSE_UNIT_RAW // 4 if cfg.fuse == 2 else sum(slab(k) for k in cfg.ks) // 4
    units = cfg.passes * NP
    e(f"  %fdb = binding {6 if cfg.gate else 3}, \"write\"")
    e(f"  %frb = binding {7 if cfg.gate else 4}, \"read\"")
    e(f"  %fdl = constant.u32 {rows * cols} : reg<aie2p.array.scalar : index>")
    e(f"  %fun = constant.u32 {units} : reg<aie2p.array.scalar : index>")
    e(f"  %fru_all = sender %frb, 0 : reg<aie2p.array.sender : tile<{cols}x{units}x{uw}xi32>>")
    for r in range(rows - 1):
        fw, nrec = fuse_record(cfg, r)
        e(f"  %fdn{r} = constant.u32 {nrec} : reg<aie2p.array.scalar : index>")
        e(f"  %fd_all{r} = receiver %fdb, 0 : reg<aie2p.array.receiver : tile<{rows * cols}x{nrec}x{fw}xi32>>")
    for c in range(cols):
        e(f"  %fru{c} = partition.sender %fru_all, %origin, %n{c}, %ncols : reg<aie2p.array.sender : tile<{uw}xi32>>")
        for r in range(rows - 1):
            dp, sp = fuse_ports("")
            fw, _ = fuse_record(cfg, r)
            e(f"  %fdr{c}_{r} = partition.receiver %fd_all{r}, %origin, %lane{c}_{r}, %fdl : reg<aie2p.array.receiver : tile<{fw}xi32>>")
            e(f"  %fds{c}_{r} = sender %k{c}_{r}, {dp} : reg<aie2p.array.sender : tile<{fw}xi32>>")
            e(f"  %chd{c}_{r} = channel %fds{c}_{r}, %fdr{c}_{r}, %two, %fdn{r} : reg<aie2p.array.channel : tile<{fw}xi32>>")
            e(f"  constrain.leaf_sync %chd{c}_{r}")
            e(f"  %frr{c}_{r} = receiver %k{c}_{r}, {sp} : reg<aie2p.array.receiver : tile<{uw}xi32>>")
            e(f"  %chr{c}_{r} = channel %fru{c}, %frr{c}_{r}, %two, %fun : reg<aie2p.array.channel : tile<{uw}xi32>>")
            e(f"  constrain.core_stream %chr{c}_{r}")


class _Fill:
    """Emits a filling worker's stream walk: skips and takes of 4-word steps; a take appends to the egress ring
    (fuse_record words a record, slot = record % 2) with its state (words in the record, record) in private storage."""

    def __init__(self, e, dp):
        self.e, self.dp, self.n = e, dp, 0

    def tag(self):
        self.n += 1
        return f"{self.n}"

    def skip(self, n):
        """n words."""
        assert n % 4 == 0
        if n == 0:
            return
        e, t = self.e, self.tag()
        e(f"  %fzk{t} = mov.i32 {n // 4}")
        e(f"  low.br ^fzs{t}(%zero: reg<aie2p.er>)")
        e(f"^fzs{t}(%fzi{t}: reg<aie2p.er>):")
        e(f"  %fzim{t} = lt %fzi{t}, %fzk{t}")
        e(f"  low.cond_br %fzim{t}, ^fzsb{t}, ^fzsx{t} : reg<aie2p.er>")
        e(f"^fzsb{t}:")
        for w in range(4):
            e(f"  %fzx{t}_{w} = mov.ss")
            e(f"  st %fzx{t}_{w}, %fzjp, {4 * w}")   # into a junk block (the pops stay)
        e(f"  %fzi1{t} = add.rr %fzi{t}, %one")
        e(f"  low.br ^fzs{t}(%fzi1{t}: reg<aie2p.er>)")
        e(f"^fzsx{t}:")

    def slot(self, dn, t):
        """The egress slot address of record dn plus dq words."""
        e = self.e
        e(f"  %fzSh{t} = lshl {dn}, %fzkh")
        e(f"  %fzSh2{t} = add.rr %fzSh{t}, %fzSh{t}")
        e(f"  %fzSr{t} = sub {dn}, %fzSh2{t}")
        e(f"  %fzSo{t} = mul %fzSr{t}, %fzrb")
        e(f"  %fzpa{t} = add.rr %fzdb, %fzSo{t}")

    def take(self, n):
        """n words."""
        assert n % 4 == 0
        if n == 0:
            return
        e, t = self.e, self.tag()
        dp = self.dp
        e(f"  %fzk{t} = mov.i32 {n // 4}")
        e(f"  %fzq0{t} = lda %fzp, 4")
        e(f"  %fzn0{t} = lda %fzp, 8")
        self.slot(f"%fzn0{t}", t)
        e(f"  %fzqb{t} = add.rr %fzq0{t}, %fzq0{t}")
        e(f"  %fzqo{t} = add.rr %fzqb{t}, %fzqb{t}")
        e(f"  %fzpq{t} = add.rr %fzpa{t}, %fzqo{t}")
        e(f"  %fzp0{t} = mov.scalar-to-address %fzpq{t}")
        e(f"  low.br ^fzt{t}(%zero: reg<aie2p.er>, %fzq0{t}: reg<aie2p.er>, %fzn0{t}: reg<aie2p.er>, %fzp0{t}: reg<aie2p.ep>)")
        e(f"^fzt{t}(%fzi{t}: reg<aie2p.er>, %fzq{t}: reg<aie2p.er>, %fzn{t}: reg<aie2p.er>, %fzpp{t}: reg<aie2p.ep>):")
        e(f"  %fzim{t} = lt %fzi{t}, %fzk{t}")
        e(f"  low.cond_br %fzim{t}, ^fztb{t}, ^fztx{t} : reg<aie2p.er>")
        e(f"^fztb{t}:")
        e(f"  %fzro{t} = lt %fzq{t}, %fzrw")
        e(f"  low.cond_br %fzro{t}, ^fztk{t}, ^fztr{t} : reg<aie2p.er>")
        e(f"^fztk{t}:")
        e(f"  low.br ^fztw{t}(%fzq{t}: reg<aie2p.er>, %fzn{t}: reg<aie2p.er>, %fzpp{t}: reg<aie2p.ep>)")
        e(f"^fztr{t}:")   # the record is full: send it, take the next slot
        e(f"  rel %one, {dp}")
        e(f"  %fzn1{t} = add.rr %fzn{t}, %one")
        e(f"  acq %am1, {dp}")
        self.slot(f"%fzn1{t}", t + "r")
        e(f"  %fzrp{t} = mov.scalar-to-address %fzpa{t}r")
        e(f"  low.br ^fztw{t}(%zero: reg<aie2p.er>, %fzn1{t}: reg<aie2p.er>, %fzrp{t}: reg<aie2p.ep>)")
        e(f"^fztw{t}(%fzwq{t}: reg<aie2p.er>, %fzwn{t}: reg<aie2p.er>, %fzwp{t}: reg<aie2p.ep>):")
        for w in range(4):
            e(f"  %fzv{t}_{w} = mov.ss")
            e(f"  st %fzv{t}_{w}, %fzwp{t}, {4 * w}")
        e(f"  %fzwp1{t} = padds.modifier %fzwp{t}, %fzm16")
        e(f"  %fzwq1{t} = add.rr %fzwq{t}, %fzfour")
        e(f"  %fzi1{t} = add.rr %fzi{t}, %one")
        e(f"  low.br ^fzt{t}(%fzi1{t}: reg<aie2p.er>, %fzwq1{t}: reg<aie2p.er>, %fzwn{t}: reg<aie2p.er>, %fzwp1{t}: reg<aie2p.ep>)")
        e(f"^fztx{t}:")
        e(f"  st %fzq{t}, %fzp, 4")
        e(f"  st %fzn{t}, %fzp, 8")

    def loop(self, n, body):
        """body() n times."""
        e, t = self.e, self.tag()
        e(f"  %fzlk{t} = mov.i32 {n}")
        e(f"  low.br ^fzl{t}(%zero: reg<aie2p.er>)")
        e(f"^fzl{t}(%fzli{t}: reg<aie2p.er>):")
        e(f"  %fzlm{t} = lt %fzli{t}, %fzlk{t}")
        e(f"  low.cond_br %fzlm{t}, ^fzlb{t}, ^fzlx{t} : reg<aie2p.er>")
        e(f"^fzlb{t}:")
        body()
        e(f"  %fzli1{t} = add.rr %fzli{t}, %one")
        e(f"  low.br ^fzl{t}(%fzli1{t}: reg<aie2p.er>)")
        e(f"^fzlx{t}:")


def fuse_prologue(L, cfg, role, ks, r):
    """A filling worker's firing (row r < rows - 1): the firing counter (private), the W slot by parity (leaf-synced
    W, acquired here and released at the exit); at each replay group's first firing (firings run group, M block,
    pass), that group's fill: per unit, skip the rows before its own, take its slab (row rows - 2 also the last
    row's: their panel is interleaved per unit), skip the rest (pass-through). A group's fill ends on a whole record;
    the record count runs on across groups and calls (the egress ring's slot parity)."""
    assert len(groups(cfg)) > 1, "fuse: the interleaved last-two-rows panel needs streamed replay groups"
    e = L.append
    rows = len(cfg.ks)
    dp, _ = fuse_ports(role)
    wrec = NP * slab(ks)
    sw = [slab(k) // 4 for k in cfg.ks]
    pre = sum(sw[:r])
    mine = sum(sw[r:]) if r == rows - 2 else sw[r]
    post = sum(sw[r + 1:]) - (mine - sw[r])
    e("  %fzs = storage {byte_alignment = 64, byte_length = 64} : low.storage<private>")
    e("  %fzp = storage_address %fzs : low.storage<private> -> reg<aie2p.ep>")
    e("  %fzjs = storage {byte_alignment = 64, byte_length = 64} : low.storage<private>")
    if cfg.fuse == 2:
        fuse2_entry(L, cfg, dp, r, wrec)
        return
    e("  %fzjp = storage_address %fzjs : low.storage<private> -> reg<aie2p.ep>")
    e("  %fzc = lda %fzp, 0")
    e("  %fzc1 = add.rr %fzc, %one")
    e(f"  %fzcl = mov.i32 {cfg.nb * cfg.passes}")
    e("  %fzcw = lt %fzc1, %fzcl")
    e("  %fzcn = mul %fzc1, %fzcw")
    e("  st %fzcn, %fzp, 0")
    e("  %fzkh = mova.i32 -1")
    e("  %fzh = lshl %fzc, %fzkh")
    e("  %fzh2 = add.rr %fzh, %fzh")
    e("  %fzpar = sub %fzc, %fzh2")
    e(f"  %fzwr = mov.i32 {wrec}")
    e("  %fzwo = mul %fzpar, %fzwr")
    e("  %fzwm = mov.modifier %fzwo")
    e("  %fzw0c = copy %w0 : reg<aie2p.ep> -> reg<aie2p.ep>")
    e("  %w = padds.modifier %fzw0c, %fzwm")
    e("  %fzdb = mov.address-to-scalar %fzd")
    fw, _ = fuse_record(cfg, r)
    e(f"  %fzrb = mov.i32 {4 * fw}")
    e(f"  %fzrw = mov.i32 {fw}")
    e("  %fzfour = mova.i32 4")
    e("  %fz16 = mova.i32 16")
    e("  %fzm16 = mov.modifier %fz16")
    fuse_dispatch(e, cfg)
    f = _Fill(e, dp)
    for i, g in enumerate(fuse_groups(cfg)):
        n = len(g) * NP
        e(f"^fzgo{i}:")
        e(f"  acq %am1, {dp}")

        def unit():
            f.skip(pre)
            f.take(mine)
            f.skip(post)
        f.loop(n, unit)
        e(f"  rel %one, {dp}")   # the group's last record (full)
        e(f"  %fzen{i} = lda %fzp, 8")
        e(f"  %fzen1{i} = add.rr %fzen{i}, %one")
        e("  st %zero, %fzp, 4")
        e(f"  st %fzen1{i}, %fzp, 8")
        e("  low.br ^fzgemm")
    e("^fzgemm:")
    e("  acq %am1, 1")


def fuse2_entry(L, cfg, dp, r, wrec):
    """fuse 2's entry: the firing counter and W slot parity, the leaf's pointers and the W slot offset parked in
    private storage (nothing lives across the decoding fill: it needs every register), the group dispatch, the fill
    (fuse_decode), then ^fzgemm rebuilds the pointers (W is acquired after the GEMM's constants)."""
    e = L.append
    e("  %fzc = lda %fzp, 0")
    e("  %fzc1 = add %fzc, 1")
    e(f"  %fzcl = mov.i32 {cfg.nb * cfg.passes}")
    e("  %fzcw = lt %fzc1, %fzcl")
    e("  %fzcn = mul %fzc1, %fzcw")
    e("  st %fzcn, %fzp, 0")
    e("  %fzc1o = sub %fzc1, %fzc")                     # 1, not a constant
    e("  %fzpar = and %fzc, %fzc1o")
    e(f"  %fzwr = mov.i32 {wrec}")
    e("  %fzwo = mul %fzpar, %fzwr")
    for o, nm in ((20, "%a_r"), (24, "%w0_r"), (28, "%fzd_r")):
        e(f"  %fzps{o} = mov.address-to-scalar {nm}")
        e(f"  st %fzps{o}, %fzp, {o}")
    e("  st %fzwo, %fzp, 4")   # (fuse 1's in-record word count slot)
    e("  %fzcf = mov.i32 780")                         # the GEMM's MMA config, loaded after the fill: as a constant,
    e("  st %fzcf, %fzp, 16")                          # the allocator's pressure repairs (for the decoder) cloned it at
                                                       # each of its 720 MMA uses (+2.3 KB of program)
    starts, s0 = [], 0
    for g in fuse_groups(cfg):
        starts.append(s0)
        s0 += cfg.nb * len(g)
    for i, st in enumerate(starts):   # firing == st: group i's fill
        last = i + 1 == len(starts)
        if st:
            e(f"  %fzbl{i} = mov.i32 {st}")
            e(f"  %fzml{i} = lt %fzc, %fzbl{i}")
            e(f"  low.cond_br %fzml{i}, ^fzgemm, ^fzck{i} : reg<aie2p.er>")
            e(f"^fzck{i}:")
            e(f"  %fzbh{i} = add %fzbl{i}, 1")
            e(f"  %fzmh{i} = lt %fzc, %fzbh{i}")
        else:
            e(f"  %fzmh{i} = eqz %fzc")
        e(f"  low.cond_br %fzmh{i}, ^fzgo{i}, {'^fzgemm' if last else f'^fzne{i}'} : reg<aie2p.er>")
        if not last:
            e(f"^fzne{i}:")
    for i, g in enumerate(fuse_groups(cfg)):
        e(f"^fzgo{i}:")
        e(f"  %fzgn{i} = mov.i32 {len(g) * NP}")
        e(f"  low.br ^fzdq(%fzgn{i}: reg<aie2p.er>)")
    fuse_decode(L, cfg, dp, r)
    e("^fzgemm:")
    e("  %fzpG = storage_address %fzs : low.storage<private> -> reg<aie2p.ep>")
    for o, nm in ((20, "a"), (24, "w0"), (28, "fzd")):
        e(f"  %fzpl{o} = lda %fzpG, {o}")
        e(f"  %{nm} = mov.scalar-to-address %fzpl{o}")
    e("  %fzwo2 = lda %fzpG, 4")
    e("  %fzwm = mov.modifier %fzwo2")
    e("  %fzw0c = copy %w0 : reg<aie2p.ep> -> reg<aie2p.ep>")
    e("  %w = padds.modifier %fzw0c, %fzwm")


def fuse_dispatch(e, cfg):
    """Branches to ^fzgo{i} at replay group i's first firing (%fzc: the firing number), else to ^fzgemm."""
    starts, s0 = [], 0
    for g in fuse_groups(cfg):
        starts.append(s0)
        s0 += cfg.nb * len(g)
    for i, st in enumerate(starts):   # firing == st: group i's fill (firings below st were checked before)
        last = i + 1 == len(starts)
        if st:
            e(f"  %fzbl{i} = mov.i32 {st}")
            e(f"  %fzml{i} = lt %fzc, %fzbl{i}")
            e(f"  low.cond_br %fzml{i}, ^fzgemm, ^fzck{i} : reg<aie2p.er>")
            e(f"^fzck{i}:")
        e(f"  %fzbh{i} = mov.i32 {st + 1}")
        e(f"  %fzmh{i} = lt %fzc, %fzbh{i}")
        e(f"  low.cond_br %fzmh{i}, ^fzgo{i}, {'^fzgemm' if last else f'^fzne{i}'} : reg<aie2p.er>")
        if not last:
            e(f"^fzne{i}:")


FZ_SP, FZ_IN, FZ_SP2, FZ_OUT = 0, 3264, 16384, 23616   # fuse 2 scratch (from the W ring's first 2048-aligned byte):
# decoder storage A (SCR_B), its input record ([16][RECROW]), storage B (SCR2_B, another bank: the table's second
# copy), its output (2 OUT_B records a super-block)


def fuse_decode(L, cfg, dp, r):
    """fuse 2: a replay group's fill (^fzdq(units)): per (pass, slab) unit, the raw stream's [16 rows][4 super-blocks]
    -> this row's super-blocks into the decoder's input record (the rest popped), the inlined IQ4_XS decoder, then the
    row's k-blocks (panel order [kb][h]) copied record by record into the egress ring. Scratch is the W ring (idle
    until this panel's drain, which starts after the last record). Unit index and count live in private storage
    (the decoder needs every register)."""
    e = L.append
    D = decoder("IQ4_XS")
    k0, k1, s0, nsb = fuse_span(cfg, r)
    fw, _ = fuse_record(cfg, r)
    W = FUSE_BLK // 4                                  # words per super-block
    pieces = fuse_pieces(cfg, r)
    assert FZ_SP + D.SCR_B <= FZ_IN and FZ_IN + 16 * D.RECROW <= FZ_SP2 and FZ_SP2 + D.SCR2_B <= FZ_OUT
    assert FZ_OUT + nsb * 2 * D.OUT_B + 2047 <= 2 * NP * slab(cfg.ks[r]) and fw % 16 == 0
    e("^fzdq(%fzN: reg<aie2p.er>):")
    e("  %fzpD = storage_address %fzs : low.storage<private> -> reg<aie2p.ep>")
    e("  st %fzN, %fzpD, 12")                       # the units left
    e("  low.br ^fzu")
    e("^fzu:")
    e("  %fzpU = storage_address %fzs : low.storage<private> -> reg<aie2p.ep>")
    e("  %fzu = lda %fzpU, 12")
    e("  %fzum = eqz %fzu")
    e("  low.cond_br %fzum, ^fzux, ^fzub : reg<aie2p.er>")
    e("^fzub:")
    e("  %fzR1 = mova.i32 1")
    e("  %fzR0 = mova.i32 0")
    # scratch pointers (re-derived each unit: nothing lives across the decoder)
    e("  %fzk2047 = mov.i32 2047")
    e("  %fzw0l = lda %fzpU, 24")
    e("  %fzwb = add.rr %fzw0l, %fzk2047")
    e("  %fzkm2048 = mov.i32 -2048")
    e("  %fzsb = and %fzwb, %fzkm2048")
    for nm, off in (("sp", FZ_SP), ("in", FZ_IN), ("sp2", FZ_SP2), ("out", FZ_OUT)):
        e(f"  %fzo_{nm} = mov.i32 {off}")
        e(f"  %fza_{nm} = add.rr %fzsb, %fzo_{nm}")
    e("  %fzjpR = storage_address %fzjs : low.storage<private> -> reg<aie2p.ep>")
    e("  %fzinp = mov.scalar-to-address %fza_in")
    # the raw unit: per row, skip / take / skip words (pairs)
    e("  %fz2 = mova.i32 2")
    e("  %fz8 = mova.i32 8")
    e("  %fzm8 = mov.modifier %fz8")
    e(f"  %fzrr = mov.i32 {D.RECROW}")
    e("  %fz16r = mova.i32 16")
    e("  low.br ^fzrw(%fzR0: reg<aie2p.er>, %fzinp: reg<aie2p.ep>)")
    e("^fzrw(%fzri: reg<aie2p.er>, %fzrp: reg<aie2p.ep>):")
    e("  %fzrm = lt %fzri, %fz16r")
    e("  low.cond_br %fzrm, ^fzrb, ^fzrx : reg<aie2p.er>")
    e("^fzrb:")

    def pairs(tag, n, take):
        """n words from the stream, as pairs: stored at %fzrp (take) or into the junk block."""
        if n == 0:
            return
        assert n % 2 == 0
        e(f"  %fzk{tag} = mov.i32 {n // 2}")
        if take:
            e(f"  low.br ^fz{tag}(%fzR0: reg<aie2p.er>, %fzrp: reg<aie2p.ep>)")
            e(f"^fz{tag}(%fzi{tag}: reg<aie2p.er>, %fzq{tag}: reg<aie2p.ep>):")
        else:
            e(f"  low.br ^fz{tag}(%fzR0: reg<aie2p.er>)")
            e(f"^fz{tag}(%fzi{tag}: reg<aie2p.er>):")
        e(f"  %fzm{tag} = lt %fzi{tag}, %fzk{tag}")
        e(f"  low.cond_br %fzm{tag}, ^fz{tag}b, ^fz{tag}x : reg<aie2p.er>")
        e(f"^fz{tag}b:")
        for w in range(2):
            e(f"  %fzv{tag}{w} = mov.ss")
            e(f"  st %fzv{tag}{w}, {f'%fzq{tag}' if take else '%fzjpR'}, {4 * w}")
        e(f"  %fzi1{tag} = add.rr %fzi{tag}, %fzR1")
        if take:
            e(f"  %fzq1{tag} = padds.modifier %fzq{tag}, %fzm8")
            e(f"  low.br ^fz{tag}(%fzi1{tag}: reg<aie2p.er>, %fzq1{tag}: reg<aie2p.ep>)")
        else:
            e(f"  low.br ^fz{tag}(%fzi1{tag}: reg<aie2p.er>)")
        e(f"^fz{tag}x:")
    pairs("pre", W * s0, False)
    pairs("tk", W * nsb, True)
    pairs("post", W * (FUSE_SB - s0 - nsb), False)
    e("  %fzrpa = mov.address-to-scalar %fzrp")
    e("  %fzrpb = add.rr %fzrpa, %fzrr")
    e("  %fzrpn = mov.scalar-to-address %fzrpb")
    e("  %fzri1 = add.rr %fzri, %fzR1")
    e("  low.br ^fzrw(%fzri1: reg<aie2p.er>, %fzrpn: reg<aie2p.ep>)")
    e("^fzrx:")
    for nm in ("sp", "in", "sp2", "out"):
        e(f"  %dq_{nm} = mov.scalar-to-address %fza_{nm}")
    # the decoder (its values and labels prefixed)
    ren = lambda m: m.group(1) + "dq_" + m.group(2)
    for ln in D.leaf_inline(nsb):
        ln = re.sub(r"([%^])([A-Za-z_]\w*)", lambda m: ren(m) if not m.group(2).startswith("dq_") else m.group(0), ln)
        e(ln if ln.startswith("^") else "  " + ln.strip())
    # the egress: per slice of this row's panel (rows - 2: its own and the last row's, interleaved per unit), its
    # padded slab from the decoded super-blocks (the pad copies whatever follows), a record (fw words) at a time
    e("  %fzE1 = mova.i32 1")
    e("  %fzE0 = mova.i32 0")
    e("  %fzEm1 = mova.i32 -1")
    e("  %fzk2047b = mov.i32 2047")
    e("  %fzpW = storage_address %fzs : low.storage<private> -> reg<aie2p.ep>")
    e("  %fzw0l2 = lda %fzpW, 24")
    e("  %fzwb2 = add.rr %fzw0l2, %fzk2047b")
    e("  %fzkm2048b = mov.i32 -2048")
    e("  %fzsb2 = and %fzwb2, %fzkm2048b")
    for i, (kb0, sw) in enumerate(pieces):
        x = f"^fze{i + 1}" if i + 1 < len(pieces) else "^fzex"
        e(f"  %fzoo{i} = mov.i32 {FZ_OUT + (kb0 - 32 * s0) * 144}")
        e(f"  %fzsrc{i} = add.rr %fzsb2, %fzoo{i}")
        e(f"  %fzsp{i} = mov.scalar-to-address %fzsrc{i}")
        e(f"  %fznr{i} = mov.i32 {sw // fw}")
        e(f"  low.br ^fze{i}h(%fzE0: reg<aie2p.er>, %fzsp{i}: reg<aie2p.ep>)")
        e(f"^fze{i}h(%fzei{i}: reg<aie2p.er>, %fzes{i}: reg<aie2p.ep>):")
        e(f"  %fzem{i} = lt %fzei{i}, %fznr{i}")
        e(f"  low.cond_br %fzem{i}, ^fze{i}b, {x} : reg<aie2p.er>")
        e(f"^fze{i}b:")
        e(f"  acq %fzEm1, {dp}")
        e(f"  %fzpE{i} = storage_address %fzs : low.storage<private> -> reg<aie2p.ep>")
        e(f"  %fzen{i} = lda %fzpE{i}, 8")
        e(f"  %fzekh{i} = mova.i32 -1")
        e(f"  %fzeh{i} = lshl %fzen{i}, %fzekh{i}")
        e(f"  %fzehh{i} = add.rr %fzeh{i}, %fzeh{i}")
        e(f"  %fzer{i} = sub %fzen{i}, %fzehh{i}")
        e(f"  %fzerb{i} = mov.i32 {4 * fw}")
        e(f"  %fzeo{i} = mul %fzer{i}, %fzerb{i}")
        e(f"  %fzdl{i} = lda %fzpE{i}, 28")
        e(f"  %fzea{i} = add.rr %fzdl{i}, %fzeo{i}")
        e(f"  %fzed{i} = mov.scalar-to-address %fzea{i}")
        if fw == 16:   # one 64-byte copy, straight-line (as a one-trip loop the records came out scrambled: unexplained)
            for q in range(4):
                e(f"  %fzsv{i}_{q} = vlda.128.i32x4 %fzes{i}, {16 * q}")
                e(f"  vst.128.i32x4 %fzsv{i}_{q}, %fzed{i}, {16 * q}")
            e(f"  %fzcs{i} = padda %fzes{i}, 64")
            e(f"  low.br ^fzc{i}x")
        else:
            e(f"  %fzck{i} = mov.i32 {fw // 16}")
            e(f"  low.br ^fzc{i}(%fzE0: reg<aie2p.er>, %fzes{i}: reg<aie2p.ep>, %fzed{i}: reg<aie2p.ep>)")
            e(f"^fzc{i}(%fzci{i}: reg<aie2p.er>, %fzcs{i}: reg<aie2p.ep>, %fzcd{i}: reg<aie2p.ep>):")
            e(f"  %fzcm{i} = lt %fzci{i}, %fzck{i}")
            e(f"  low.cond_br %fzcm{i}, ^fzc{i}b, ^fzc{i}x : reg<aie2p.er>")
            e(f"^fzc{i}b:")
            for q in range(4):
                e(f"  %fzcv{i}_{q} = vlda.128.i32x4 %fzcs{i}, {16 * q}")
                e(f"  vst.128.i32x4 %fzcv{i}_{q}, %fzcd{i}, {16 * q}")
            e(f"  %fzcs1{i} = padda %fzcs{i}, 64")
            e(f"  %fzcd1{i} = padda %fzcd{i}, 64")
            e(f"  %fzci1{i} = add.rr %fzci{i}, %fzE1")
            e(f"  low.br ^fzc{i}(%fzci1{i}: reg<aie2p.er>, %fzcs1{i}: reg<aie2p.ep>, %fzcd1{i}: reg<aie2p.ep>)")
        e(f"^fzc{i}x:")
        e(f"  rel %fzE1, {dp}")
        e(f"  %fzen1{i} = add.rr %fzen{i}, %fzE1")
        e(f"  st %fzen1{i}, %fzpE{i}, 8")
        e(f"  %fzei1{i} = add.rr %fzei{i}, %fzE1")
        e(f"  low.br ^fze{i}h(%fzei1{i}: reg<aie2p.er>, %fzcs{i}: reg<aie2p.ep>)")
        if i + 1 < len(pieces):
            e(f"^fze{i + 1}:")
    e("^fzex:")
    e("  %fzpX = storage_address %fzs : low.storage<private> -> reg<aie2p.ep>")
    e("  %fzu2 = lda %fzpX, 12")
    e("  %fzu21 = add %fzu2, -1")
    e("  st %fzu21, %fzpX, 12")
    e("  low.br ^fzu")
    e("^fzux:")
    e("  low.br ^fzgemm")


def cfg_row(cfg, role):
    return {"head": 0, "mid": 1, "tail": len(cfg.ks) - 1}[role]


def fill_leaf(L, cfg, nports):
    """Pass-through fill leaf: copies each port's record (in i -> out nports + i)."""
    e = L.append
    e(f"low.func.def schedule(locked) target<amd.xdna.aie2p.core>(@core_target) abi(object_function) @fill{nports}() asm {{")
    for i in range(2 * nports):
        e(f"  %r{i} = resource<native_pointer> {{index = {i}, source_type = buffer}} : reg<aie2p.ep>")
    for j in range(nports):
        src, dst = f"%r{j}", f"%r{nports + j}"
        cur = src
        for v in range(cfg.frec // 64):
            if v and v % 7 == 0:
                e(f"  %d{j}_{v} = padda {dst}, 448")
                dst = f"%d{j}_{v}"
            u = v // FILL_DIV                       # FILL_DIV > 1: each input vector written FILL_DIV times
            if v % FILL_DIV == 0:
                if u and u % 7 == 0:
                    e(f"  %s{j}_{v} = padda {cur}, 448")
                    cur = f"%s{j}_{v}"
                e(f"  %v{j}_{v} = vlda.512.i8x64 {cur}, {64 * (u % 7)}")
                last = f"%v{j}_{v}"
            e(f"  vst.512.i8x64 {last}, {dst}, {64 * (v % 7)}")
    e("  return")
    e("}")
    e("")


def leaf(L, cfg, role, ks, gate=None, pair=None, frow=None):
    e = L.append
    tail = role == "tail"
    fa = slab(ks)
    acap = cfg.acap_t if tail else cfg.acap
    a_adv = acap == MP   # a ring of MP slabs is addressed like a whole record: the base advances per iteration
    # mu 1 with a 2-slot ring: the slot alternates per iteration (body: the base steps +1 / -1 slab by parity)
    assert MP % cfg.mu == 0 and (a_adv or cfg.mu % acap == 0 or (cfg.mu, acap) == (1, 2)), \
        "ring slot of a row must be static per iteration"
    name = role + ("_g" if gate else "") + ("_s" if pair == "send" else "")
    fills = cfg.fuse and not tail
    if fills:
        frow = cfg_row(cfg, role) if frow is None else frow
        if frow == len(cfg.ks) - 2 and frow > 1:
            name = "mid_l"
    e(f"low.func.def schedule(locked) target<amd.xdna.aie2p.core>(@core_target) abi(object_function) @{name}() asm {{")
    rs = "_r" if fills and cfg.fuse == 2 else ""   # fuse 2: parked as scalars across the decoding fill (^fzgemm)
    e(f"  %a{rs} = resource<native_pointer> {{index = 0, source_type = buffer}} : reg<aie2p.ep>")
    if fills:   # the W ring's first slot and the fused fill's egress ring (fuse_prologue)
        e(f"  %w0{rs} = resource<native_pointer> {{index = 1, source_type = buffer}} : reg<aie2p.ep>")
        e(f"  %fzd{rs} = resource<native_pointer> {{index = {fuse_ports(role)[0]}, source_type = buffer}} : reg<aie2p.ep>")
    else:
        e("  %w = resource<native_pointer> {index = 1, source_type = buffer} : reg<aie2p.ep>")
    if gate:   # the neighbor channel to / from the relay (leaf-synchronized)
        e(f"  %gn1 = resource<native_pointer> {{index = {4 if gate == 'waiter' else 2}, source_type = buffer}} : reg<aie2p.ep>")
    if tail:
        e("  %o = resource<native_pointer> {index = 2, source_type = buffer} : reg<aie2p.ep>")
    if role != "head":
        e("  set.scd-enable 1")
    if role != "tail":
        e("  set.mcd-enable 1")
    else:
        e("  set.rounding 12")   # round to nearest even (the bf16 C)
    early = fills and cfg.fuse == 2   # the decoding fill first: nothing of the GEMM's lives across the decoder
    if early:
        fuse_prologue(L, cfg, role, ks, frow)
    if early:   # (fuse2_entry)
        e("  %fzpL = storage_address %fzs : low.storage<private> -> reg<aie2p.ep>")
        e("  %conf = lda %fzpL, 16")
    else:
        e("  %conf = mova.i32 780")
    e("  %am1 = mova.i32 -1")
    if early:
        e("  acq %am1, 1")
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
    if fills and not early:
        fuse_prologue(L, cfg, role, ks, frow)
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
        if pair:   # a segment's gate values (bf16) on their way through the core stream
            e(f"  %gbs = storage {{byte_alignment = 64, byte_length = {NP * 512}}} : low.storage<private>")
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
    if gate:   # job state: private storage, zeroed by the array setup of a core stream plan
        e("  %gst = storage {byte_alignment = 64, byte_length = 64} : low.storage<private>")
        e("  %gsp = storage_address %gst : low.storage<private> -> reg<aie2p.ep>")
    body(L, cfg, role, ks, a_adv, acap, gate, pair)


def gate_epilogue(L, cfg, gate):
    """The gated leaves' end of firing (see GATE): job accounting, background supply drain, the wait, the relay."""
    e = L.append
    S, FPC = cfg.gate, cfg.nb * cfg.passes
    e("  %g0 = mov.i32 0")
    e("  %g1 = mov.i32 1")
    e("  %g2 = mov.i32 2")
    e("  %gm1 = mov.i32 -1")
    e(f"  %gfpc = mov.i32 {FPC}")
    e("  %gF = lda %gsp, 0")
    e("  %gN = lda %gsp, 4")
    if gate == "waiter":
        e("  %gL = lda %gsp, 8")
        e("  %gQ = lda %gsp, 12")
        e("  %gseq = lda %gsp, 16")
        # read back last firing's ticks, then tick up to GB more of the leftover supply
        e("  low.br ^gq(%g0: reg<aie2p.er>)")
        e("^gq(%gqi: reg<aie2p.er>):")
        e("  %gqm = lt %gqi, %gQ")
        e("  low.cond_br %gqm, ^gq1, ^gqd : reg<aie2p.er>")
        e("^gq1:")
        for i in range(4):
            e(f"  %gqx{i} = mov.ss")
        e("  %gqn = add.rr %gqi, %g1")
        e("  low.br ^gq(%gqn: reg<aie2p.er>)")
        e("^gqd:")
        e(f"  %ggb = mov.i32 {GB}")
        e("  %gls = lt %gL, %ggb")          # L < GB: send L, else GB
        e("  %glsn = sub %g1, %gls")
        e("  %gb1 = mul %gL, %gls")
        e("  %gb2 = mul %ggb, %glsn")
        e("  %gbt = add.rr %gb1, %gb2")
        e("  low.br ^gt(%g0: reg<aie2p.er>)")
        e("^gt(%gti: reg<aie2p.er>):")
        e("  %gtm = lt %gti, %gbt")
        e("  low.cond_br %gtm, ^gt1, ^gtd : reg<aie2p.er>")
        e("^gt1:")
        for i in range(4):
            e("  mov.ms %g0")
        e("  %gtn = add.rr %gti, %g1")
        e("  low.br ^gt(%gtn: reg<aie2p.er>)")
        e("^gtd:")
        e("  %gL2 = sub %gL, %gbt")
    # this firing ends the job?
    e("  %gF1 = add.rr %gF, %g1")
    e("  %gnz = lt %gN, %g1")
    e("  %gdef = mul %gnz, %gfpc")
    e("  %glim = add.rr %gN, %gdef")
    e("  %gcont = lt %gF1, %glim")
    e("  low.cond_br %gcont, ^gmid, ^gend : reg<aie2p.er>")
    e("^gmid:")
    e("  st %gF1, %gsp, 0")
    if gate == "waiter":
        e("  st %gL2, %gsp, 8")
        e("  st %gbt, %gsp, 12")
    e("  return")
    e("^gend:")
    if gate == "waiter":
        # finish the leftover: the pending reads, then one tick + read per remaining record
        e("  low.br ^gr(%g0: reg<aie2p.er>)")
        e("^gr(%gri: reg<aie2p.er>):")
        e("  %grm = lt %gri, %gbt")
        e("  low.cond_br %grm, ^gr1, ^grd : reg<aie2p.er>")
        e("^gr1:")
        for i in range(4):
            e(f"  %grx{i} = mov.ss")
        e("  %grn = add.rr %gri, %g1")
        e("  low.br ^gr(%grn: reg<aie2p.er>)")
        e("^grd:")
        e("  low.br ^gl(%g0: reg<aie2p.er>)")
        e("^gl(%gli: reg<aie2p.er>):")
        e("  %glm = lt %gli, %gL2")
        e("  low.cond_br %glm, ^gl1, ^gld : reg<aie2p.er>")
        e("^gl1:")
        for i in range(4):
            e("  mov.ms %g0")
        for i in range(4):
            e(f"  %glx{i} = mov.ss")
        e("  %gln = add.rr %gli, %g1")
        e("  low.br ^gl(%gln: reg<aie2p.er>)")
        e("^gld:")
        # the wait for ready >= seq + 1
        e("  %gnext = add.rr %gseq, %g1")
        e(f"  %g64 = mov.i32 {GATE_CALLS}")
        e("  %gexp = mul %gnext, %g64")   # ready = sequence * GATE_CALLS + calls
        e(f"  %gsup = mov.i32 {S}")
        e("  %gfast = mov.i32 64")
        e(f"  %gp0 = mov.i32 {GP0}")
        e(f"  %gps = mov.i32 {GPS}")
        e("  low.br ^gp(%g0: reg<aie2p.er>, %g0: reg<aie2p.er>)")
        e("^gp(%gpn: reg<aie2p.er>, %gnc0: reg<aie2p.er>):")
        e("  %gpo = lt %gpn, %gsup")
        e("  low.cond_br %gpo, ^gask, ^gquit : reg<aie2p.er>")
        e("^gask:")
        for i in range(4):
            e("  mov.ms %g0")
        e("  %grdy = mov.ss")
        e("  %gy1 = mov.ss")
        e("  %gy2 = mov.ss")
        e("  %gy3 = mov.ss")
        e("  %gnc = sub %grdy, %gexp")   # the job's calls (when ready holds exactly its sequence)
        e("  %gpn1 = add.rr %gpn, %g1")
        e("  %glate = lt %grdy, %gexp")
        e("  low.cond_br %glate, ^gpace, ^gopen : reg<aie2p.er>")
        e("^gpace:")
        e("  %gslow = lt %gfast, %gpn1")
        e("  %gover = sub %gpn1, %gfast")
        e("  %govs = mul %gover, %gslow")
        e("  %ggrow = mul %govs, %gps")
        e("  %gpace = add.rr %gp0, %ggrow")
        e("  low.br ^gw(%g0: reg<aie2p.er>, %gpace: reg<aie2p.er>, %gpn1: reg<aie2p.er>, %g0: reg<aie2p.er>)")
        e("^gw(%gwi: reg<aie2p.er>, %gwl: reg<aie2p.er>, %gwp: reg<aie2p.er>, %gwc: reg<aie2p.er>):")
        e("  %gwm = lt %gwi, %gwl")
        e("  low.cond_br %gwm, ^gw1, ^gwd : reg<aie2p.er>")
        e("^gw1:")
        e("  %gwn = add.rr %gwi, %g1")
        e("  low.br ^gw(%gwn: reg<aie2p.er>, %gwl: reg<aie2p.er>, %gwp: reg<aie2p.er>, %gwc: reg<aie2p.er>)")
        e("^gwd:")
        e("  low.br ^gp(%gwp: reg<aie2p.er>, %gwc: reg<aie2p.er>)")
        e("^gopen:")
        e("  low.br ^ghand(%gpn1: reg<aie2p.er>, %gnc: reg<aie2p.er>, %g1: reg<aie2p.er>, %grdy: reg<aie2p.er>)")
        e("^gquit:")
        e("  low.br ^ghand(%gpn: reg<aie2p.er>, %gnc0: reg<aie2p.er>, %g2: reg<aie2p.er>, %gexp: reg<aie2p.er>)")
        e("^ghand(%ghp: reg<aie2p.er>, %ghc: reg<aie2p.er>, %ghs: reg<aie2p.er>, %ghv: reg<aie2p.er>):")
        e("  acq %gm1, 4")
        e("  st %ghv, %gn1, 0")
        e("  st %ghs, %gn1, 4")
        e("  st %ghp, %gn1, 8")
        e("  st %ghc, %gn1, 12")
        e("  rel %g1, 4")
        e("  %gNn = mul %ghc, %gfpc")
        e("  %gLn = sub %gsup, %ghp")
        e("  st %g0, %gsp, 0")
        e("  st %gNn, %gsp, 4")
        e("  st %gLn, %gsp, 8")
        e("  st %g0, %gsp, 12")
        e("  st %gnext, %gsp, 16")
        e("  return")
    else:
        e("  acq %gm1, 2")
        e("  %gexp = lda %gn1, 0")
        e("  %gsts = lda %gn1, 4")
        e("  %gpol = lda %gn1, 8")
        e("  %gnc = lda %gn1, 12")
        e("  rel %g1, 2")
        e("  mov.ms %gexp")
        e("  mov.ms %gsts")
        e("  mov.ms %gpol")
        e("  mov.ms %g0")
        e("  %gbad = lt %g1, %gsts")
        e(f"  %gfb = mov.i32 {GATE_FAILED - (1 << 32)}")   # the waiter gave up
        e("  %gmark = mul %gbad, %gfb")
        e("  %gdone = add.rr %gexp, %gmark")
        e("  mov.ms %gdone")
        e("  mov.ms %gsts")
        e("  mov.ms %gpol")
        e("  mov.ms %g0")
        e("  %gNn = mul %gnc, %gfpc")
        e("  st %g0, %gsp, 0")
        e("  st %gNn, %gsp, 4")
        e("  return")


def body(L, cfg, role, ks, a_adv, acap, gate=None, pair=None):
    """The locked stream: per sub-tile MMA + pop bundles, then the cascade boundary.
    head: chains start with mmul; after each sub-tile, 16 cascade writes (the next sub-tile's pops ride with them)
    mid:  mmul at k 0, then k 1-4 add the incoming partial quarter by quarter (fused cascade MMA); writes like the head
    tail: chains start from C, k 0-3 add the incoming quarters; after each sub-tile C is stored and the next loaded"""
    e = L.append
    tail = role == "tail"
    mid = role == "mid"
    NT = NP * cfg.mu      # sub-tiles per iteration, M-slab major
    # swiglu tail: the A pointer is loop-carried as a scalar, a short-lived pointer per sub-tile. Its store fifo pins
    # p2; a pointer live across the loop would otherwise be pushed into p0 / p1, which the operand load fifos need.
    spa = tail and cfg.swiglu
    pat = "reg<aie2p.er>" if spa else "reg<aie2p.ep>"
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
        if cfg.swiglu:   # a segment's C slot is SW_SLOT bytes: the pointers skip its tail past the f32 segment
            e(f"  %swgap = mov.i32 {SW_SLOT - 1024 * NP}")
            e("  %lgap = mul %swgap, %nf")
            e("  %mlb = mov.modifier %lgap")
            e("  %mcb = mov.modifier %swgap")
        pla, plp = ", %plb: reg<aie2p.ep>", ", %pl: reg<aie2p.ep>"
    else:
        po0, pct = "%zero", "reg<aie2p.er>"
        pla = plp = ""
    a0 = "%a"
    if spa:
        e("  %as0 = mov.address-to-scalar %a")
        a0 = "%as0"
    e(f"  low.br ^outer(%zero: reg<aie2p.er>, {a0}: {pat}, {po0}: {pct}, %fa0: reg<aie2p.eldfiforeg>, %fw0: reg<aie2p.eldfiforeg>{pla})")
    e(f"^outer(%mp: reg<aie2p.er>, %pa: {pat}, %po: {pct}, %fa: reg<aie2p.eldfiforeg>, %fw: reg<aie2p.eldfiforeg>{plp}):")
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
        # spa: a distinct opaque zero per sub-tile (outer counter >> 8 + t) keeps the conversions from being merged
        # into one long-lived pointer
        zs = [f"  %{x}kz = mova.i32 {-(8 + t)}", f"  %{x}oz = lshl %mp, %{x}kz", f"  %{x}pas = add.rr %pa, %{x}oz"] if spa else []
        if spa:
            ap, asrc = zs + [f"  %{x}pae = mov.scalar-to-address %{x}pas"], f"%{x}pae"
        sm = tm if a_adv else tm % acap
        if sm:
            ap = zs + [f"  %{x}am0 = mov.scalar-to-address %{x}pas"] if spa else [f"  %{x}am0 = copy %pa : reg<aie2p.ep> -> reg<aie2p.ep>"]
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
            if cfg.swiglu and row_end:
                e(f"  %pl{t + 1}g = padds.modifier %plc{t + 1}, %mls")
                e(f"  %pl{t + 1} = padds.modifier %pl{t + 1}g, %mlb")
            else:
                e(f"  %pl{t + 1} = padds.modifier %plc{t + 1}, %mls")
        late = row_end and nxt   # the convert runs between the stores and the next C loads
        for c in range(4):
            for q in range(4):
                if nxt and not late:
                    e(f"  %l{t + 1}_{c}_{q} = vlda.acc {pl[t + 1]}, {256 * c + 64 * q - 512}")
                e(f"  %o{t}_{c}_{q} = slice {cur[c]}[{q}] : reg<aie2p.mbms x4> -> reg<aie2p.mbms>")
                e(f"  vst.acc %o{t}_{c}_{q}, {pc[t]}, {256 * c + 64 * q - 512}")
        def convert(base, lp, back):
            """Last pass: pack the segment's f32 C (NP sub-tiles) in place to bf16 in its first half, which the C ring sends
            (LOOM_EXP_LS_SEND_PITCH=2). base is a store pointer back bytes past the segment's slot + 512; on a last pass the C
            load pointer lp equals it, so lp walks the f32 source (post-increment loads) while base walks the bf16
            destination (post-increment stores); both are rebuilt for the join.
            It runs between a sub-tile's C stores and the next C loads, so all accumulators are free and loads run 16 ahead.
            Returns the join's (base, lp)."""
            e(f"  low.cond_br %clast, ^cv{t}, ^nc{t} : reg<aie2p.er>")
            e(f"^nc{t}:")
            e(f"  low.br ^cj{t}({base}: reg<aie2p.ep>, {lp}: reg<aie2p.ep>)")
            e(f"^cv{t}:")
            d, k, rest = base, 0, back + 512
            while rest > 0:
                step = min(rest, 448)
                rest -= step
                e(f"  %cvb{t}_{k} = padda {d}, {-step}")
                d, k = f"%cvb{t}_{k}", k + 1
            e(f"  %cvs{t}_0 = copy {d} : reg<aie2p.ep> -> reg<aie2p.ep>")
            s_ = f"%cvs{t}_0"
            n = 16 * NP

            def load(i):
                nonlocal s_
                e(f"  %cvl{t}_{i}, %cvs{t}_{i + 1} = vlda.acc.post {s_}, 64")
                s_ = f"%cvs{t}_{i + 1}"
            for i in range(16):
                load(i)
            for i in range(n):
                e(f"  %cvd{t}_{i} = vst.convert.f32x16.to.bf16x16.post %cvl{t}_{i}, {d}, 32")
                d = f"%cvd{t}_{i}"
                if i + 16 < n:
                    load(i + 16)
            bump(L, f"%cvr{t}", d, back + 512 - 32 * n)
            e(f"  %cvq{t} = copy %cvr{t} : reg<aie2p.ep> -> reg<aie2p.ep>")
            e(f"  low.br ^cj{t}(%cvr{t}: reg<aie2p.ep>, %cvq{t}: reg<aie2p.ep>)")
            e(f"^cj{t}(%cj{t}p: reg<aie2p.ep>, %cj{t}l: reg<aie2p.ep>):")
            e(f"  %cqr{t} = copy %clast : reg<aie2p.er> -> reg<aie2p.mr26_lock>")
            e(f"  rel.cond %one, %cqr{t}, 2")
            return f"%cj{t}p", f"%cj{t}l"
        def convert_f32(base, lp, back):
            """Last pass (cfg.ofeat): permute each (sub-tile, mh) half of the segment's f32 C in place from [nh][row][8]
            to [row][nh][8] (whole 32-byte chain rows), so the slot is [sub-tile][16 tokens][16 f32] for the strided C
            receiver (fragment-major output). A half's 16 rows are all loaded before any is stored. Returns the join's (base, lp)."""
            e(f"  low.cond_br %clast, ^cv{t}, ^nc{t} : reg<aie2p.er>")
            e(f"^nc{t}:")
            e(f"  low.br ^cj{t}({base}: reg<aie2p.ep>, {lp}: reg<aie2p.ep>)")
            e(f"^cv{t}:")
            e(f"  %cvk{t} = copy {base} : reg<aie2p.ep> -> reg<aie2p.ep>")
            d, k, rest = f"%cvk{t}", 0, back + 512 - 256   # the first half's middle
            while rest > 0:
                step = min(rest, 448)
                rest -= step
                e(f"  %cvb{t}_{k} = padda {d}, {-step}")
                d, k = f"%cvb{t}_{k}", k + 1
            for h in range(2 * NP):
                if h:
                    bump(L, f"%cvh{t}_{h}", d, 512)
                    d = f"%cvh{t}_{h}"
                # piece i = 8 nh + row (32 bytes at 32 i) belongs at 2 row + nh: a perfect shuffle, cycles of <= 4 pieces
                for cyc in SHUFFLE16_CYCLES:
                    for i in cyc:
                        e(f"  %cvx{t}_{h}_{i} = vlda.256.i32x8 {d}, {32 * i - 256}")
                    for i in cyc:
                        e(f"  vst.256.i32x8 %cvx{t}_{h}_{i}, {d}, {32 * (2 * (i % 8) + i // 8) - 256}")
            e(f"  %cvq{t} = copy {base} : reg<aie2p.ep> -> reg<aie2p.ep>")
            e(f"  low.br ^cj{t}({base}: reg<aie2p.ep>, %cvq{t}: reg<aie2p.ep>)")
            e(f"^cj{t}(%cj{t}p: reg<aie2p.ep>, %cj{t}l: reg<aie2p.ep>):")
            e(f"  %cqr{t} = copy %clast : reg<aie2p.er> -> reg<aie2p.mr26_lock>")
            e(f"  rel.cond %one, %cqr{t}, 2")
            return f"%cj{t}p", f"%cj{t}l"
        def convert_swiglu(base, lp, back, segs=1):
            """Last pass (cfg.swiglu): slabs 0 .. NP / 2 - 1 of a segment are gate rows, the rest up rows of the same
            features. Per 8 x 8 block pair (8 tokens, 8 features) h = silu(g) * u through bf16 products with f32
            results (sigmoid below); a segment's 2 NP h blocks ([feature block][token half] in push order) go out as
            bfp16ebs8 fragments (SW_REC) at its slot start, in place (every block is read before the store fifo reaches
            it). A loop over the iteration's segs segments (the earlier ones deferred to here; the 16 KB program memory
            holds one block's code). Returns the join's (base, lp)."""
            e(f"  low.cond_br %clast, ^cv{t}, ^nc{t} : reg<aie2p.er>")
            e(f"^nc{t}:")
            e(f"  low.br ^cj{t}({base}: reg<aie2p.ep>, {lp}: reg<aie2p.ep>)")
            e(f"^cv{t}:")
            d, k, rest = base, 0, back + 512 + SW_SLOT * (segs - 1)   # the first deferred segment's slot
            while rest > 0:
                step = min(rest, 448)
                rest -= step
                e(f"  %cvb{t}_{k} = padda {d}, {-step}")
                d, k = f"%cvb{t}_{k}", k + 1
            p = f"%sw{t}_"
            V1, V2 = "reg<aie2p.vec256>", "reg<aie2p.vec256 x2>"
            M1, M2, M4 = "reg<aie2p.mbms>", "reg<aie2p.mbms x2>", "reg<aie2p.mbms x4>"
            ER = "reg<aie2p.er>"
            e(f"  {p}ds = mov.address-to-scalar {d}")
            e(f"  {p}k0 = mova.i32 0")
            e(f"  {p}nseg = mova.i32 {segs}")
            e(f"  {p}kslot = mov.i32 {SW_SLOT}")
            e(f"  low.br ^swl{t}({p}k0: {ER})")
            e(f"^swl{t}({p}k: {ER}):")
            e(f"  {p}more = lt {p}k, {p}nseg")
            e(f"  low.cond_br {p}more, ^swb{t}, ^swx{t} : {ER}")
            e(f"^swb{t}:")
            e(f"  {p}so = mul {p}k, {p}kslot")
            e(f"  {p}ss = add.rr {p}ds, {p}so")
            e(f"  {p}c = mova.i32 60")

            kn = [0]

            def kbf(x):                                         # a bf16 constant in 32 lanes, made at its use
                kn[0] += 1
                e(f"  {p}kc{kn[0]} = mov.i32 {bf16_bits(x)}")
                e(f"  {p}kv{kn[0]} = vbcst.16 {p}kc{kn[0]}")
                return f"{p}kv{kn[0]}"

            def kacc(x):                                        # an f32 constant (exact in bf16) in 32 accumulator lanes
                kn[0] += 1
                nm, one_, x_ = f"{p}ka{kn[0]}", kbf(1.0), kbf(x)
                e(f"  {nm} = vmul.bf16x32 {one_}, {x_}, {p}c")
                return nm
            L2E = 1 / math.log(2)
            e(f"  {p}amk = mov.i32 32767")                      # bf16 |x| mask
            e(f"  {p}amv = vbcst.16 {p}amk")
            C2, C1, C3 = bf16_round(math.log(2) ** 2 / 2), bf16_round(math.log(2)), math.log(2) ** 3 / 6
            if pair:   # the gate buffer's address as a scalar
                e(f"  {p}gba = storage_address %gbs : low.storage<private> -> reg<aie2p.ep>")
                e(f"  {p}gbs0 = mov.address-to-scalar {p}gba")
                e(f"  {p}k32 = mov.i32 32")
                e(f"  {p}k64n = mova.i32 {NP * 16}")   # 512-bit vector loops: 32 bytes x NP * 16 per segment
            if pair == "recv":   # the gate tail's segment: NP * 512 bytes of bf16 from the core stream
                e(f"  {p}rq0 = mova.i32 0")
                e(f"  low.br ^rv{t}({p}rq0: {ER})")
                e(f"^rv{t}({p}rq: {ER}):")
                e(f"  {p}rqm = lt {p}rq, {p}k64n")
                e(f"  low.cond_br {p}rqm, ^rvb{t}, ^rvx{t} : {ER}")
                e(f"^rvb{t}:")
                e(f"  {p}rqo = mul {p}rq, {p}k32")
                e(f"  {p}rqa = add.rr {p}gbs0, {p}rqo")
                e(f"  {p}rqp = mov.scalar-to-address {p}rqa")
                for w in range(8):
                    e(f"  {p}rw{w} = mov.ss")
                    e(f"  st {p}rw{w}, {p}rqp, {4 * w}")
                e(f"  {p}rq1 = add.rr {p}rq, %one")
                e(f"  low.br ^rv{t}({p}rq1: {ER})")
                e(f"^rvx{t}:")
            e(f"  {p}sp = mov.scalar-to-address {p}ss")
            e(f"  {p}sl = vlda.store-fifo.low512 {p}sp, 0")
            e(f"  {p}f0 = vlda.store-fifo.high512 {p}sp, {p}sl, 64")
            e(f"  {p}op = copy {p}sp : reg<aie2p.ep> -> reg<aie2p.mpfs>")
            e(f"  {p}ps = mova.fifo.store.position 0")
            fifo = (f"{p}f0", f"{p}op", f"{p}ps")
            ptrn = [0]

            def ptr(sub):                                       # this block's sub-tile: offsets -512 .. 448 around its middle
                ptrn[0] += 1                                    # short-lived (the tail's ep budget is 8)
                q_ = f"{p}pp{ptrn[0]}"
                e(f"  {q_} = mov.scalar-to-address {sub}")
                return q_
            nf = 0
            MF, MP_, MS = "reg<aie2p.mstfifo>", "reg<aie2p.mpfs>", "reg<aie2p.mr26_fifo_st>"
            e(f"  {p}i0 = mova.i32 0")
            e(f"  {p}k1024 = mov.i32 1024")
            # outer loop j over the gate slabs (column pairs: every slab), inner i over their two 8-row halves
            e(f"  {p}nj = mova.i32 {NP if pair else NP // 2}")
            e(f"  low.br ^swj{t}({p}i0: {ER}, {fifo[0]}: {MF}, {fifo[1]}: {MP_}, {fifo[2]}: {MS})")
            e(f"^swj{t}({p}j: {ER}, {p}JF: {MF}, {p}JO: {MP_}, {p}JP: {MS}):")
            e(f"  {p}jm = lt {p}j, {p}nj")
            e(f"  low.cond_br {p}jm, ^swjb{t}, ^swjx{t} : {ER}")
            e(f"^swjb{t}:")
            if pair == "recv":   # the record as two store streams (one stream lost its 18th line): the second at 576
                h = NP // 2
                e(f"  {p}kh1 = mova.i32 {h - 1}")
                e(f"  {p}kh3 = mova.i32 {h + 1}")
                e(f"  {p}ja1 = lt {p}kh1, {p}j")
                e(f"  {p}jb3 = lt {p}j, {p}kh3")
                e(f"  {p}jeq = mul {p}ja1, {p}jb3")
                e(f"  low.cond_br {p}jeq, ^swre{t}, ^swno{t} : {ER}")
                e(f"^swno{t}:")
                e(f"  low.br ^swin{t}({p}JF: {MF}, {p}JO: {MP_}, {p}JP: {MS})")
                e(f"^swre{t}:")
                e(f"  {p}rf, {p}rp, {p}rq_ = vst.flush.512 {p}JF, {p}JO, {p}JP")
                e(f"  {p}k576 = mov.i32 {2 * h * 144}")
                e(f"  {p}s2 = add.rr {p}ss, {p}k576")
                e(f"  {p}sp2 = mov.scalar-to-address {p}s2")
                e(f"  {p}sl2 = vlda.store-fifo.low512 {p}sp2, 0")
                e(f"  {p}f2 = vlda.store-fifo.high512 {p}sp2, {p}sl2, 64")
                e(f"  {p}op2 = copy {p}sp2 : reg<aie2p.ep> -> reg<aie2p.mpfs>")
                e(f"  {p}ps2 = mova.fifo.store.position 0")
                e(f"  low.br ^swin{t}({p}f2: {MF}, {p}op2: {MP_}, {p}ps2: {MS})")
                e(f"^swin{t}({p}NF: {MF}, {p}NO: {MP_}, {p}NP: {MS}):")
                e(f"  {p}ji0 = mova.i32 0")
                e(f"  low.br ^swi{t}({p}ji0: {ER}, {p}NF: {MF}, {p}NO: {MP_}, {p}NP: {MS})")
            else:
                e(f"  {p}ji0 = mova.i32 0")
                e(f"  low.br ^swi{t}({p}ji0: {ER}, {p}JF: {MF}, {p}JO: {MP_}, {p}JP: {MS})")
            e(f"^swi{t}({p}i: {ER}, {p}F: {MF}, {p}O: {MP_}, {p}P: {MS}):")
            e(f"  {p}nbk = mova.i32 2")
            e(f"  {p}im = lt {p}i, {p}nbk")
            e(f"  low.cond_br {p}im, ^swib{t}, ^swix{t} : {ER}")
            e(f"^swib{t}:")
            # gate rows: slab j, half i (sub-tiles at 256 i, 512 + 256 i); up: NP / 2 slabs on. The record takes the
            # 8-feature blocks 2 j + i in loop order; P4 (and Q4K's 4 gate k-blocks of one format, its slabs 32 rows
            # apart): per two slabs blocks 0, 2, 1, 3, so the loop's (j, i) stand for slab (j & ~1) | i, half j & 1;
            # Q4K over 4 slabs (pairs): blocks 0, 4, 1, 5, 2, 6, 3, 7, slab (j >> 1) + 2 i, half j & 1
            jj, ii, jstep = f"{p}j", f"{p}i", 1
            if cfg.hord == "Q4K" and pair:
                e(f"  {p}hk1 = mov.i32 1")
                e(f"  {p}hk2 = mov.i32 2")
                e(f"  {p}jod = and {p}j, {p}hk1")
                e(f"  {p}jev = sub {p}j, {p}jod")
                e(f"  {p}jhf = mul {p}i, {p}hk2")
                e(f"  {p}jh2 = add.rr {p}jev, {p}jhf")
                e(f"  {p}jpe = add.rr {p}jh2, {p}jhf")      # 2 slab: the slab offsets below take half the step
                jj, ii, jstep = f"{p}jpe", f"{p}jod", 2
            elif cfg.hord:
                e(f"  {p}hk1 = mov.i32 1")
                e(f"  {p}jod = and {p}j, {p}hk1")
                e(f"  {p}jev = sub {p}j, {p}jod")
                e(f"  {p}jpe = add.rr {p}jev, {p}i")
                jj, ii = f"{p}jpe", f"{p}jod"
            e(f"  {p}jsl = mov.i32 {1024 // jstep}")
            e(f"  {p}jo = mul {jj}, {p}jsl")
            e(f"  {p}k256 = mov.i32 256")
            e(f"  {p}io = mul {ii}, {p}k256")
            e(f"  {p}ja = add.rr {p}ss, {p}jo")
            e(f"  {p}ia = add.rr {p}ja, {p}io")
            e(f"  {p}k512 = mov.i32 512")
            e(f"  {p}gib = add.rr {p}ia, {p}k512")
            if pair:   # the slab is up (recv) or gate (send) rows; the gate values' buffer block (j, i): 256 bytes
                e(f"  {p}uib = add.rr {p}gib, %zero")
                e(f"  {p}kj512 = mov.i32 {512 // jstep}")
                e(f"  {p}gjo = mul {jj}, {p}kj512")
                e(f"  {p}gio = mul {ii}, {p}k256")
                e(f"  {p}gja = add.rr {p}gbs0, {p}gjo")
                e(f"  {p}gbb = add.rr {p}gja, {p}gio")
            else:
                e(f"  {p}kup = mov.i32 {1024 * NP // 2}")
                e(f"  {p}uib = add.rr {p}gib, {p}kup")
            fifo = (f"{p}F", f"{p}O", f"{p}P")
            for i in (0,):                                  # one 8-feature block per loop body
                for r in range(2):                              # token half: chain row bit
                    n = f"{p}b{r}"

                    def load(sub, tag):
                        pp = ptr(sub)
                        q4 = []
                        for q in range(4):
                            x = f"{n}{tag}l{q}"
                            e(f"  {x} = vlda.acc {pp}, {512 * r + 64 * q - 512}")
                            q4.append(x)
                        return q4
                    if pair == "send":   # the gate values to the buffer, (r, h) blocks of 32 bf16
                        g4 = load(f"{p}gib", "g")
                        for h in range(2):
                            g2 = f"{n}g2{h}"
                            e(f"  {g2} = concat({g4[2 * h]}, {g4[2 * h + 1]}) : ({M1}, {M1}) -> {M2}")
                            e(f"  {n}gb{h} = vconv.bf16.fp32 {g2}")
                            e(f"  {n}gp{h} = mov.scalar-to-address {p}gbb")   # short-lived (the ep budget is 8)
                            e(f"  vst.512.bf16x32 {n}gb{h}, {n}gp{h}, {64 * (2 * r + h)}")
                        continue
                    if pair == "recv":
                        u4 = load(f"{p}uib", "u")
                    else:
                        g4, u4 = load(f"{p}gib", "g"), load(f"{p}uib", "u")
                    hs = []
                    for h in range(2):
                        u2 = f"{n}u2{h}"
                        e(f"  {u2} = concat({u4[2 * h]}, {u4[2 * h + 1]}) : ({M1}, {M1}) -> {M2}")
                        if pair == "recv":
                            e(f"  {n}gp{h} = mov.scalar-to-address {p}gbb")
                            e(f"  {n}gb{h} = vlda.512.bf16x32 {n}gp{h}, {64 * (2 * r + h)}")
                        else:
                            g2 = f"{n}g2{h}"
                            e(f"  {g2} = concat({g4[2 * h]}, {g4[2 * h + 1]}) : ({M1}, {M1}) -> {M2}")
                            e(f"  {n}gb{h} = vconv.bf16.fp32 {g2}")
                        e(f"  {n}ub{h} = vconv.bf16.fp32 {u2}")

                        # vmul / vmac.bf16x32: 32 f32 lanes in the first two units of the x4 accumulator
                        def lo32(v, tag):
                            q0, q1, r2 = f"{n}{tag}0", f"{n}{tag}1", f"{n}{tag}2"
                            e(f"  {q0} = slice {v}[0] : {M4} -> {M1}")
                            e(f"  {q1} = slice {v}[1] : {M4} -> {M1}")
                            e(f"  {r2} = concat({q0}, {q1}) : ({M1}, {M1}) -> {M2}")
                            return q0, q1, r2
                        # sigmoid(g): e = e^-|g| = 2^n 2^r (n = round(-|g| log2 e); vexp2 is Mitchell's 2^floor(z)
                        # (1 + frac(z)), exact only at integers; 2^r by a cubic), y = 1 / (1 + e) by two Newton steps
                        # from 1.5 - d / 2, sigmoid = y for g >= 0, e y for g < 0
                        def bf(v, tag):                         # the 32 f32 lanes of an x4 accumulator as bf16
                            return f"{n}{tag}b{h}", e(f"  {n}{tag}b{h} = vconv.bf16.fp32 {lo32(v, f'{tag}q{h}')[2]}")
                        e(f"  {n}ga{h} = vband {n}gb{h}, {p}amv")                     # |g|
                        e(f"  {n}z0{h} = vmul.bf16x32 {n}ga{h}, {kbf(-bf16_round(L2E))}, {p}c")
                        e(f"  {n}z{h} = vmac.bf16x32 {n}z0{h}, {n}ga{h}, {kbf(-(L2E - bf16_round(L2E)))}, {p}c")   # z = -|g| log2 e (f32)
                        e(f"  {n}t1{h} = vadd.f32x64 {n}z{h}, {kacc(12582912.0)}, {p}c")
                        e(f"  {n}nf{h} = vsub.f32x64 {n}t1{h}, {kacc(12582912.0)}, {p}c")    # n = round(z)
                        e(f"  {n}r{h} = vsub.f32x64 {n}z{h}, {n}nf{h}, {p}c")     # r = z - n, |r| <= 1/2
                        rb = bf(f"{n}r{h}", "r")[0]
                        n0, n1, _ = lo32(f"{n}nf{h}", f"nq{h}")
                        e(f"  {n}x{h}0 = vexp2.bf16x16 {n0}")                    # 2^n, exact
                        e(f"  {n}x{h}1 = vexp2.bf16x16 {n1}")
                        e(f"  {n}xx{h} = concat({n}x{h}0, {n}x{h}1) : ({V1}, {V1}) -> {V2}")
                        e(f"  {n}h0{h} = vmac.bf16x32 {kacc(C2)}, {rb}, {kbf(C3)}, {p}c")
                        hb = bf(f"{n}h0{h}", "hh0")[0]
                        e(f"  {n}h1{h} = vmac.bf16x32 {kacc(C1)}, {rb}, {hb}, {p}c")
                        hb = bf(f"{n}h1{h}", "hh1")[0]
                        e(f"  {n}h2{h} = vmac.bf16x32 {kacc(1.0)}, {rb}, {hb}, {p}c")    # 2^r
                        pb = bf(f"{n}h2{h}", "hh2")[0]
                        e(f"  {n}ee{h} = vmul.bf16x32 {n}xx{h}, {pb}, {p}c")       # e = e^-|g|
                        eb = bf(f"{n}ee{h}", "e")[0]
                        e(f"  {n}d{h} = vmac.bf16x32 {kacc(1.0)}, {eb}, {kbf(1.0)}, {p}c")  # d = 1 + e
                        db = bf(f"{n}d{h}", "dd")[0]
                        e(f"  {n}y0{h} = vmac.bf16x32 {kacc(1.5)}, {db}, {kbf(-0.5)}, {p}c")  # y0 = 1.5 - d / 2
                        yb = bf(f"{n}y0{h}", "y0")[0]
                        e(f"  {n}nd{h} = vneg.f32x64 {n}d{h}, {p}c")
                        ndb = bf(f"{n}nd{h}", "nd")[0]
                        for it in range(2):                     # y <- y (2 - d y)
                            e(f"  {n}w{it}{h} = vmac.bf16x32 {kacc(2.0)}, {ndb}, {yb}, {p}c")
                            wb = bf(f"{n}w{it}{h}", f"w{it}")[0]
                            e(f"  {n}y{it + 1}{h} = vmul.bf16x32 {yb}, {wb}, {p}c")
                            yb = bf(f"{n}y{it + 1}{h}", f"y{it + 1}")[0]
                        e(f"  {n}sn{h} = vmul.bf16x32 {eb}, {yb}, {p}c")           # e y: sigmoid(g) for g < 0
                        snb = bf(f"{n}sn{h}", "sn")[0]
                        e(f"  {n}lt{h} = vlt.s16x32.el.low32 {n}gb{h}, {kbf(0.0)}")      # g < 0 (sign bit)
                        e(f"  {n}sgb{h} = vsel.16.mask64 {yb}, {snb}, {n}lt{h}")
                        e(f"  {n}si{h} = vmul.bf16x32 {n}gb{h}, {n}sgb{h}, {p}c")      # silu(g)
                        e(f"  {n}sib{h} = vconv.bf16.fp32 {lo32(f'{n}si{h}', f'siq{h}')[2]}")
                        e(f"  {n}h{h} = vmul.bf16x32 {n}sib{h}, {n}ub{h}, {p}c")       # silu(g) * u
                        q0, q1, _ = lo32(f"{n}h{h}", f"hq{h}")
                        hs += [q0, q1]
                    e(f"  {n}H = concat({hs[0]}, {hs[1]}, {hs[2]}, {hs[3]}) : ({M1}, {M1}, {M1}, {M1}) -> {M4}")
                    nf += 1
                    nxt = (f"{p}xf{nf}", f"{p}xp{nf}", f"{p}xq{nf}")
                    e(f"  {nxt[0]}, {nxt[1]}, {nxt[2]} = vst.push.bfp16ebs8.from.fp32 {fifo[0]}, {n}H, {fifo[1]}, {fifo[2]}")
                    fifo = nxt
            e(f"  {p}i1 = add.rr {p}i, %one")
            e(f"  low.br ^swi{t}({p}i1: {ER}, {fifo[0]}: {MF}, {fifo[1]}: {MP_}, {fifo[2]}: {MS})")
            e(f"^swix{t}:")
            fifo = (f"{p}F", f"{p}O", f"{p}P")
            e(f"  {p}j1 = add.rr {p}j, %one")
            e(f"  low.br ^swj{t}({p}j1: {ER}, {fifo[0]}: {MF}, {fifo[1]}: {MP_}, {fifo[2]}: {MS})")
            e(f"^swjx{t}:")
            fifo = (f"{p}JF", f"{p}JO", f"{p}JP")
            if pair == "send":   # the buffer out on the core stream, 8 words at a time
                e(f"  {p}sq0 = mova.i32 0")
                e(f"  low.br ^sd{t}({p}sq0: {ER})")
                e(f"^sd{t}({p}sq: {ER}):")
                e(f"  {p}sqm = lt {p}sq, {p}k64n")
                e(f"  low.cond_br {p}sqm, ^sdb{t}, ^sdx{t} : {ER}")
                e(f"^sdb{t}:")
                e(f"  {p}sqo = mul {p}sq, {p}k32")
                e(f"  {p}sqa = add.rr {p}gbs0, {p}sqo")
                e(f"  {p}sqp = mov.scalar-to-address {p}sqa")
                for w in range(8):
                    e(f"  {p}sw{w} = lda {p}sqp, {4 * w}")
                    e(f"  mov.ms {p}sw{w}")
                e(f"  {p}sq1 = add.rr {p}sq, %one")
                e(f"  low.br ^sd{t}({p}sq1: {ER})")
                e(f"^sdx{t}:")
            if pair != "send":
                # the flush writes whole 64-byte lines only, and drops the last one when the stream ends on a 128-byte
                # boundary: pad with zero fragments (72 B each) until it ends at 64 mod 128 (the slot's bytes below
                # the pads have all been read by now)
                e(f"  {p}zc = acc.clear.f32x64")
                frags = 2 * NP   # column pairs: the second of two streams
                for z in range(next(n for n in range(16) if 72 * (frags + n) % 128 == 64)):
                    nf += 1
                    nxt = (f"{p}xf{nf}", f"{p}xp{nf}", f"{p}xq{nf}")
                    e(f"  {nxt[0]}, {nxt[1]}, {nxt[2]} = vst.push.bfp16ebs8.from.fp32 {fifo[0]}, {p}zc, {fifo[1]}, {fifo[2]}")
                    fifo = nxt
                e(f"  {p}yf, {p}yp, {p}yq = vst.flush.512 {fifo[0]}, {fifo[1]}, {fifo[2]}")
            e(f"  {p}k1 = add.rr {p}k, %one")
            e(f"  low.br ^swl{t}({p}k1: {ER})")
            e(f"^swx{t}:")
            e(f"  {p}rb = mov.i32 {back + 512 + SW_SLOT * (segs - 1)}")
            e(f"  {p}ra = add.rr {p}ds, {p}rb")
            e(f"  %cvr{t} = mov.scalar-to-address {p}ra")
            e(f"  %cvq{t} = copy %cvr{t} : reg<aie2p.ep> -> reg<aie2p.ep>")
            e(f"  low.br ^cj{t}(%cvr{t}: reg<aie2p.ep>, %cvq{t}: reg<aie2p.ep>)")
            e(f"^cj{t}(%cj{t}p: reg<aie2p.ep>, %cj{t}l: reg<aie2p.ep>):")
            for j in range(segs):
                e(f"  %cqr{t}_{j} = copy %clast : reg<aie2p.er> -> reg<aie2p.mr26_lock>")
                e(f"  rel.cond %one, %cqr{t}_{j}, 2")
            return f"%cj{t}p", f"%cj{t}l"
        if cfg.swiglu:
            assert not cfg.ofeat and not groups(cfg), "swiglu needs the full sum: one replay group, bf16-slot C"
            convert = convert_swiglu
        if cfg.ofeat:
            convert = convert_f32
        if row_end and not nxt:
            if cfg.swiglu:   # the iteration's segments (mu), the earlier ones deferred to here
                pc[t], pl[t] = convert(pc[t], pl[t], 1024 * (NP - 1), segs=NT // NP)
            else:
                pc[t], pl[t] = convert(pc[t], pl[t], 1024 * (NP - 1))
        if nxt:
            e(f"  %pcc{t + 1} = copy {pc[t]} : reg<aie2p.ep> -> reg<aie2p.ep>")
            if cfg.swiglu and row_end:
                e(f"  %pc{t + 1}g = padds.modifier %pcc{t + 1}, %mcs")
                e(f"  %pc{t + 1} = padds.modifier %pc{t + 1}g, %mcb")
            else:
                e(f"  %pc{t + 1} = padds.modifier %pcc{t + 1}, %mcs")
            if late:   # from the next store pointer, so this one is dead
                if not cfg.swiglu:   # swiglu: deferred to the iteration end
                    pc[t + 1], pl[t + 1] = convert(pc[t + 1], pl[t + 1], 1024 * NP)
                for c in range(4):
                    for q in range(4):
                        e(f"  %l{t + 1}_{c}_{q} = vlda.acc {pl[t + 1]}, {256 * c + 64 * q - 512}")
            for c in range(4):
                e(f"  %z{t + 1}_{c} = concat(%l{t + 1}_{c}_0, %l{t + 1}_{c}_1, %l{t + 1}_{c}_2, %l{t + 1}_{c}_3) : {MBMS4}")
                cur[c] = f"%z{t + 1}_{c}"
    e("  rel %one, 0")
    steps = cfg.mu if a_adv else 1
    if not a_adv and (cfg.mu, acap) == (1, 2):   # alternate slots: step = fstep (1 - 2 (mp & 1))
        e("  %kh = mova.i32 -1")
        e("  %ph = lshl %mp, %kh")
        e("  %ph2 = add.rr %ph, %ph")
        e("  %par = sub %mp, %ph2")
        e("  %fst2 = add.rr %fstep, %fstep")
        e("  %pst0 = mul %par, %fst2")
        e("  %pst = sub %fstep, %pst0")
    if spa and not a_adv and (cfg.mu, acap) == (1, 2):
        e("  %pan = add.rr %pa, %pst")
    elif not a_adv and (cfg.mu, acap) == (1, 2):
        e("  %pac = copy %pa : reg<aie2p.ep> -> reg<aie2p.ep>")
        e("  %pmod = mov.modifier %pst")
        e("  %pan = padds.modifier %pac, %pmod")
    elif spa:   # scalar A pointer: advance by steps slab strides
        e(f"  %kpst = mova.i32 {steps if a_adv else 0}")
        e("  %pastep = mul %fstep, %kpst")
        e("  %pan = add.rr %pa, %pastep")
    else:
        e("  %pac = copy %pa : reg<aie2p.ep> -> reg<aie2p.ep>")
        e(f"  %pan0 = padds.modifier %pac, {'%mf' if a_adv else '%mz'}")
        for j in range(1, steps):
            e(f"  %pan{j}c = copy %pan{j - 1} : reg<aie2p.ep> -> reg<aie2p.ep>")
            e(f"  %pan{j} = padds.modifier %pan{j}c, %mf")
        e(f"  %pan = copy %pan{steps - 1} : reg<aie2p.ep> -> reg<aie2p.ep>")
    if tail:
        e(f"  %pcl = copy {pc[-1]} : reg<aie2p.ep> -> reg<aie2p.ep>")
        if cfg.swiglu:
            e("  %pong = padds.modifier %pcl, %mcs")
            e("  %pon = padds.modifier %pong, %mcb")
        else:
            e("  %pon = padds.modifier %pcl, %mcs")
        e(f"  %plx = copy {pl[-1]} : reg<aie2p.ep> -> reg<aie2p.ep>")
        if cfg.swiglu:
            e("  %plng = padds.modifier %plx, %mls")
            e("  %pln = padds.modifier %plng, %mlb")
        else:
            e("  %pln = padds.modifier %plx, %mls")
    else:
        e("  %pon = copy %po : reg<aie2p.er> -> reg<aie2p.er>")
    e("  %mpn = add.rr %mp, %one")
    pln = ", %pln: reg<aie2p.ep>" if tail else ""
    e(f"  low.br ^outer(%mpn: reg<aie2p.er>, %pan: {pat}, %pon: {pct}, {fifo['a']}: reg<aie2p.eldfiforeg>, {fifo['w']}: reg<aie2p.eldfiforeg>{pln})")
    e("^exit:")
    if cfg.fuse and not tail:
        e("  rel %one, 1")
    if gate:
        gate_epilogue(L, cfg, gate)
        e("}\n")
    else:
        e("  return\n}\n")


def gen(cfg):
    assert len(cfg.ks) == 4 and cfg.passes >= 2 and 1 <= cfg.cols <= 8
    L = []
    array_program(L, cfg)
    leaf(L, cfg, "head", cfg.ks[0])
    assert len(set(cfg.ks[1:-1])) == 1, "the mids share one leaf"
    leaf(L, cfg, "mid", cfg.ks[1])
    if cfg.fuse and len(cfg.ks) > 3:
        leaf(L, cfg, "mid", cfg.ks[-2], frow=len(cfg.ks) - 2)
    leaf(L, cfg, "tail", cfg.ks[-1], pair="recv" if cfg.ufmt else None)
    if cfg.ufmt:
        leaf(L, cfg, "tail", cfg.ks[-1], pair="send")
    if cfg.dcol:
        for n, fmt in sorted(set(fill_map(cfg)[1])):
            if cfg.dcol == 2:
                key = (fmt, cfg.gord, n, fill_name(cfg, n, fmt))
                if key not in _LEAVES:   # scheduled in Python (gen_npu_dec, npu_lsched): once per process
                    _LEAVES[key] = decoder(fmt, cfg.gord).leaf(fill_name(cfg, n, fmt), n)
                L.append(_LEAVES[key])
                L.append("")
            else:
                fill_leaf(L, cfg, n)
    if cfg.gate:
        leaf(L, cfg, "head", cfg.ks[0], gate="waiter")
        leaf(L, cfg, "mid", cfg.ks[1], gate="relay")
    return "\n".join(L)


def main():
    if len(sys.argv) < 5:
        sys.exit("usage: gen_npu_gemm.py <cols> <m_blocks> <ks,ks,ks,ks> <passes> [entry]")
    cfg = Config(int(sys.argv[1]), int(sys.argv[2]), tuple(int(v) for v in sys.argv[3].split(",")), int(sys.argv[4]),
                 entry=sys.argv[5] if len(sys.argv) > 5 else "npu_gemm", gate=int(sys.argv[6]) if len(sys.argv) > 6 else 0)
    sys.stdout.write(gen(cfg))


if __name__ == "__main__":
    main()
