#!/usr/bin/env python3
"""NPU weight decoders for the fill column of gen_npu_gemm (dcol = 2), one per GGUF format (DQ_FMT: IQ4_XS, Q4_K, IQ3_S,
IQ3_XXS, IQ2_XXS, IQ2_XS, Q3_K; the module's constants follow DQ_FMT at import, gen_npu_gemm.decoder reloads it per format).
Standalone check: DQ_FMT=<fmt> [DQ_PORTS=2] gen_npu_dec.py <work> <model.gguf> (decodes 80 rows per port of the
first matching K = 5120 tensor and compares every fragment with the oracle, bit for bit).
The IQ4_XS decoder (16-row panel units):
Input record (one panel unit): [16 rows][4 super-blocks x 136 B] (the (pass, slab) unit read from the row-major GGUF
tensor by an iterated shim view). Output: 4 leaf-synchronized records of 64 bfp16 fragments (super-block j of the
unit), in panel order [kb 32][h 2] (h = rows 0-7 / 8-15).
Per super-block: both halves are prepped (header words; qs rows realigned by vshift and transposed to [chunk][row][8 B];
the IQ4_XS scale path into the half's TB), then 8 sub-block iterations of 4 chunk-halves (chunks 2 s, 2 s + 1 x h):
the low-nibble fragments are pushed in order, the high-nibble accumulators parked and pushed after them.
"""
import os, subprocess, sys
os.environ.setdefault("LS_LAT", "vshuffle=3")     # the packer's shuffle spacing (measured: 520 -> 505 bundles)
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import hrx_paths
import npu_dec_emit as Q
I = Q
from npu_dec_emit import Emit, V1, V2, V4, M1, M2, M4
TABLES = os.path.join(HERE, "..", "tables")   # the GPU's grid tables (u8 [entry][4] grid values)


def grid_table(name, n):
    return np.fromfile(os.path.join(TABLES, name), np.uint8).reshape(n, 4).astype(np.float64)

FMT = os.environ.get("DQ_FMT", "IQ4_XS")
BLK = {"IQ4_XS": 136, "Q4_K": 144, "IQ3_S": 110, "IQ3_XXS": 98, "IQ2_XXS": 66, "IQ2_XS": 74,
       "Q3_K": 110}[FMT]   # bytes per super-block
HBY = {"IQ4_XS": 8, "Q4_K": 16, "IQ3_S": 2, "IQ3_XXS": 2, "IQ2_XXS": 2, "IQ2_XS": 2,
       "Q3_K": 0}[FMT]        # header bytes before qs
GRID = FMT in ("IQ3_S", "IQ3_XXS", "IQ2_XXS", "IQ2_XS")   # grid formats: gathers of 4-value grid pieces + sign masks (rodata)
GP4 = os.environ.get("DQ_GORD", "P4") == "P4"   # grid formats push in IQ4_XS's k-block order (0, 2, 1, 3) per sub-block
gpair = (lambda kb: 2 * (kb // 4) + kb % 2) if GP4 else (lambda kb: kb // 2)   # hot-loop pair of a k-block
gpos = (lambda kb: (kb % 4) // 2) if GP4 else (lambda kb: kb % 2)               # its position in the pair
XXS = FMT in ("IQ3_XXS", "IQ2_XXS")   # sign fields and scales in the aux words (7-bit ksigns indices, s = aux >> 28)
# IQ2_XXS: per sub-block [4 grid index bytes | aux word]; an 8-value grid entry is two 4-value pieces (the halves of a
# table line, like the signs: one address per row, duplicated, the odd lane reads the second piece). The index and aux
# words are the even / odd 4-byte units of the realigned rows (shuffle modes 4 / 5); an index byte becomes an address
# the way a sign field does. Otherwise IQ3_XXS: T = d (2 s + 1), the table holds grid / 8 (w = d (0.5 + s) grid / 4).
X2 = FMT == "IQ2_XXS"
# IQ2_XS: per k-block a u16 (9-bit grid index | 7-bit sign field << 9; 512 entries: 8 KB table, the page added),
# a scale nibble per 16 values (8 bytes at +66): T = d (2 ls + 1) per 16 values, Q3_K's two chains (NPARTS 4, record c
# = sub-blocks 4 c .. 4 c + 3) with ls' = 33 + 2 ls; the hot loop takes T per k-block (P4 pairs span two groups)
XS2 = FMT == "IQ2_XS"
# Q3_K: no gathers. Per record (128 values: qs bits of half n, hmask bit 4 n + j) both halves' rows of [hmask 32 B |
# qs 32 B] are transposed to [chunk][row][8 B] and nibble-unpacked to scratch (NIB); a fragment (k-block 4 j + c) is
# then (qs nibble & crumb mask) | (hmask nibble & 1 << j) doubled into place, interleaved with a high byte 0x43 (j even:
# bf16 128 + v) or 0x42 (j odd, crumbs and bit 2 positions up: 32 + v), v = crumb + 4 hbit; w = T (X - K), K = 132 / 36,
# as C = -K T0 - K T1, + T0 X, + T1 X: every step exact (T = d (sc - 32) has 16 significant bits), so w is exactly
# d (sc - 32) (v - 4) in f32. T: IQ4_XS's chain (the same formula) for sub-blocks 0-7 and 8-15 (one per record).
Q3K = FMT == "Q3_K"
GORD = os.environ.get("DQ_GORD", "P4")       # Q3_K pushes any k-block order: "P4", "N" or "PK" (Q4_K's, for its pairs)
RECROW = 576                    # record row bytes, the same for every format (Q4_K's 4 x 144): image switches between
                                # formats then re-program only the decoder tiles (the staging buffers stay identical)
ROWB = RECROW if os.environ.get("DQ_FIXREC", "1") == "1" else 4 * BLK   # a row's 4 super-blocks (+ ignored tail)
IN_B, NF = 16 * ROWB, 64
OUT_B = 32 * 72                 # one output record (half a super-block: 4 sub-blocks)
GT, HR = 0, 2048                # table; half regions at HR + h HS
HDR, SCM, HB, TAB, PARTS, TB, QT = 0, 64, 192, 256, 384, 896, 2944
HS = 512                       # half regions now hold only the scale-path temporaries (HDR .. PARTS)
STATE, KVS = HR + 2 * HS, HR + 2 * HS + 64
SCR_B = KVS + 64                # storage A (2048-aligned: the table)
GAS, PARTSS, GT2 = 0, 4096, 5120
SCR2_B = GT2 + 2048 + 64        # storage B (gather addresses, T parts, the table's second copy; +64: gathers read a + 32)
TABV, BASEV = [TAB], ["%sp"]
KV = I.KV if FMT == "IQ4_XS" else [float(v) for v in range(16)]
OUTCAP = 2
INLINE = [0]                    # leaf_inline: super-blocks per input record (0: a standalone leaf)
TC_IQ4 = (("su", 0x380), ("sv", 0), ("u", 0xC6A0), ("v", 0xC320), ("s", 0xC681), ("1", 0x4A01), ("1", 0x4901))
TC_Q4K = (("su", 0x380), ("sv", 0), ("u", 0xC680), ("v", 0xC300), ("s", 0xC681), ("1", 0x4A01))   # T = d sc
NPARTS = 4 if FMT in ("Q4_K", "Q3_K", "IQ2_XS") else 2   # bf16 parts per scale line (T hi / lo; Q4_K adds -M hi / lo, Q3_K
                                               # holds 8 sub-blocks: lines (f, p) = 2 f + p)
NIB = 0                         # Q3_K: storage B nibble rows per half: hmask nibble n of chunk c at 64 c, qs lo / hi at
                                # 64 (4 + c) / 64 (8 + c); 768 B per half (the gather addresses' region, unused)
KVT = list(range(16)) if FMT == "Q4_K" else None    # the gather table's values (IQ4_XS: its kvalues)
CTOFF = 0                       # splat-constant table in storage A (DQ_CT; 0 = timing ablation: the table area)
DUMP = False


def bf16_bits_np(v):
    return (np.asarray(v, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def grid_rodata(name):
    """IQ3_S tables (4096-aligned rodata; a gather reads an 8-byte piece at (L & ~63) + (L & 63) / 2, +32 on odd lanes):
    grid 512 x [4 bf16] (both block halves), signs 256 x (masks of bits 0-3 | bits 4-7 in the two halves),
    qh bits 256 x [8 bytes 0 / 1] (both halves)."""
    hexs = lambda a: a.view(np.uint8).tobytes().hex()
    if XXS or XS2:
        # IQ3_XXS: grid / 4 (T = d (2 s + 1)), 256 entries; signs by the 7-bit field f: ksigns(f) = f | parity(f) << 7
        gt = np.zeros((128 if XS2 else 64, 2, 4, 4), np.uint16)
        if XS2:                                     # IQ2_XS: as IQ2_XXS, 512 entries
            g8 = (np.fromfile(os.path.join(TABLES, "grid_iq2xs.bin"), np.uint8).reshape(512, 8) / 8).astype(np.float32)
            for idx in range(512):
                for h_ in range(2):
                    gt[idx >> 2, h_, idx & 3] = bf16_bits_np(g8[idx, 4 * h_:4 * h_ + 4])
        elif X2:                                      # IQ2_XXS: grid / 8, entry idx's values 4 h .. 4 h + 3 in half h
            g8 = (np.fromfile(os.path.join(TABLES, "grid_iq2xxs.bin"), np.uint8).reshape(256, 8) / 8).astype(np.float32)
            for idx in range(256):
                for h_ in range(2):
                    gt[idx >> 2, h_, idx & 3] = bf16_bits_np(g8[idx, 4 * h_:4 * h_ + 4])
        else:
            grid = (grid_table("grid_iq3xxs.bin", 256) / 4).astype(np.float32)
            for idx in range(256):
                gt[idx >> 2, :, idx & 3] = bf16_bits_np(grid[idx])
        st = np.zeros((32, 2, 4, 4), np.uint16)
        for f in range(128):
            ks = f | ((bin(f).count("1") & 1) << 7)
            for j in range(8):
                st[f >> 2, j // 4, f & 3, j % 4] = 0x8000 if (ks >> j) & 1 else 0
        # one object, the sign table on the page after the grid's (its 2 KB alone would take a whole aligned page)
        return [f'global.rodata.def @gt_{name} = align(4096) bytes("{hexs(gt)}{hexs(st)}")']
    # IQ3_S: the grid's 2 pages, then (page 2) the signs by nibble n (even lanes: the low nibble -> values 0-3, odd
    # lanes: the high nibble -> values 4-7: both halves alike) and the qh bit masks [kb 4][row 8][e 2] = 1 << (2 kb + e)
    grid = grid_table("grid_iq3s.bin", 512).astype(np.float32)
    gt = np.zeros((128, 2, 4, 4), np.uint16)
    for idx in range(512):
        gt[idx >> 2, :, idx & 3] = bf16_bits_np(grid[idx])
    st = np.zeros((4, 2, 4, 4), np.uint16)
    for n in range(16):
        for v in range(4):
            st[n >> 2, :, n & 3, v] = 0x8000 if (n >> v) & 1 else 0
    hm = np.array([1 << (2 * (i // 16) + i % 2) for i in range(64)], np.uint8)
    return [f'global.rodata.def @gt_{name} = align(4096) bytes("{hexs(gt)}{hexs(st)}{hexs(hm)}")']


def leaf(name="dec", nports=1):
    e = Emit()
    e.L.append(f"low.func.def schedule(locked) target<amd.xdna.aie2p.core>(@core_target) abi(object_function) @{name}() asm {{")

    for pp_ in range(nports):
        sfx = "" if pp_ == 0 else str(pp_)
        e(f"%in{sfx} = resource<native_pointer> {{index = {pp_}, source_type = buffer}} : reg<aie2p.ep>")
    for pp_ in range(nports):
        sfx = "" if pp_ == 0 else str(pp_)
        e(f"%out{sfx} = resource<native_pointer> {{index = {nports + pp_}, source_type = buffer}} : reg<aie2p.ep>")
    pad_a = int(os.environ.get("DQ_PADA", "0"))      # standalone tests: push storage A out of storage B's bank
    e(f"%scr = storage {{byte_alignment = 64, byte_length = {SCR_B + pad_a}}} : low.storage<private>")
    # storage B in its own space: one region per space, so the gathers (table in A) and the address / T-part loads (B)
    # can sit in different banks (both in one bank: a conflict in nearly every hot-loop bundle)
    sp2 = os.environ.get("DQ_SPACE2", "scratch")
    e(f"%scr2 = storage {{byte_alignment = 64, byte_length = {SCR2_B}}} : low.storage<{sp2}>")
    e(f"%sp2 = storage_address %scr2 : low.storage<{sp2}> -> reg<aie2p.ep>")
    e("%sp = storage_address %scr : low.storage<private> -> reg<aie2p.ep>")
    e("set.unpack-size 0")
    e("set.rounding 12")
    hdr_end = len(e.L)
    k = e.const
    conf = k(60)
    m0c, m1c, m20, m21, m52, m53 = k(0), k(1), k(20), k(21), k(52), k(53)
    m2c, m3c, m4c, m18, m19 = k(2), k(3), k(4), k(18), k(19)
    # ---- helpers (x: the stream to emit into)
    CTV = [0x380, 0, 0xC6A0, 0xC320, 0xC681, 0x4A01, 0x4901, 0x3F80, 0xBF80, 0x4300, 128, 3200, 1024, 0x7C, 0x8000]
    CTP = {}                                            # stream id -> pointer at CT + 512

    def splat16(x, v):
        r_ = x.t("sp")
        if CTP.get("on") and v in CTV:
            x(f"{r_} = vlda.512.i8x64 %ctp, {64 * CTV.index(v) - 512}")
            return r_
        x(f"{r_} = vbcst.16 {k(v)}")
        return r_

    def wide(x, x2):
        """vec256 x2 -> x4 by repeating."""
        c, r4 = x.t("cp"), x.t("w4")
        x(f"{c} = vmov.512 {x2}")
        x(f"{r4} = concat({x2}, {c}) : ({V2}, {V2}) -> {V4}")
        return r4

    def col16(x, off):
        """The [row][f] 16-lane int16 table at TAB + off repeated to 32 lanes (vec256 x2)."""
        l0, l1, r2 = x.t("c"), x.t("c"), x.t("c2")
        pp, oo = x.addr(BASEV[0], TABV[0] + off, 224, 32, "sp_tab" + BASEV[0])
        x(f"{l0} = vlda.256.i16x16 {pp}, {oo}")
        x(f"{l1} = vlda.256.i16x16 {pp}, {oo}")
        x(f"{r2} = concat({l0}, {l1}) : ({V1}, {V1}) -> {V2}")
        return r2

    def add16(x, a_, b_):
        r_ = x.t("ad")
        x(f"{r_} = vadd.16 {a_}, {b_}")
        return r_

    def sub16(x, a_, b_):
        r_ = x.t("sb")
        x(f"{r_} = vsub.16 {a_}, {b_}")
        return r_

    def band(x, a_, v):
        r_ = x.t("an")
        x(f"{r_} = vband {a_}, {splat16(x, v)}")
        return r_

    def mac(x, acc, s1, s2):
        r_ = x.t("ac")
        if acc is None:
            x(f"{r_} = mmul.bf16bf16.m8n8k1 {s1}, {s2}, {conf}")
        else:
            x(f"{r_} = mma.bf16bf16.m8n8k1 {acc}, {s1}, {s2}, {conf}")
        return r_

    def conv(x, acc):
        """f32 x64 -> bf16 x64 (round to nearest even): (x4, (half0, half1))."""
        q = [x.t("q") for _ in range(4)]
        for i in range(4):
            x(f"{q[i]} = slice {acc}[{i}] : {M4} -> {M1}")
        h0, h1, c0, c1, r4 = x.t("h"), x.t("h"), x.t("cv"), x.t("cv"), x.t("cb")
        x(f"{h0} = concat({q[0]}, {q[1]}) : ({M1}, {M1}) -> {M2}")
        x(f"{h1} = concat({q[2]}, {q[3]}) : ({M1}, {M1}) -> {M2}")
        x(f"{c0} = vconv.bf16.fp32 {h0}")
        x(f"{c1} = vconv.bf16.fp32 {h1}")
        x(f"{r4} = concat({c0}, {c1}) : ({V2}, {V2}) -> {V4}")
        return r4, (c0, c1)

    def shr8(x, a_):
        o, r_ = x.t("o"), x.t("r8")
        x(f"{o} = vshuffle {a_}, {a_}, {m1c}")
        x(f"{r_} = vshuffle {o}, %z8, {m20}")
        return r_

    RB = {}
    win = {}

    def window(x, base, at, name):
        """A pointer to base + at, defined by the first stream that needs it (the scheduler orders users after it)."""
        if (name, at) not in win:
            cur = x.t("pw")
            x(f"{cur} = copy {base} : reg<aie2p.ep> -> reg<aie2p.ep>")
            left = at
            while left:
                st_ = min(448, left) if left % 64 == 0 or left > 448 else left
                if st_ % 64:
                    md, nxt = x.t("md"), x.t("pw")
                    x(f"{md} = mov.modifier {k(st_)}")
                    x(f"{nxt} = padda.modifier {cur}, {md}")
                else:
                    nxt = x.t("pw")
                    x(f"{nxt} = padda {cur}, {st_}")
                cur, left = nxt, left - st_
            win[(name, at)] = (cur, at)
        return win[(name, at)]

    import npu_lsched as lsched
    WIN = int(os.environ.get("DQ_WIN", "4"))

    def run(streams, window_=None, tails=None, ext=(0, 0, 0), hv=None):
        if os.environ.get("DQ_LS", "1") == "1":
            e.L += lsched.schedule([x.L for x in streams], window_ or WIN, tails, ext=ext,
                                   hard=(hv or int(os.environ.get("DQ_HV", "24")), 20, int(os.environ.get("DQ_PH", "2" if GRID else "3"))))
            if os.environ.get("DQ_PRED"):
                print(f"pred {len(streams)} streams: {lsched.schedule.cycles} cycles", file=sys.stderr)
        else:
            e.zip(streams)

    def rel(fp, off, hi=448, step=64):
        o = off - fp[1]
        assert -(hi + step) <= o <= hi and o % step == 0, (off, fp)
        return fp[0], o

    def rows(x, s, which, p, fp=None, before=None):
        """Row-broadcast bf16 x64 of part p of T (which 0) or M (which 1) for sub-block s (fp: shared pointer);
        before(ra, rb) runs on the two halves ahead of the concat."""
        blk, sl = s // 4, s % 4
        off = PARTS + (((3 * blk + p) * 2 + which) * 2 + sl // 2) * 64
        pp, o = rel(fp, off) if fp else x.addr("%sp", off, cursor="sp_parts_r")
        ld, bc, ra, rb, r4 = x.t("pl"), x.t("bc"), x.t("ra"), x.t("rb"), x.t("tb")
        x(f"{ld} = vlda.512.bf16x32 {pp}, {o}")
        x(f"{bc} = vbroadcast.bf16x8.to.bf16x32 {ld}, {sl % 2}")
        x(f"{ra} = vshuffle {bc}, {bc}, {m52}")
        x(f"{rb} = vshuffle {bc}, {bc}, {m53}")
        if before:
            before(ra, rb)
        x(f"{r4} = concat({ra}, {rb}) : ({V2}, {V2}) -> {V4}")
        return r4


    for pp_ in range(nports):
        sfx = "" if pp_ == 0 else str(pp_)
        e(f"%ina{sfx} = mov.address-to-scalar %in{sfx}")
        e(f"%outa{sfx} = mov.address-to-scalar %out{sfx}")
    e("%spa = mov.address-to-scalar %sp")
    e("%spb = mov.address-to-scalar %sp2")
    e(f"%cgt = lshl %spa, {k(-3)}")                  # (GT = 0) >> 3: u16 b + it, << 3 = GT + 8 b (GT < 0x7F800)
    e("%sh3 = mov.shift 3")

    e("%z8 = vbcst.8 %k0z")
    e.pre.append("  %k0z = mova.i32 0")
    e.consts["%k0z"] = True

    def P(x, base_scalar, off):
        a_, p_ = x.t("pa"), x.t("pm")
        x(f"{a_} = add.rr {base_scalar}, {k(off)}" if off else f"{a_} = or {base_scalar}, {k(0)}")
        x(f"{p_} = mov.scalar-to-address {a_}")
        return p_

    def Pr(x, scalar):
        p_ = x.t("pm")
        x(f"{p_} = mov.scalar-to-address {scalar}")
        return p_

    lab = [0]

    def label(nm):
        lab[0] += 1
        return f"{nm}{lab[0]}"

    if GRID:
        # rodata tables: their 4 KB pages as address high bytes (address = page << 12 | 16 index via vups << 4)
        e(f"%gtp = mov.local-address @gt_{name}")
        e(f"%gta = mov.address-to-scalar %gtp")
        e(f"%gtc = lshl %gta, {k(-12)}")
        # the sign table: the pages after the grid's (grid_rodata)
        e(f"%stc = add.rr %gtc, {k(1 if XXS else 2)}")
        if not (XXS or XS2):
            e(f"%hma = add.rr %gta, {k(8192 + 256)}")      # IQ3_S: the qh bit masks
    elif not Q3K:
        # ---- table (once per invocation of the program: a flag in private storage)
        tb_, tdone = label("tbuild"), label("tdone")
        if INLINE[0]:                                        # the storages are scratch: build the table every time
            e(f"low.br ^{tb_}")
        else:
            stp = P(e, "%spa", STATE)
            e(f"%tflag = lda {stp}, 0")
            e(f"%tx = xor %tflag, {k(0x5A5A1234)}")          # private storage starts undefined: a magic marks the table
            e(f"low.cond_br %tx, ^{tb_}, ^{tdone} : reg<aie2p.er>")
        e.L.append(f"^{tb_}:")
        for i in range(8):
            w_ = (I.bf16_bits(KV[2 * i + 1]) << 16) | I.bf16_bits(KV[2 * i])
            pp = P(e, "%spa", KVS + 4 * i)
            e(f"%kw{i} = mov.i32 {w_ - (1 << 32) if w_ >= 1 << 31 else w_}")
            e(f"st %kw{i}, {pp}, 0")
        pk = P(e, "%spa", KVS)
        e(f"%kvv = vlda.512.bf16x32 {pk}, 0")

        def tline(m, kb, pa_, pb_):
            sp_, r_, h0, h1 = e.t("ks"), e.t("kr"), e.t("kh"), e.t("kh")
            e(f"{sp_} = vbcst.16 {kb}")
            e(f"{r_} = vshuffle %kvv, {sp_}, {m18}")
            e(f"{h0} = slice {r_}[0] : {V2} -> {V1}")
            e(f"{h1} = slice {r_}[1] : {V2} -> {V1}")
            # two copies in different banks (storages A and B): a gather (vldb.4x32) costs one stall cycle unless its
            # address lanes 0 / 2 and 1 / 3 hit different banks (measured, gbench.py) -> lanes 4 q + 2, 4 q + 3 use copy 2
            for pg in (pa_, pb_) if m is None else (P(e, b_, o_ + 128 * m) for b_, o_ in (("%spa", GT), ("%spb", GT2))):
                for blk, hv_ in ((0, h0), (1, h1)):
                    for half in range(2):
                        e(f"vst.256.i16x16 {hv_}, {pg}, {64 * blk + 32 * half}")
        if INLINE[0] or os.environ.get("DQ_TROLL") == "1":   # rolled (program memory; DQ_TROLL=1: standalone test)
            e(f"%tla0 = add.rr %spa, {k(GT)}")
            e(f"%tlb0 = add.rr %spb, {k(GT2)}")
            e(f"low.br ^tlh({k(0)}: reg<aie2p.er>, %tla0: reg<aie2p.er>, %tlb0: reg<aie2p.er>)")
            e.L.append("^tlh(%tl_i: reg<aie2p.er>, %tla: reg<aie2p.er>, %tlb: reg<aie2p.er>):")
            e(f"%tl_m = lt %tl_i, {k(16)}")
            e("low.cond_br %tl_m, ^tlb, ^tlx : reg<aie2p.er>")
            e.L.append("^tlb:")
            e.L.append("// LOOP-BEGIN")
            e("%tlkb = vextract.16.reg %kvv, %tl_i")       # (not a load: the KV words were just stored)
            e("%tlpa = mov.scalar-to-address %tla")
            e("%tlpb = mov.scalar-to-address %tlb")
            tline(None, "%tlkb", "%tlpa", "%tlpb")
            e(f"%tla1 = add.rr %tla, {k(128)}")
            e(f"%tlb1 = add.rr %tlb, {k(128)}")
            e("%tl_n = add %tl_i, 1")
            e("low.br ^tlh(%tl_n: reg<aie2p.er>, %tla1: reg<aie2p.er>, %tlb1: reg<aie2p.er>)")
            e.L.append("// LOOP-END")
            e.L.append("^tlx:")
        else:
            for m in range(16):
                e(f"%kb{m} = mov.i32 {I.bf16_bits(KV[m])}")
                tline(m, f"%kb{m}", None, None)
            stp2 = P(e, "%spa", STATE)
            e(f"st {k(0x5A5A1234)}, {stp2}, 0")
        e(f"low.br ^{tdone}")
        e.L.append(f"^{tdone}:")

    # ---- loop helpers
    def loop_begin(nm, n, args=()):
        """args: [(init, type)]; returns the loop counter and the block arg names."""
        z = k(0)
        names = [f"%{nm}_a{i}" for i in range(len(args))]
        ext = "".join(f", {v}: {t}" for v, t in args)
        e(f"low.br ^{nm}h({z}: reg<aie2p.er>{ext})")
        e.L.append(f"^{nm}h(%{nm}_i: reg<aie2p.er>" + "".join(f", {n_}: {t}" for n_, (v, t) in zip(names, args)) + "):")
        e(f"%{nm}_m = lt %{nm}_i, {k(n)}")
        e(f"low.cond_br %{nm}_m, ^{nm}b, ^{nm}x : reg<aie2p.er>")
        e.L.append(f"^{nm}b:")
        e.L.append("// LOOP-BEGIN")
        return f"%{nm}_i", names

    def loop_end(nm, nxt=()):
        e(f"%{nm}_n = add %{nm}_i, 1")
        e(f"low.br ^{nm}h(%{nm}_n: reg<aie2p.er>" + "".join(f", {v}: {t}" for v, t in nxt) + ")")
        e.L.append("// LOOP-END")
        e.L.append(f"^{nm}x:")


    def aux_words(x, base_scalar, row0, sel=None):
        """8 rows' aux words (realigned 64 B at base + 576 row) transposed: O[i] = words [2 i, 2 i + 1][row 0..7]
        (zips 16, 14 / 15, 12 / 13 on 4-byte units). sel: a shuffle mode applied to each row first (IQ2_XXS: 4 / 5,
        its even / odd 4-byte units)."""
        sh_, al_ = x.t("ash"), x.t("aal")
        x(f"{sh_} = and {base_scalar}, {k(63)}")
        x(f"{al_} = and {base_scalar}, {k(-64)}")
        R = []
        for r in range(8):
            pa_ = x.t("pa")
            x(f"{pa_} = add.rr {al_}, {k(ROWB * (row0 + r))}" if row0 + r else f"{pa_} = or {al_}, {k(0)}")
            pv = Pr(x, pa_)
            va, vb, q = x.t("va"), x.t("vb"), x.t("q")
            x(f"{va} = vlda.512.i8x64 {pv}, 0")
            x(f"{vb} = vlda.512.i8x64 {pv}, 64")
            x(f"{q} = vshift {va}, {vb}, {sh_}")
            if sel is not None:
                q2_ = x.t("qs")
                x(f"{q2_} = vshuffle {q}, {q}, {k(sel)}")
                q = q2_
            R.append(q)
        P_ = []
        for q in range(4):
            v = x.t("ap")
            x(f"{v} = vshuffle {R[2 * q]}, {R[2 * q + 1]}, {k(16)}")
            P_.append(v)
        Qv = []
        for i_, j_ in ((0, 1), (2, 3)):
            for md in (k(14), k(15)):
                v = x.t("aq")
                x(f"{v} = vshuffle {P_[i_]}, {P_[j_]}, {md}")
                Qv.append(v)                              # (w0-3, r0-3), (w4-7, r0-3), (w0-3, r4-7), (w4-7, r4-7)
        O = []
        for i_, j_ in ((0, 2), (1, 3)):
            for md in (k(12), k(13)):
                v = x.t("ao")
                x(f"{v} = vshuffle {Qv[i_]}, {Qv[j_]}, {md}")
                O.append(v)
        return O

    def field(x, oa, ob, mask, shreg):
        """u16 lanes [word 4][row 8] of (w & mask) >> sh for the words of oa, ob (the accumulator path is signed)."""
        mv = x.t("mv")
        x(f"{mv} = vbcst.32 {k(mask - (1 << 32) if mask >= 1 << 31 else mask)}")
        accs = []
        for o in (oa, ob):
            an, ac = x.t("an"), x.t("xa")
            x(f"{an} = vband {o}, {mv}")
            x(f"{ac} = vmov.vector512.to.accumulator512 {an}")
            accs.append(ac)
        a2, r_ = x.t("x2"), x.t("fr")
        x(f"{a2} = concat({accs[0]}, {accs[1]}) : ({M1}, {M1}) -> {M2}")
        x(f"{r_} = vsrs.2x.c-to-x.unsigned {a2}, {shreg}")
        return r_

    def xxs_scales(x, h, b8):
        """IQ3_XXS: ls = 33 + 2 s (s = aux >> 28, one per sub-block) in the lane order [s'][row][f]."""
        ab = x.t("xab")
        x(f"{ab} = add.rr %hin_{h}, {k(2 if X2 else 66)}")
        O = aux_words(x, ab, 0, 5 if X2 else None)
        sh28 = x.t("s28")
        x(f"{sh28} = mov.shift 28")
        cs_ = []
        for g in range(2):                                # sub-blocks 4 g .. 4 g + 3
            sv = field(x, O[2 * g], O[2 * g + 1], 0xF0000000, sh28)
            sm, d2, c_ = x.t("sm"), x.t("d2"), x.t("lc")
            x(f"{sm} = vband {sv}, {splat16(x, 15)}")    # the signed shift: s - 16 for s >= 8
            x(f"{d2} = vadd.16 {sm}, {sm}")
            x(f"{c_} = vshuffle {d2}, {d2}, {m0c}")       # u16 -> low bytes [s'][row]
            cs_.append(c_)
        z, a_ = x.t("lz"), x.t("lsr")
        x(f"{z} = vshuffle {cs_[0]}, {cs_[1]}, {m20}")    # [s'][row][f]
        x(f"{a_} = vadd.8 {z}, {b8(33)}")
        return a_

    # ---- super-block loop (port 0; other ports are renamed copies of its section)
    PLOOP = nports > 1 and os.environ.get("DQ_PLOOP", "0" if FMT == "IQ4_XS" else "1") == "1"   # one code copy for all ports (16 KiB limit; IQ4_XS fits twice: +2% without)
    INA, OUTA = ["%ina"], ["%outa"]
    lk = [0]

    def lockop(op, val):
        """acq / rel on the output ring of the current port (immediate port index: a branch per port in PLOOP)."""
        if INLINE[0]:                                   # leaf_inline: the output is plain memory
            return
        if not PLOOP:
            e(f"{op} {val}, {nports}")
            return
        lk[0] += 1
        n_ = lk[0]
        e(f"%lkv{n_} = or {val}, {k(0)}")          # before the branch: both arms use it (constants land at first use)
        e(f"low.cond_br {PL[0]}, ^lk{n_}b, ^lk{n_}a : reg<aie2p.er>")
        e.L.append(f"^lk{n_}a:")
        e(f"{op} %lkv{n_}, {nports}")
        e(f"low.br ^lk{n_}j")
        e.L.append(f"^lk{n_}b:")
        e(f"{op} %lkv{n_}, {nports + 1}")
        e(f"low.br ^lk{n_}j")
        e.L.append(f"^lk{n_}j:")
    PL = [None]
    if PLOOP:
        assert nports == 2
        e("%pdi = sub %ina1, %ina")
        e("%pdo = sub %outa1, %outa")
        pl_, _ = loop_begin("pl", nports)
        PL[0] = pl_
        e(f"%plmi = mul {pl_}, %pdi")
        e(f"%plmo = mul {pl_}, %pdo")
        e("%pina = add.rr %ina, %plmi")
        e("%pouta = add.rr %outa, %plmo")
        INA[0], OUTA[0] = "%pina", "%pouta"
    sec0 = len(e.L)
    j, _ = loop_begin("jl", INLINE[0] or 4)
    e(f"%j136 = mul {j}, {k(BLK)}")
    e(f"%jin = add.rr {INA[0]}, %j136")
    # -- halves: straight-line, one stream each (the scale path is one long dependency chain per half: the two
    # chains interleave)
    for h in range(2):
        e(f"%hb0_{h} = add.rr %spa, {k(HR + h * HS)}")              # this half's region (scalar)
        e(f"%hin_{h} = add.rr %jin, {k(h * 8 * ROWB)}" if h else f"%hin_{h} = or %jin, {k(0)}")
        if FMT != "Q4_K" and not GRID and not Q3K:
            e(f"%hdpa_{h} = add.rr %hb0_{h}, {k(HDR + 32)}")
            e(f"%hdp_{h} = mov.scalar-to-address %hdpa_{h}")
    # header words -> HDR [row][8 B]
    rows_s = []
    for h in range(2):
        if GRID:                                # one run per half (pointer budget)
            e(f"%hdpa_{h} = add.rr %hb0_{h}, {k(HDR + 32)}")
            e(f"%hdp_{h} = mov.scalar-to-address %hdpa_{h}")
            if XS2:                             # the scale bytes -> SCM [row][8 B]
                e(f"%hsaa_{h} = add.rr %hb0_{h}, {k(SCM + 32)}")
                e(f"%hsa_{h} = mov.scalar-to-address %hsaa_{h}")
            rows_s = []
        if Q3K:                                 # one run per half: d words -> HDR [row][8 B], sc -> SCM [c][row][8 B]
            for nm_, off_ in (("hdp", HDR + 32), ("hsa", SCM + 32), ("hsb", SCM + 96)):
                e(f"%{nm_}a_{h} = add.rr %hb0_{h}, {k(off_)}")
                e(f"%{nm_}_{h} = mov.scalar-to-address %{nm_}a_{h}")
            rows_s = []
        if FMT == "Q4_K":                       # one run per half: its 2 store pointers live only there (pointer budget)
            e(f"%hdpa_{h} = add.rr %hb0_{h}, {k(HDR + 32)}")
            e(f"%hdp_{h} = mov.scalar-to-address %hdpa_{h}")
            e(f"%hdqa_{h} = add.rr %hb0_{h}, {k(HDR + 96)}")    # word planes 2, 3 (scalar st offsets -32 .. 28)
            e(f"%hdq_{h} = mov.scalar-to-address %hdqa_{h}")
            rows_s = []
        for r in range(8):
            x = e.stream()
            rows_s.append(x)
            ra, w0, w1 = x.t("ra"), x.t("w0"), x.t("w1")
            x(f"{ra} = add.rr %hin_{h}, {k(ROWB * r)}" if r else f"{ra} = or %hin_{h}, {k(0)}")
            pr = Pr(x, ra) if not Q3K else None
            if GRID:                                        # IQ3_S: d (+0) and the 4 scale bytes (+106), u16 loads
                rb_, pr2, dw, s0, s1, s1h, sw = (x.t(n) for n in ("rb", "pq", "dw", "s0", "s1", "s1h", "sw"))
                x(f"{dw} = lda.u16 {pr}, 0")                # d first: one pointer per stream at a time
                if XS2:                                     # d | 0 -> HDR, the 8 scale bytes (+66, u16 loads) -> SCM
                    x(f"st {dw}, %hdp_{h}, {8 * r - 32}")
                    x(f"st {k(0)}, %hdp_{h}, {8 * r - 28}")
                    x(f"{rb_} = add.rr {ra}, {k(66)}")
                    x(f"{pr2} = mov.scalar-to-address {rb_}")
                    hs_ = []
                    for i_ in range(4):
                        hs_.append(x.t("hs"))
                        x(f"{hs_[-1]} = lda.u16 {pr2}, {2 * i_}")
                    for i_ in range(2):
                        sh_, w_ = x.t("s1h"), x.t("sw")
                        x(f"{sh_} = lshl {hs_[2 * i_ + 1]}, {k(16)}")
                        x(f"{w_} = or {hs_[2 * i_]}, {sh_}")
                        x(f"st {w_}, %hsa_{h}, {8 * r + 4 * i_ - 32}")
                    if r == 7:
                        run(rows_s, int(os.environ.get("DQ_HDRWIN", "1")))
                    continue
                if XXS:
                    x(f"st {dw}, %hdp_{h}, {8 * r - 32}")
                    x(f"st {k(0)}, %hdp_{h}, {8 * r - 28}")
                    if r == 7:
                        run(rows_s, int(os.environ.get("DQ_HDRWIN", "2")))
                    continue
                x(f"{rb_} = add.rr {ra}, {k(106)}")
                x(f"{pr2} = mov.scalar-to-address {rb_}")
                x(f"{s0} = lda.u16 {pr2}, 0")
                x(f"{s1} = lda.u16 {pr2}, 2")
                x(f"{s1h} = lshl {s1}, {k(16)}")
                x(f"{sw} = or {s0}, {s1h}")
                x(f"st {dw}, %hdp_{h}, {8 * r - 32}")      # HDR [row]: d | 0, scales: the IQ4_XS layout with scales_h 0
                x(f"st {sw}, %hdp_{h}, {8 * r - 28}")
                if r == 7:
                    run(rows_s, int(os.environ.get("DQ_HDRWIN", "2")))
                continue
            if Q3K:   # d (+108) and the 12 scale bytes (+96) by u16 loads; sc (6 bit, 16) by the K-quant SWAR
                rb_ = x.t("rb")
                x(f"{rb_} = add.rr {ra}, {k(96)}")
                pr2 = Pr(x, rb_)
                hw_ = []
                for i_ in range(6):
                    hw_.append(x.t("hw"))
                    x(f"{hw_[-1]} = lda.u16 {pr2}, {2 * i_}")
                dw = x.t("dw")
                x(f"{dw} = lda.u16 {pr2}, 12")
                aux = []
                for i_ in range(3):
                    hs_, a_ = x.t("hs"), x.t("aux")
                    x(f"{hs_} = lshl {hw_[2 * i_ + 1]}, {k(16)}")
                    x(f"{a_} = or {hw_[2 * i_]}, {hs_}")
                    aux.append(a_)
                ws = []
                for i_, (src, lsh, hsh) in enumerate(((0, 0, 4), (1, 0, 2), (0, -4, 0), (1, -4, -2))):
                    lo_, hi_, h2_, w_ = x.t("lo"), x.t("hi"), x.t("h2"), x.t("sw")
                    if lsh:
                        sh_ = x.t("ls")
                        x(f"{sh_} = lshl {aux[src]}, {k(lsh)}")
                        x(f"{lo_} = and {sh_}, {k(0x0F0F0F0F)}")
                    else:
                        x(f"{lo_} = and {aux[src]}, {k(0x0F0F0F0F)}")
                    if hsh:
                        x(f"{hi_} = lshl {aux[2]}, {k(hsh)}")
                        x(f"{h2_} = and {hi_}, {k(0x30303030)}")
                    else:
                        x(f"{h2_} = and {aux[2]}, {k(0x30303030)}")
                    x(f"{w_} = or {lo_}, {h2_}")
                    ws.append(w_)
                x(f"st {dw}, %hdp_{h}, {8 * r - 32}")
                for i_, w_ in enumerate(ws):              # sub-blocks 4 i .. 4 i + 3: [c = i // 2][row][8 B]
                    x(f"st {w_}, %{'hsa' if i_ < 2 else 'hsb'}_{h}, {8 * r + 4 * (i_ % 2) - 32}")
                if r == 7:                                  # one row at a time: 3 store pointers (pointer budget)
                    run(rows_s, int(os.environ.get("DQ_HDRWIN", "1")))
                continue
            if FMT == "Q4_K":                               # 4 words -> word planes [k][row] (vector loads transposed)
                ws = [x.t("w") for _ in range(4)]
                for kk in range(4):
                    x(f"{ws[kk]} = lda {pr}, {4 * kk}")
                for kk in range(4):
                    x(f"st {ws[kk]}, {'%hdp_' if kk < 2 else '%hdq_'}{h}, {32 * (kk % 2) + 4 * r - 32}")
                if r == 7:
                    run(rows_s, int(os.environ.get("DQ_HDRWIN", "2")))
                continue
            x(f"{w0} = lda {pr}, 0")
            x(f"{w1} = lda {pr}, 4")
            x(f"st {w0}, %hdp_{h}, {8 * r - 32}")         # one shared store pointer per half at HDR + 32
            x(f"st {w1}, %hdp_{h}, {8 * r - 28}")
    if FMT != "Q4_K" and not GRID and not Q3K:
        run(rows_s, int(os.environ.get("DQ_HDRWIN", "2")))
    if os.environ.get("DQ_CT"):
        e(f"%ctpa = add.rr %spa, {k(CTOFF + 512)}")
        e("%ctp = mov.scalar-to-address %ctpa")          # one constant-table pointer shared by both halves
        CTP["on"] = True
    halves_s = []
    for h in range(2):
        x = e.stream()
        halves_s.append(x)
        hb0 = f"%hb0_{h}"
        hbp, pph1, pph = x.t("hbp"), x.t("pph"), x.t("pph")
        x(f"{hbp} = mov.scalar-to-address {hb0}")
        x(f"{pph1} = add.rr %spb, {k(PARTSS + h * NPARTS * 64)}")
        x(f"{pph} = mov.scalar-to-address {pph1}")
        BASEV[0] = hbp
        # ---- headers, vectorized: ls [row][j] = low bits from the scales_l nibbles | high bits from scales_h, then
        # the lane order [s'][row][f] (j = 4 f + s'); d per [row][f] for the U / V / OFF path
        k43, H, SLv, DSv, Dv, SHv = (x.t(n) for n in ("k43", "H", "SLv", "DSv", "Dv", "SHv"))
        x(f"{k43} = vbcst.8 {k(0x43)}")
        phd = P(x, hb0, HDR)
        def lsplit(LS):
            LT, LU, a_ = x.t("LT"), x.t("LU"), x.t("lsr")
            x(f"{LT} = vshuffle {LS}, {LS}, {k(35)}")            # [j][row] = [f][s'][row]
            x(f"{LU} = vshuffle {LT}, {LT}, {k(11)}")            # f = 1 half to the front
            x(f"{a_} = vshuffle {LT}, {LU}, {m20}")              # [s'][row][f]
            return a_
        if XS2:   # d as Q3_K; nibbles [row 0-3 | 4-7][16] -> record c's [row][j] (8-byte units of parity c: modes 6 / 7)
            x(f"{H} = vlda.512.i8x64 {phd}, 0")
            x(f"{DSv} = vshuffle {H}, {H}, {m4c}")
            x(f"{Dv} = vshuffle {DSv}, {DSv}, {m2c}")
            hv = x.t("hv")
            x(f"{hv} = vshuffle {Dv}, {Dv}, {m18}")              # d [row][f]
            scv = x.t("scv")
            x(f"{scv} = vlda.512.i8x64 {phd}, {SCM}")
            un = []
            for i_ in range(2):
                sl_, u_ = x.t("ss"), x.t("su")
                x(f"{sl_} = slice {scv}[{i_}] : {V2} -> {V1}")
                x(f"{u_} = vunpack.u4.to.u8x64 {sl_}")
                un.append(u_)
            chains = []
            for c in range(2):
                n_, n2_, ls_, b33 = x.t("sn"), x.t("n2"), x.t("LS"), x.t("b8")
                x(f"{n_} = vshuffle {un[0]}, {un[1]}, {k(6 + c)}")
                x(f"{n2_} = vadd.8 {n_}, {n_}")
                x(f"{b33} = vbcst.8 {k(33)}")
                x(f"{ls_} = vadd.8 {n2_}, {b33}")                # ls' = 33 + 2 ls: T = d (ls' - 32)
                chains.append((lsplit(ls_), hv, TC_IQ4, c, False))
        elif Q3K:   # d [row] as IQ4_XS's header shuffles leave it; per record c its 8 sub-blocks' sc [row][j] (SCM + 64 c)
            x(f"{H} = vlda.512.i8x64 {phd}, 0")
            x(f"{DSv} = vshuffle {H}, {H}, {m4c}")
            x(f"{Dv} = vshuffle {DSv}, {DSv}, {m2c}")
            hv = x.t("hv")
            x(f"{hv} = vshuffle {Dv}, {Dv}, {m18}")              # d [row][f]
            chains = []
            for c in range(2):
                ls_ = x.t("LS")
                x(f"{ls_} = vlda.512.i8x64 {phd}, {SCM + 64 * c}")
                chains.append((lsplit(ls_), hv, TC_IQ4, c, False))
        elif FMT == "Q4_K":
            # planes: 0 d | dmin [row], 1 s0..3, 2 s4..7, 3 s8..11; sc / m (6 bit) by the K-quant packing:
            # j < 4: sc = s_j & 63, m = s_{j+4} & 63; j >= 4: sc = (s_{j+4} & 15) | (s_{j-4} >> 6) << 4,
            # m = (s_{j+4} >> 4) | (s_j >> 6) << 4  (nibble-unpacked: lo + (hi & 3) << 4, lo / hi of plane 3 + (hi & 12) << 2)
            PW, PZ = x.t("PW"), x.t("PZ")
            x(f"{PW} = vlda.512.i8x64 {phd}, 0")
            x(f"{PZ} = vlda.512.i8x64 {phd}, 64")
            DMv, p0s, p0c, DD = x.t("DM"), x.t("p0"), x.t("p0"), x.t("DD")
            # plane 0 doubled (as IQ4_XS's d words come out of its header shuffles): the [row][f] layout below reads
            # the upper 16 lanes for odd sub-blocks
            x(f"{p0s} = slice {PW}[0] : {V2} -> {V1}")
            x(f"{p0c} = vmov.256 {p0s}" if False else f"{p0c} = slice {PW}[0] : {V2} -> {V1}")
            x(f"{DD} = concat({p0s}, {p0c}) : ({V1}, {V1}) -> {V2}")
            x(f"{Dv} = vshuffle {DD}, {DD}, {m2c}")            # d [row] (twice)
            x(f"{DMv} = vshuffle {DD}, {DD}, {m3c}")           # dmin [row] (twice)
            un = []
            for src, i_ in ((PW, 1), (PZ, 0), (PZ, 1)):
                sl_, u_ = x.t("ps"), x.t("pu")
                x(f"{sl_} = slice {src}[{i_}] : {V2} -> {V1}")
                x(f"{u_} = vunpack.u4.to.u8x64 {sl_}")
                un.append(u_)
            A_, B_, C_ = un

            def lo16(v):
                return band(x, v, 0x00FF)

            def dbl16(v, n):
                for _ in range(n):
                    v = add16(x, v, v)
                return v
            aH, bH = shr8(x, A_), shr8(x, B_)
            sc_lo = add16(x, lo16(A_), dbl16(band(x, aH, 3), 4))
            m_lo = add16(x, lo16(B_), dbl16(band(x, bH, 3), 4))
            sc_hi = add16(x, lo16(C_), dbl16(band(x, aH, 12), 2))
            m_hi = add16(x, shr8(x, C_), dbl16(band(x, bH, 12), 2))
            avs = []
            for lo_, hi_ in ((sc_lo, sc_hi), (m_lo, m_hi)):
                tt, LS = x.t("tt"), x.t("LS")
                x(f"{tt} = vshuffle {lo_}, {hi_}, {m0c}")
                x(f"{LS} = vshuffle {tt}, {tt}, {k(49)}")         # [row][j] (j < 4 from lo, j >= 4 from hi)
                avs.append(lsplit(LS))
            hvd, hvm = x.t("hv"), x.t("hv")
            x(f"{hvd} = vshuffle {Dv}, {Dv}, {m18}")
            x(f"{hvm} = vshuffle {DMv}, {DMv}, {m18}")
            chains = [(avs[0], hvd, TC_Q4K, 0, False), (avs[1], hvm, TC_Q4K, 2, True)]
        else:
            x(f"{H} = vlda.512.i8x64 {phd}, 0")
            x(f"{SLv} = vshuffle {H}, {H}, {k(5)}")              # odd 32-bit lanes: scales_l [row][4]
            x(f"{DSv} = vshuffle {H}, {H}, {m4c}")               # even 32-bit lanes: d | scales_h
            x(f"{Dv} = vshuffle {DSv}, {DSv}, {m2c}")            # d [row]
            x(f"{SHv} = vshuffle {DSv}, {DSv}, {m3c}")           # scales_h [row]
            pp = P(x, hb0, SCM)
            sl0, sh0 = x.t("sl"), x.t("sh")
            x(f"{sl0} = slice {SLv}[0] : {V2} -> {V1}")
            x(f"{sh0} = slice {SHv}[0] : {V2} -> {V1}")
            NL, NH, NH2 = x.t("NL"), x.t("NH"), x.t("NH2")
            if os.environ.get("DQ_RUNPACK", "0") == "1":         # register unpack (bit-exact, neutral: 1446 vs 1443): no SCM store / unpacking-load round trip
                x(f"{NL} = vunpack.u4.to.u8x64 {sl0}")           # [row][j]: the low 4 bits of ls_j
                x(f"{NH} = vunpack.u4.to.u8x64 {sh0}")           # [row][i]: scales_h nibble i (fields j = 2i, 2i + 1)
            else:
                x(f"vst.256.i8x32 {sl0}, {pp}, 0")
                x(f"vst.256.i8x32 {sh0}, {pp}, 32")
                x(f"{NL} = vldb.unpack.u4.to.u8x64 {pp}, 0")
                x(f"{NH} = vldb.unpack.u4.to.u8x64 {pp}, 32")
            x(f"{NH2} = vshuffle {NH}, {NH}, {m20}")             # [row][j]: nibble j // 2

            def b8(v):
                r_ = x.t("b8")
                x(f"{r_} = vbcst.8 {k(v)}")
                return r_

            def dbl(v, n):
                for _ in range(n):
                    r_ = x.t("db")
                    x(f"{r_} = vadd.8 {v}, {v}")
                    v = r_
                return v
            he0, ho0, mev, mod, he1, ho1, hp, LS, LT, LU, a, hv = (x.t(n) for n in (
                "he", "ho", "mev", "mod", "he", "ho", "hp", "LS", "LT", "LU", "lsr", "hv"))
            x(f"{he0} = vband {NH2}, {b8(3)}")
            he = dbl(he0, 4)                                     # (n & 3) << 4: even j
            x(f"{ho0} = vband {NH2}, {b8(12)}")
            ho = dbl(ho0, 2)                                     # ((n >> 2) & 3) << 4: odd j
            x(f"{mev} = vbcst.16 {k(0x00FF)}")
            x(f"{mod} = vbcst.16 {k(0xFF00)}")
            x(f"{he1} = vband {he}, {mev}")
            x(f"{ho1} = vband {ho}, {mod}")
            x(f"{hp} = vbor {he1}, {ho1}")
            if XXS:
                a = xxs_scales(x, h, b8)
            else:
                if GRID:                                     # IQ3_S: T = d (1 + 2 s) = d (ls - 32), ls = 33 + 2 s
                    n2_ = x.t("n2")
                    x(f"{n2_} = vadd.8 {NL}, {NL}")
                    x(f"{LS} = vadd.8 {n2_}, {b8(33)}")
                else:
                    x(f"{LS} = vbor {NL}, {hp}")             # ls [row][j]
                a = lsplit(LS)
            x(f"{hv} = vshuffle {Dv}, {Dv}, {m18}")              # d [row][f]
            chains = [(a, hv, TC_IQ4, 0, False)]
        for a, hv, tconst, pbase, neg in chains:
            e4 = band(x, shr8(x, hv), 0x7C)
            e128 = e4
            for _ in range(5):
                e128 = add16(x, e128, e128)
            ep128 = x.t("ep")
            x(f"{ep128} = max.u16x32 {e128}, {splat16(x, 128)}")
            offv = add16(x, sub16(x, ep128, splat16(x, 3200)), band(x, hv, 0x8000))
            z128 = sub16(x, ep128, e128)
            z1024 = add16(x, add16(x, z128, z128), add16(x, z128, z128))
            z1024 = add16(x, z1024, z1024)
            sd = add16(x, band(x, hv, 0x3FF), sub16(x, splat16(x, 1024), z1024))
            if neg:                                              # -M: the f16's sign flipped
                offv = add16(x, offv, splat16(x, 0x8000))
            uv = add16(x, shr8(x, add16(x, sd, sd)), splat16(x, 0x4300))
            vv = add16(x, band(x, sd, 127), splat16(x, 0x4300))
            NOTAB = os.environ.get("DQ_NOTAB", "1") == "1"   # T32 reads U / V / OFF from registers: no TAB round trip
            for val, o16 in (() if NOTAB and os.environ.get("DQ_T32", "1") == "1" else ((uv, 0), (vv, 32), (offv, 64))):
                lo = x.t("lo")
                x(f"{lo} = slice {val}[0] : {V2} -> {V1}")
                pp_, oo = x.addr(hbp, TAB + o16, 224, 32, "hb_tab_w")
                x(f"vst.256.i16x16 {lo}, {pp_}, {oo}")

            # ---- T = d (ls - 32) for all 8 sub-blocks in one block (lanes [s'][row][f]), split into bf16 parts
            def sreg():
                slo, shi = x.t("S"), x.t("S")
                x(f"{slo} = vshuffle {a}, {k43}, {m20}")
                x(f"{shi} = vshuffle {a}, {k43}, {m21}")
                return slo, shi

            def offc(v):
                return add16(x, col16(x, 64), splat16(x, v))

            def scaled(v):
                slo, shi = sreg()
                ov = offc(v)
                r4 = x.t("A")
                x(f"{r4} = concat({add16(x, slo, ov)}, {add16(x, shi, ov)}) : ({V2}, {V2}) -> {V4}")
                return r4

            def s4():
                slo, shi = sreg()
                r4 = x.t("S4")
                x(f"{r4} = concat({slo}, {shi}) : ({V2}, {V2}) -> {V4}")
                return r4

            if os.environ.get("DQ_T32", "1") == "1":
                # two 32-lane chains (sub-block pairs s' 0-1 / 2-3: slo / shi) with vmul / vmac.bf16x32 (accumulator lanes
                # 0..31): no 64-lane operand duplication (wide) or operand concats; the [row][f] tables load once
                if NOTAB:       # their 32 lanes already hold [row][f] twice (bit-exact; the TAB round trip was redundant)
                    U2, V2c, O2 = uv, vv, offv
                else:
                    U2, V2c, O2 = col16(x, 0), col16(x, 32), col16(x, 64)
                sl_, sh_ = sreg()
                sxs = (sl_, sh_)

                def m32(acc, s1, s2):
                    r_ = x.t("ac")
                    x(f"{r_} = vmul.bf16x32 {s1}, {s2}, {conf}" if acc is None else
                      f"{r_} = vmac.bf16x32 {acc}, {s1}, {s2}, {conf}")
                    return r_

                def conv32(acc):
                    q0, q1, h_, c_ = x.t("q"), x.t("q"), x.t("h"), x.t("cv")
                    x(f"{q0} = slice {acc}[0] : {M4} -> {M1}")
                    x(f"{q1} = slice {acc}[1] : {M4} -> {M1}")
                    x(f"{h_} = concat({q0}, {q1}) : ({M1}, {M1}) -> {M2}")
                    x(f"{c_} = vconv.bf16.fp32 {h_}")
                    return c_
                accs = [None, None]
                # the steps in both chains together: each operand built right before its two uses
                for kind, v in tconst:
                    ov = add16(x, O2, splat16(x, v))
                    one = splat16(x, 0x3F80) if kind == "1" else None
                    for c in range(2):
                        if kind == "su":
                            accs[c] = m32(accs[c], U2, add16(x, sxs[c], ov))
                        elif kind == "sv":
                            accs[c] = m32(accs[c], V2c, add16(x, sxs[c], ov))
                        elif kind == "u":
                            accs[c] = m32(accs[c], U2, ov)            # -20480 P U
                        elif kind == "v":
                            accs[c] = m32(accs[c], V2c, ov)           # -160 P V
                        elif kind == "s":
                            accs[c] = m32(accs[c], sxs[c], ov)        # -16512 P S
                        else:
                            accs[c] = m32(accs[c], one, ov)           # (2^21 + 2^14) P, 528384 P
                m1 = splat16(x, 0xBF80)
                for p in range(2):
                    cs_ = [conv32(accs[hi_]) for hi_ in range(2)]
                    for f in range(2):
                        # packed line [pair][s'][row] (both pairs: broadcast index 2 t + u): PARTS + (f 2 + h) NP 64 + p 64
                        dv = x.t("dv")
                        x(f"{dv} = vshuffle {cs_[0]}, {cs_[1]}, {k(2 + f)}")
                        # Q3_K: record pbase, its lines (f, p) at 2 f + p
                        pp_, o = x.addr(pph, (pbase * 2 * NPARTS * 64 + (2 * f + p) * 64) if Q3K or XS2 else
                                        (f * 2 * NPARTS * 64 + (pbase + p) * 64), cursor="pph_w")
                        x(f"vst.512.bf16x32 {dv}, {pp_}, {o}")
                    if p < 1:
                        accs = [m32(accs[hi_], cs_[hi_], m1) for hi_ in range(2)]
        continue                                                # (the 64-lane form below: DQ_T32=0, IQ4_XS only)
        acc = mac(x, None, wide(x, col16(x, 0)), scaled(0x380))
        acc = mac(x, acc, wide(x, col16(x, 32)), scaled(0))
        acc = mac(x, acc, wide(x, col16(x, 0)), wide(x, offc(0xC6A0)))     # -20480 P U
        acc = mac(x, acc, wide(x, col16(x, 32)), wide(x, offc(0xC320)))    # -160 P V
        acc = mac(x, acc, s4(), wide(x, offc(0xC681)))                     # -16512 P S
        acc = mac(x, acc, wide(x, splat16(x, 0x3F80)), wide(x, offc(0x4A01)))   # (2^21 + 2^14) P
        acc = mac(x, acc, wide(x, splat16(x, 0x3F80)), wide(x, offc(0x4901)))   # 528384 P
        res = acc
        for p in range(2):
            part, halves = conv(x, res)
            for hi_, hh in enumerate(halves):
                for f in range(2):
                    dv = x.t("dv")
                    x(f"{dv} = vshuffle {hh}, {hh}, {k(2 + f)}")
                    pp_, o = x.addr(pph, ((p * 2 + f) * 2 + hi_) * 64, cursor="pph_w")
                    x(f"vst.512.bf16x32 {dv}, {pp_}, {o}")
            if p < 1:
                res = mac(x, res, part, wide(x, splat16(x, 0xBF80)))
    run(halves_s, 2)
    CTP["on"] = False


    e("%mark_hotsetup = mova.i32 0") if os.environ.get("DQ_MARK") else None
    # -- hot: two output records (slots 0, 1), each 2 iterations of 2 sub-blocks (u) x 4 chunk-halves; T row-broadcast
    # from the halves' PARTS ((4 p + 2 r + t) 64: record r = sub-block group f, iteration t = sl // 2, u = sl % 2)
    e(f"%pp0a = add.rr %spb, {k(PARTSS)}")              # record r's parts: lines (hh NP + p) 64 at + 2 NP 64 r
    e("%pp0 = mov.scalar-to-address %pp0a")
    r_, (rp0,) = loop_begin("rl", 2, [("%pp0", "reg<aie2p.ep>")])
    # -- record r: qs rows of both halves realigned (vshift) and transposed for chunks 8 r .. 8 r + 7, then their 64
    # gather addresses each (GT + 8 b = (b + (GT >> 3)) << 3) at GAS + 1024 (local sub-block) + 512 h + 256 cc
    if GRID:
        # ---- IQ3_S record r (sub-blocks 4 r .. 4 r + 3) of both halves: qs (32 B at +2 + 32 r) and signs (16 B at
        # +74 + 16 r) realigned per row; qh (4 B at +66 + 4 r) by u16 loads into QS staging [row][s]; qh bits [s][l][e]
        # per row from the qh-bit table (gathers); qs and bits transposed to [kb][row][e] (zips 18, 16 / 17, 14 / 15),
        # signs to [kb][row] (zips 20, 18, 16 / 17); addresses (page << 12) + 16 index by vups << 4 into GA
        for nm_, off_, mul_ in ((("q", 2, 32),) if X2 or XS2 else (("q", 2, 32), ("r", 66, 16)) if XXS else
                                (("q", 2, 32), ("s", 74, 16), ("h", 66, 4))):
            e(f"%gr{nm_} = mul {r_}, {k(mul_)}")
            e(f"%gb{nm_} = add.rr %jin, %gr{nm_}")
            e(f"%go{nm_} = add.rr %gb{nm_}, {k(off_)}")
        for nm_ in (() if X2 or XS2 else ("q",) if XXS else ("q", "s")):
            e(f"%gsh{nm_} = and %go{nm_}, {k(63)}")
            e(f"%gal{nm_} = and %go{nm_}, {k(-64)}")
        if not (X2 or XS2):
            e("%cgv = vbcst.8 %gtc")
            e("%csv = vbcst.8 %stc")
        e("%sh4 = mov.shift 4")

        def qhrow(x, row):
            """IQ3_S: the row's qh word w (4 B at +66 + 4 r) as [s 4][8] bytes w_s (V1; bits picked after the network)."""
            ph_, h0_, h1_, h1s, hw, bc, sp, r1 = (x.t(n) for n in ("ph", "h0", "h1", "h1s", "hw", "hbc", "hsp", "hr"))
            x(f"{ph_} = add.rr %goh, {k(ROWB * row)}" if row else f"{ph_} = or %goh, {k(0)}")
            pq_ = Pr(x, ph_)
            x(f"{h0_} = lda.u16 {pq_}, 0")
            x(f"{h1_} = lda.u16 {pq_}, 2")
            x(f"{h1s} = lshl {h1_}, {k(16)}")
            x(f"{hw} = or {h0_}, {h1s}")
            x(f"{bc} = vbcst.32 {hw}")
            x(f"{sp} = vshuffle {bc}, {bc}, {k(35)}")       # 8 x 8 transpose of 16 copies: [s][8 copies] (twice)
            x(f"{r1} = slice {sp}[0] : {V2} -> {V1}")
            return r1

        def rowload(x, nm_, row, v1):
            pa_ = x.t("pa")
            x(f"{pa_} = add.rr %gal{nm_}, {k(ROWB * row)}" if row else f"{pa_} = or %gal{nm_}, {k(0)}")
            pv = Pr(x, pa_)
            va, vb, q = x.t("va"), x.t("vb"), x.t("q")
            x(f"{va} = vlda.512.i8x64 {pv}, 0")
            x(f"{vb} = vlda.512.i8x64 {pv}, 64")
            x(f"{q} = vshift {va}, {vb}, %gsh{nm_}")
            if not v1:
                return q
            q1 = x.t("q1")
            x(f"{q1} = slice {q}[0] : {V2} -> {V1}")
            return q1

        def net16(x, R):
            """8 rows of 32 B (V1, u16 units [kb][e]) -> 4 vectors [kb 4 g .. 4 g + 3][row][e] (zips 45, 16 / 17, 14 / 15)."""
            A = []
            for q in range(4):
                c_, a_ = x.t("pc"), x.t("za")
                x(f"{c_} = concat({R[2 * q]}, {R[2 * q + 1]}) : ({V1}, {V1}) -> {V2}")
                x(f"{a_} = vshuffle {c_}, {c_}, {k(45)}")
                A.append(a_)
            B = []
            for i_, j_ in ((0, 1), (2, 3)):
                for md in (k(16), k(17)):
                    b_ = x.t("zb")
                    x(f"{b_} = vshuffle {A[i_]}, {A[j_]}, {md}")
                    B.append(b_)                         # B0 (kb 0-7, r 0-3), B1 (kb 8-15, r 0-3), B2, B3 (r 4-7)
            O = []
            for i_, j_ in ((0, 2), (1, 3)):
                for md in (k(14), k(15)):
                    o_ = x.t("zo")
                    x(f"{o_} = vshuffle {B[i_]}, {B[j_]}, {md}")
                    O.append(o_)
            return O
        for hh in range(2):
            if XS2:
                # IQ2_XS record r: 8 rows' u16 words (k-blocks 2 w, 2 w + 1) transposed; grid index (9 bit, + page << 8:
                # the 8 KB table crosses a page) and sign field (7 bit, | page << 8) per k-block -> addresses
                x = e.stream()
                xa = x.t("xra")
                x(f"{xa} = add.rr %goq, {k(ROWB * 8 * hh)}" if hh else f"{xa} = or %goq, {k(0)}")
                O = aux_words(x, xa, 0)
                cvs = {}
                for kind, page_ in (("g", "%gtc"), ("s", "%stc")):
                    cw_, cv_ = x.t("cw"), x.t("cv")
                    x(f"{cw_} = lshl {page_}, {k(8)}")
                    x(f"{cv_} = vbcst.16 {cw_}")
                    cvs[kind] = cv_
                for a_ in range(2):                             # words 4 a .. 4 a + 3
                    for e_ in range(2):                         # the u16 of k-block 2 w + e
                        for kind, msk, sh0, mx in (("g", 0x1FF, 0, 0), ("s", 0xFE00, 9, 127)):
                            shr = x.t("shl")
                            x(f"{shr} = mov.shift {sh0 + 16 * e_}")
                            f_ = field(x, O[2 * a_], O[2 * a_ + 1], msk << (16 * e_), shr)
                            if mx and e_:                       # the field's shift is arithmetic
                                f_ = band(x, f_, mx)
                            u_ = x.t("su")
                            x(f"{u_} = {'vadd.16' if kind == 'g' else 'vbor'} {f_}, {cvs[kind]}")
                            for h2, md2 in enumerate((k(18), k(19))):
                                du, ua = x.t("sd"), x.t("sa")
                                x(f"{du} = vshuffle {u_}, {u_}, {md2}")    # [w 2][row][dup]
                                x(f"{ua} = vups.2x.x-to-c.unsigned {du}, %sh4")
                                for q2 in range(2):
                                    q1 = x.t("sg")
                                    kb_ = 2 * (4 * a_ + 2 * h2 + q2) + e_
                                    x(f"{q1} = slice {ua}[{q2}] : {M2} -> {M1}")
                                    pgs = P(x, "%spb", GAS + 2048 * hh + 256 * gpair(kb_) +
                                            (128 if kind == "s" else 0) + 64 * gpos(kb_))
                                    x(f"vst.acc {q1}, {pgs}, 0")
                run([x], 1)
                continue
            if X2:
                # IQ2_XXS record r: the 4 sub-blocks' index words (even units) and aux words (odd units) of 8 rows,
                # transposed; index byte l / sign field l -> (v | page << 8) [ib 4][row], duplicated -> addresses
                for kind, sel_, page_, wd in (("g", 4, "%gtc", 8), ("s", 5, "%stc", 7)):
                    x = e.stream()
                    xa = x.t("xra")
                    x(f"{xa} = add.rr %goq, {k(ROWB * 8 * hh)}" if hh else f"{xa} = or %goq, {k(0)}")
                    O = aux_words(x, xa, 0, sel_)
                    cw_, cv_ = x.t("cw"), x.t("cv")
                    x(f"{cw_} = lshl {page_}, {k(8)}")
                    x(f"{cv_} = vbcst.16 {cw_}")
                    for l in range(4):
                        shr = x.t("shl")
                        x(f"{shr} = mov.shift {wd * l}")
                        f_ = field(x, O[0], O[1], ((1 << wd) - 1) << (wd * l), shr)
                        if wd * l + wd == 32:                   # the field's shift is arithmetic: v - 256 for v >= 128
                            f_ = band(x, f_, 0xFF)
                        u_ = x.t("su")
                        x(f"{u_} = vbor {f_}, {cv_}")
                        for h2, md2 in enumerate((k(18), k(19))):
                            du, ua = x.t("sd"), x.t("sa")
                            x(f"{du} = vshuffle {u_}, {u_}, {md2}")    # [ib 2][row][dup]
                            x(f"{ua} = vups.2x.x-to-c.unsigned {du}, %sh4")
                            for q2 in range(2):
                                q1 = x.t("sg")
                                kb_ = 4 * (2 * h2 + q2) + l
                                x(f"{q1} = slice {ua}[{q2}] : {M2} -> {M1}")
                                pgs = P(x, "%spb", GAS + 2048 * hh + 256 * gpair(kb_) + (128 if kind == "s" else 0) +
                                        64 * gpos(kb_))
                                x(f"vst.acc {q1}, {pgs}, 0")
                    run([x], 1)
                continue
            # grid: qs rows (V1) and the qh-bit rows (gathers from the bit table) through the same network, then per
            # group: u16 qs | (page_g + bit) << 8, vups << 4 -> grid addresses
            rq = []
            Rq, Hrows = [], []
            for r in range(8):
                x = e.stream()
                rq.append(x)
                Rq.append(rowload(x, "q", 8 * hh + r, True))
                if not XXS:
                    Hrows.append(qhrow(x, 8 * hh + r))
            x = e.stream()
            rq.append(x)
            if XXS:
                Os = net16(x, Rq)
                for g in range(4):
                    pgg = P(x, "%spb", GAS + 2048 * hh + 512 * g)
                    for part, md in enumerate((m20, m21)):
                        z, ua = x.t("gz"), x.t("gu")
                        x(f"{z} = vshuffle {Os[g]}, %cgv, {md}")          # qs | page_g << 8
                        x(f"{ua} = vups.2x.x-to-c.unsigned {z}, %sh4")
                        for q2 in range(2):
                            q1 = x.t("ga")
                            kb_ = 4 * g + 2 * part + q2
                            x(f"{q1} = slice {ua}[{q2}] : {M2} -> {M1}")
                            x(f"vst.acc {q1}, {pgg}, {256 * (gpair(kb_) - 2 * g) + 64 * gpos(kb_)}")
                run(rq, int(os.environ.get("DQ_QWIN", "6")))
                # signs: the record's 4 aux words per row -> fields f_l = (w >> 7 l) & 127 [ib][row] -> (f | page_s << 8)
                # twice -> sign addresses of kb 4 ib + l
                x = e.stream()
                xa = x.t("xra")
                x(f"{xa} = add.rr %gor, {k(ROWB * 8 * hh)}" if hh else f"{xa} = or %gor, {k(0)}")
                O = aux_words(x, xa, 0)
                csw, csv16 = x.t("cw"), x.t("cv")
                x(f"{csw} = lshl %stc, {k(8)}")
                x(f"{csv16} = vbcst.16 {csw}")
                for l in range(4):
                    shr = x.t("shl")
                    x(f"{shr} = mov.shift {7 * l}")
                    f_ = field(x, O[0], O[1], 127 << (7 * l), shr)
                    u_ = x.t("su")
                    x(f"{u_} = vbor {f_}, {csv16}")                     # f + 256 page_s, lanes [ib 4][row]
                    for h2, md2 in enumerate((k(18), k(19))):
                        du, ua = x.t("sd"), x.t("sa")
                        x(f"{du} = vshuffle {u_}, {u_}, {md2}")        # [ib 2][row][dup]
                        x(f"{ua} = vups.2x.x-to-c.unsigned {du}, %sh4")
                        for q2 in range(2):
                            q1 = x.t("sg")
                            kb_ = 4 * (2 * h2 + q2) + l
                            x(f"{q1} = slice {ua}[{q2}] : {M2} -> {M1}")
                            pgs = P(x, "%spb", GAS + 2048 * hh + 256 * gpair(kb_) + 128 + 64 * gpos(kb_))
                            x(f"vst.acc {q1}, {pgs}, 0")
                run([x], 1)
                continue
            Hs = net16(x, Hrows)                                # [kb 4][row][e]: w_g (sub-block g's qh byte)
            Os = net16(x, Rq)
            hmv, one = x.t("hmv"), x.t("one")
            x(f"{hmv} = vlda.512.i8x64 {Pr(x, '%hma')}, 0")      # 1 << (2 kb + e)
            x(f"{one} = vbcst.8 {k(1)}")
            for g in range(4):
                pgg = P(x, "%spb", GAS + 2048 * hh + 512 * g)    # pairs 2 g, 2 g + 1 (short-lived: pointer budget)
                hb, h1, hc = x.t("hb"), x.t("h1"), x.t("hc")
                x(f"{hb} = vband {Hs[g]}, {hmv}")
                x(f"{h1} = min.u8x64 {hb}, {one}")                # the index's bit 8
                x(f"{hc} = vadd.8 {h1}, %cgv")
                for part, md in enumerate((m20, m21)):
                    z, ua = x.t("gz"), x.t("gu")
                    x(f"{z} = vshuffle {Os[g]}, {hc}, {md}")
                    x(f"{ua} = vups.2x.x-to-c.unsigned {z}, %sh4")
                    for q2 in range(2):
                        q1 = x.t("ga")
                        kb_ = 4 * g + 2 * part + q2
                        x(f"{q1} = slice {ua}[{q2}] : {M2} -> {M1}")
                        x(f"vst.acc {q1}, {pgg}, {256 * (gpair(kb_) - 2 * g) + 64 * gpos(kb_)}")
            run(rq, int(os.environ.get("DQ_QWIN", "6")))
            # signs: [row][16 B] (V2 rows) -> [kb][row] bytes (zips 20, 18, 16 / 17); (page_s << 8 | byte) twice -> GA
            rsg = []
            Rs = []
            for r in range(8):
                x = e.stream()
                rsg.append(x)
                Rs.append(rowload(x, "s", 8 * hh + r, False))
            x = e.stream()
            rsg.append(x)
            C_ = []
            for q in range(4):
                c_ = x.t("sc")
                x(f"{c_} = vshuffle {Rs[2 * q]}, {Rs[2 * q + 1]}, {k(20)}")
                C_.append(c_)
            D_ = []
            for i_, j_ in ((0, 1), (2, 3)):
                d_ = x.t("sd")
                x(f"{d_} = vshuffle {C_[i_]}, {C_[j_]}, {k(18)}")
                D_.append(d_)
            SV = []
            for md in (k(16), k(17)):
                sv_ = x.t("sv")
                x(f"{sv_} = vshuffle {D_[0]}, {D_[1]}, {md}")
                SV.append(sv_)
            for i in range(2):                               # kb 8 i .. 8 i + 7
                for part in range(2):                        # kb 8 i + 4 part .. + 3
                    pgs = P(x, "%spb", GAS + 2048 * hh + 256 * (4 * i + 2 * part) + 128)
                    sl_, nb_ = x.t("sl"), x.t("snb")
                    x(f"{sl_} = slice {SV[i]}[{part}] : {V2} -> {V1}")
                    x(f"{nb_} = vunpack.u4.to.u8x64 {sl_}")   # [kb 4][row][nibble]: even lanes values 0-3, odd 4-7
                    for h2, md2 in enumerate((m20, m21)):
                        du, ua = x.t("sd"), x.t("sa")
                        x(f"{du} = vshuffle {nb_}, %csv, {md2}")   # nibble | page_s << 8, [kb 2][row][nibble]
                        x(f"{ua} = vups.2x.x-to-c.unsigned {du}, %sh4")
                        for q2 in range(2):
                            q1 = x.t("sg")
                            kb_ = 8 * i + 4 * part + 2 * h2 + q2
                            x(f"{q1} = slice {ua}[{q2}] : {M2} -> {M1}")
                            x(f"vst.acc {q1}, {pgs}, {256 * (gpair(kb_) - 4 * i - 2 * part) + 64 * gpos(kb_)}")
            run(rsg, int(os.environ.get("DQ_QWIN", "6")))
    elif Q3K:
        # ---- Q3_K record r (values 128 r ..): per half the rows' [hmask 32 B | qs 32 B at 32 + 32 r] realigned and
        # transposed to [chunk][row][8 B] (chunks 0-3 hmask, 4-7 qs), nibble-unpacked into NIB: hmask nibble r (mode r:
        # even / odd bytes of the unpacked pair), qs lo and hi
        e(f"%qr32 = mul {r_}, {k(32)}")
        e("%qqb = add.rr %jin, %qr32")
        e(f"%qqa = add.rr %qqb, {k(32)}")
        for nm_, src in (("m", "%jin"), ("q", "%qqa")):
            e(f"%q3s{nm_} = and {src}, {k(63)}")
            e(f"%q3a{nm_} = and {src}, {k(-64)}")
        qstreams = []
        for hh in range(2):
            rws = []
            for r in range(8):
                x = e.stream()
                qstreams.append(x)
                row = 8 * hh + r
                hv_ = []
                for nm_ in ("m", "q"):
                    pa_ = x.t("pa")
                    x(f"{pa_} = add.rr %q3a{nm_}, {k(ROWB * row)}" if row else f"{pa_} = or %q3a{nm_}, {k(0)}")
                    pv = Pr(x, pa_)
                    va, vb, q, q1 = x.t("va"), x.t("vb"), x.t("q"), x.t("q1")
                    x(f"{va} = vlda.512.i8x64 {pv}, 0")
                    x(f"{vb} = vlda.512.i8x64 {pv}, 64")
                    x(f"{q} = vshift {va}, {vb}, %q3s{nm_}")
                    x(f"{q1} = slice {q}[0] : {V2} -> {V1}")
                    hv_.append(q1)
                rv = x.t("rv")
                x(f"{rv} = concat({hv_[0]}, {hv_[1]}) : ({V1}, {V1}) -> {V2}")
                rws.append(rv)
            x = e.stream()
            qstreams.append(x)
            vs = rws
            for d in (4, 2, 1):
                out = list(vs)
                for i in range(8):
                    jj = i ^ d
                    if i < jj:
                        a_, b_ = x.t("tq"), x.t("tq")
                        x(f"{a_} = vshuffle {vs[i]}, {vs[jj]}, {k(14)}")
                        x(f"{b_} = vshuffle {vs[i]}, {vs[jj]}, {k(15)}")
                        out[i], out[jj] = a_, b_
                vs = out
            pn = P(x, "%spb", NIB + 768 * hh + 512)
            for c in range(8):
                us = []
                for hf_ in range(2):
                    sl_, u_ = x.t("ns"), x.t("nu")
                    x(f"{sl_} = slice {vs[c]}[{hf_}] : {V2} -> {V1}")
                    x(f"{u_} = vunpack.u4.to.u8x64 {sl_}")
                    us.append(u_)
                for md, slot in ([(r_, c)] if c < 4 else [(m0c, c), (m1c, c + 4)]):
                    nv = x.t("nv")
                    x(f"{nv} = vshuffle {us[0]}, {us[1]}, {md}")
                    x(f"vst.512.i8x64 {nv}, {pn}, {64 * slot - 512}")
        run(qstreams, int(os.environ.get("DQ_QWIN", "1")))          # 8 rows of 64 B live: one row stream at a time
    else:
        e(f"%r64 = mul {r_}, {k(64)}")
        # 544 = 32 (mod 64): a row's qs realignment depends only on its parity: aligned bases and shifts of rows 0 / 1
        for par in range(2):
            e(f"%qs{par} = add.rr %jin, {k(ROWB * par + HBY)}")
            e(f"%sh{par} = and %qs{par}, {k(63)}")
            e(f"%al{par} = and %qs{par}, {k(-64)}")
            e(f"%ar{par} = add.rr %al{par}, %r64")
        # the table base in 32 accumulator lanes: addresses = (b << 3) + GT on the MAC slot (vadd.acc.integer mode 0)
        e("%gtv1 = vbcst.32 %spa")
        e(f"%gt2s = add.rr %spb, {k(GT2)}")
        e("%gtv2 = vbcst.32 %gt2s")
        e(f"%gtv = vshuffle %gtv1, %gtv2, {k(14)}" if os.environ.get("DQ_GT2", "1") == "1" else "%gtv = vbor %gtv1, %gtv1")   # u32 lanes A A B B ...
        for q in range(4):
            e(f"%gta{q} = vmov.vector512.to.accumulator512 %gtv")
        e(f"%gtacc = concat(%gta0, %gta1, %gta2, %gta3) : ({M1}, {M1}, {M1}, {M1}) -> {M4}")
        e(f"%amode = mova.i32 0")
        qstreams = []
        if os.environ.get("DQ_PKC", "1") == "1":            # transpose modes from an opaque zero: no per-use remat
            e(f"%pz = lshl {r_}, {k(-8)}")
            e("%pk14 = add %pz, 14")
            e("%pk15 = add %pz, 15")
            pk14, pk15 = "%pk14", "%pk15"
        else:
            pk14, pk15 = k(14), k(15)
        for hh in range(2):
            qs_ = []
            for r in range(8):
                x = e.stream()
                qstreams.append(x)
                row = 8 * hh + r
                sh, pa_ = f"%sh{row % 2}", x.t("pa")
                x(f"{pa_} = add.rr %ar{row % 2}, {k(ROWB * (row - row % 2))}" if row > 1 else f"{pa_} = or %ar{row % 2}, {k(0)}")
                pv = Pr(x, pa_) if os.environ.get("DQ_ABL") != "3" else P(x, "%spa", 128 * r + 1024 * hh)   # 3: timing ablation (rows from the table)
                va, vb, q = x.t("va"), x.t("vb"), x.t("q")
                x(f"{va} = vlda.512.i8x64 {pv}, 0")
                x(f"{vb} = vlda.512.i8x64 {pv}, 64")
                x(f"{q} = vshift {va}, {vb}, {sh}")
                qs_.append(q)
            x = e.stream()
            qstreams.append(x)
            vs = qs_
            for d in (4, 2, 1):
                out = list(vs)
                for i in range(8):
                    jj = i ^ d
                    if i < jj:
                        a_, b_ = x.t("tq"), x.t("tq")
                        x(f"{a_} = vshuffle {vs[i]}, {vs[jj]}, {pk14}")
                        x(f"{b_} = vshuffle {vs[i]}, {vs[jj]}, {pk15}")
                        out[i], out[jj] = a_, b_
                vs = out
            for i in range(8):                       # local chunk i: sub-block i // 2, cc = i % 2
                x = e.stream()
                qstreams.append(x)
                pg = P(x, "%spb", GAS + 1024 * (i // 2) + 512 * hh + 512)     # +512: offsets -512 .. 448
                u8s = []
                for hb_ in range(2):                     # bytes 32 hb_ .. + 31 (rows 4 hb_ .. + 3): 32 x u32 b << 3
                    bv, u8 = x.t("bv"), x.t("u8")
                    x(f"{bv} = slice {'%z8' if os.environ.get('DQ_ABL') == '1' else vs[i]}[{hb_}] : {V2} -> {V1}")   # DQ_ABL=1: timing ablation (all gathers -> entry 0)
                    x(f"{u8} = vups.4x.w-to-c.unsigned {bv}, %sh3")
                    u8s.append(u8)
                u4, a_ = x.t("u4"), x.t("ua")
                x(f"{u4} = concat({u8s[0]}, {u8s[1]}) : ({M2}, {M2}) -> {M4}")
                x(f"{a_} = vadd.acc.integer {u4}, %gtacc, %amode")         # + GT (64 lanes, MAC slot)
                for q2 in range(4):
                    q1 = x.t("ga")
                    x(f"{q1} = slice {a_}[{q2}] : {M4} -> {M1}")
                    x(f"vst.acc {q1}, {pg}, {256 * (i % 2) + 64 * q2 - 512}")
        run(qstreams, int(os.environ.get("DQ_QWIN", "6")))
    e(f"%qp0a = add.rr %spb, {k(GAS + 512)}")           # GA of local sub-block 0, +512 (offsets -512 .. 448)
    e("%qp0 = mov.scalar-to-address %qp0a")
    rq = "%qp0"
    e(f"%jso = mul {r_}, {k(OUT_B)}")
    if INLINE[0]:                               # no output ring: super-block j's records at 2 j + r
        e(f"%jsj = mul {j}, {k(2 * OUT_B)}")
        e("%jsr = add.rr %jso, %jsj")
        e(f"%osa = add.rr {OUTA[0]}, %jsr")
    else:
        e(f"%osa = add.rr {OUTA[0]}, %jso")
    e("%osp = mov.scalar-to-address %osa")
    lockop("acq", "%km1o")
    e.pre.append("  %km1o = mov.i32 -1")
    e.consts["%km1o"] = True
    e("%op = copy %osp : reg<aie2p.ep> -> reg<aie2p.mpfs>")
    e("%sl = vlda.store-fifo.low512 %osp, 0")
    e("%f0 = vlda.store-fifo.high512 %osp, %sl, 64")
    e("%pos = mova.fifo.store.position 0")
    SLU = os.environ.get("DQ_SLU", "1") == "1"   # the two sl iterations unrolled into one schedule (-5.5% decoder)
    if SLU:
        s_, (hf, hp, hq, sq, sp0) = r_, ("%f0", "%op", "%pos", rq, rp0)
    else:
        s_, (hf, hp, hq, sq, sp0) = loop_begin("sl", 2, [("%f0", "reg<aie2p.mstfifo>"), ("%op", "reg<aie2p.mpfs>"),
                                                        ("%pos", "reg<aie2p.mr26_fifo_st>"), (rq, "reg<aie2p.ep>"),
                                                        (rp0, "reg<aie2p.ep>")])
    st = {"fifo": (hf, hp, hq), "nf": 0}
    # loop-local constants: hoisted ones live across the prep and get rematerialized at every use
    nonlocal_consts = {}
    # from an opaque zero (loop counter >> 8): identical constants elsewhere would be CSE'd into one long-lived value,
    # which the allocator then rematerializes at every use (52 extra load-slot ops per iteration)
    e(f"%hz = lshl {s_}, {k(-8)}")
    for v, nm in ((2, "m2c"), (3, "m3c"), (4, "m4c"), (52, "m52"), (53, "m53"), (60, "conf")):
        e(f"%hk{v} = add %hz, {v}")
        nonlocal_consts[nm] = f"%hk{v}"
    m2c, m3c, m4c, m52, m53, conf = (nonlocal_consts[n] for n in ("m2c", "m3c", "m4c", "m52", "m53", "conf"))

    def push(x, wv):
        st["nf"] += 1
        nf, fifo = st["nf"], st["fifo"]
        nxt = (f"%xf{nf}", f"%xp{nf}", f"%xq{nf}")
        x(f"{nxt[0]}, {nxt[1]}, {nxt[2]} = vst.push.bfp16ebs8.from.fp32 {fifo[0]}, {wv}, {fifo[1]}, {fifo[2]}")
        st["fifo"] = nxt

    def flush(x):
        st["nf"] += 1
        nf, fifo = st["nf"], st["fifo"]
        nxt = (f"%xf{nf}", f"%xp{nf}", f"%xq{nf}")
        x(f"{nxt[0]}, {nxt[1]}, {nxt[2]} = vst.flush.512 {fifo[0]}, {fifo[1]}, {fifo[2]}")
        st["fifo"] = nxt

    if GRID:
        # ---- grid hot: per k-block pair (same sub-block) and row half. G streams (per k-block and table): one
        # address vector -> 4 gathers (rows in lane order: [row][8] directly); C stream: T rows (2 parts), X = grid |
        # signs, 2 MACs per k-block, pushes [kb][h]. Small G streams keep many gathers in flight (IQ4_XS's shape).
        frags, tails = [], []
        gpp = {}
        for pr in range(8):
            sl_ = pr // 2
            parked = None
            for hh in range(2):
                xs4 = []
                for pos in range(2):
                    x = e.stream()
                    frags.append(x)
                    tails.append([])
                    if (hh, pr // 4) not in gpp:     # pairs 4 j .. 4 j + 3 of a half: offsets -512 .. 448
                        gpp[(hh, pr // 4)] = P(x, "%spb", GAS + 2048 * hh + 1024 * (pr // 4) + 512)
                    gv = {}
                    for kind, off in (("g", 0), ("s", 128)):
                        ad = x.t("ad")
                        x(f"{ad} = vlda.512.i8x64 {gpp[(hh, pr // 4)]}, {256 * (pr % 4) + off + 64 * pos - 512}")
                        for hf_ in range(2):
                            av, gl, gh, gc = x.t("av"), x.t("gl"), x.t("gh"), x.t("gc")
                            x(f"{av} = slice {ad}[{hf_}] : {V2} -> {V1}")
                            x(f"{gl} = vldb.4x32.lo {av}")
                            x(f"{gh} = vldb.4x32.hi {av}")
                            x(f"{gc} = concat({gl}, {gh}) : ({V1}, {V1}) -> {V2}")
                            gv[(kind, hf_)] = gc
                    xvs = []
                    for hf_ in range(2):
                        xv = x.t("xv")
                        x(f"{xv} = vbor {gv[('g', hf_)]}, {gv[('s', hf_)]}")
                        xvs.append(xv)
                    x4 = x.t("X")
                    x(f"{x4} = concat({xvs[0]}, {xvs[1]}) : ({V2}, {V2}) -> {V4}")
                    xs4.append(x4)
                x = e.stream()
                frags.append(x)

                def tload(line0, lane):
                    t4s = []
                    for p in range(2):
                        ld, bc, ra, rb, t4 = x.t("pl"), x.t("bc"), x.t("ra"), x.t("rb"), x.t("TB")
                        x(f"{ld} = vlda.512.bf16x32 {sp0}, {(line0 + p) * 64}")
                        x(f"{bc} = vbroadcast.bf16x8.to.bf16x32 {ld}, {lane}")
                        x(f"{ra} = vshuffle {bc}, {bc}, {m52}")
                        x(f"{rb} = vshuffle {bc}, {bc}, {m53}")
                        x(f"{t4} = concat({ra}, {rb}) : ({V2}, {V2}) -> {V4}")
                        t4s.append(t4)
                    return t4s
                accs = []
                if XS2 and GP4:   # the k-block's 16-value group g = kb // 2: line 2 (g // 4) + p, lane g % 4 (one T live)
                    for pos, x4 in enumerate(xs4):
                        kb_ = 4 * (pr // 2) + 2 * pos + pr % 2 if GP4 else 2 * pr + pos
                        t4s = tload(hh * NPARTS + 2 * (kb_ // 8), (kb_ // 2) % 4)
                        acc = mac(x, None, t4s[0], x4)
                        accs.append(mac(x, acc, t4s[1], x4))
                else:             # (IQ2_XS in natural order: the pair is one group, pr)
                    t4s = tload(hh * NPARTS + 2 * (pr // 4), pr % 4) if XS2 else tload(hh * NPARTS, sl_)
                    for x4 in xs4:
                        acc = mac(x, None, t4s[0], x4)
                        accs.append(mac(x, acc, t4s[1], x4))
                if hh == 0:
                    parked = accs
                    tails.append([])
                    continue
                pt = e.stream()
                for n_ in range(2):
                    push(pt, parked[n_])
                    push(pt, accs[n_])
                if pr % 2 == 1:
                    flush(pt)
                tails.append(pt.L)
    elif Q3K:
        # ---- Q3_K hot: one stream per panel position (k-block 4 j + c of the record) and half; pushes [kb][h]
        frags, tails = [], []
        pns = {}
        npush = 0
        for pos in range(16):
            kbl = int(KPERM[pos])
            j_, c_ = kbl // 4, kbl % 4
            sbl = 2 * j_ + c_ // 2                              # the record's sub-block: line 2 (sbl // 4) + p, lane sbl % 4
            for hh in range(2):
                x = e.stream()
                frags.append(x)
                if hh not in pns:
                    pns[hh] = P(x, "%spb", NIB + 768 * hh + 512)
                nq, nh, cm, hm_ = x.t("nq"), x.t("nh"), x.t("cm"), x.t("hm")
                x(f"{nq} = vlda.512.i8x64 {pns[hh]}, {64 * ((4 if j_ < 2 else 8) + c_) - 512}")
                x(f"{nh} = vlda.512.i8x64 {pns[hh]}, {64 * c_ - 512}")
                b1, b2, b3 = x.t("b8"), x.t("b8"), x.t("b8")
                x(f"{b1} = vbcst.8 {k(3 << (2 * (j_ % 2)))}")
                x(f"{cm} = vband {nq}, {b1}")                     # crumb (j odd: 4 crumb)
                x(f"{b2} = vbcst.8 {k(1 << j_)}")
                x(f"{hm_} = vband {nh}, {b2}")                    # hbit 2^j -> 4 hbit (j odd: 16 hbit)
                for _ in range((2, 3, 0, 1)[j_]):
                    d_ = x.t("hd")
                    x(f"{d_} = vadd.8 {hm_}, {hm_}")
                    hm_ = d_
                v_, xl, xh, X = x.t("v"), x.t("xl"), x.t("xh"), x.t("X")
                x(f"{v_} = vbor {cm}, {hm_}")
                x(f"{b3} = vbcst.8 {k(0x43 if j_ % 2 == 0 else 0x42)}")
                x(f"{xl} = vshuffle {v_}, {b3}, {m20}")
                x(f"{xh} = vshuffle {v_}, {b3}, {m21}")
                x(f"{X} = concat({xl}, {xh}) : ({V2}, {V2}) -> {V4}")   # bf16 128 + v / 32 + v, [row][8]
                tb = []
                for p in range(2):
                    ld, bc, ra, rb, t4 = x.t("pl"), x.t("bc"), x.t("ra"), x.t("rb"), x.t("TB")
                    x(f"{ld} = vlda.512.bf16x32 {sp0}, {(hh * NPARTS + 2 * (sbl // 4) + p) * 64}")
                    x(f"{bc} = vbroadcast.bf16x8.to.bf16x32 {ld}, {sbl % 4}")
                    x(f"{ra} = vshuffle {bc}, {bc}, {m52}")
                    x(f"{rb} = vshuffle {bc}, {bc}, {m53}")
                    x(f"{t4} = concat({ra}, {rb}) : ({V2}, {V2}) -> {V4}")
                    tb.append(t4)
                nk = wide(x, splat16(x, 0xC304 if j_ % 2 == 0 else 0xC210))     # -K: -132 / -36
                acc = mac(x, None, tb[0], nk)
                acc = mac(x, acc, tb[1], nk)
                acc = mac(x, acc, tb[0], X)
                acc = mac(x, acc, tb[1], X)
                pt = e.stream()
                push(pt, acc)
                npush += 1
                if npush % 8 == 0:
                    flush(pt)                                   # every 8 pushes
                tails.append(pt.L)
    else:
        order = [(0, 0), (0, 1), (1, 0), (1, 1)]           # (chunk cc of the sub-block, half)
        frags, tails = [], []
        gpp = {}
        for t_, u in ([(t_, u) for t_ in range(2) for u in range(2)] if SLU else [(0, 0), (0, 1)]):
          if u == 0:
            hi_acc = {}
          if SLU:
            gp_ = None                                      # GA of (t, u): + 2048 t + 1024 u, a lazily defined pointer
          elif u == 0:
            e(f"%sqbc = copy {sq} : reg<aie2p.ep> -> reg<aie2p.ep>")      # GA of sub-block u = 1: +1024
            e("%sqb1 = padda %sqbc, 448")
            e("%sqb2 = padda %sqb1, 448")
            e("%sqb = padda %sqb2, 128")
            gptr = [sq, "%sqb"]
          if True:
            for i, (cc, hh) in enumerate(order):
                # G streams: per 16 bytes one address vector -> 4 gathers -> compaction (small register footprint, many in
                # flight); C: lo / hi extraction, T, X, MACs, push
                ents = {}
                for half in range(2):
                    for blk in range(2):
                        x = e.stream()
                        frags.append(x)
                        tails.append([])
                        ad = x.t("ad")
                        if SLU and (t_, u) not in gpp:    # GA of (t, u) from the scalar base: one add + one move
                            gpp[(t_, u)] = P(x, "%qp0a", 2048 * t_ + 1024 * u)
                        gpu_ = gpp[(t_, u)] if SLU else gptr[u]
                        x(f"{ad} = vlda.512.i8x64 {gpu_}, {512 * hh + 256 * cc + 128 * half + 64 * blk - 512}")
                        gls = []
                        for hf_ in range(2):
                            av, gl, gh = x.t("av"), x.t("gl"), x.t("gh")
                            x(f"{av} = slice {ad}[{hf_}] : {V2} -> {V1}")
                            x(f"{gl} = vldb.4x32.lo {av}")
                            if os.environ.get("DQ_ABL") == "4":           # timing ablation: half the gathers
                                gls.append((gl, gl))
                                continue
                            x(f"{gh} = vldb.4x32.hi {av}")
                            gls.append((gl, gh))
                        gcs = []
                        for gl, gh in gls:
                            gc = x.t("gc")
                            x(f"{gc} = concat({gl}, {gh}) : ({V1}, {V1}) -> {V2}")
                            gcs.append(gc)
                        ev = x.t("en")
                        x(f"{ev} = vshuffle {gcs[0]}, {gcs[1]}, {m4c}")
                        ents[(half, blk)] = ev
                x = e.stream()
                frags.append(x)
                ppt = sp0
                lo_p, hi_p = [], []
                lds = []
                for p in range(2 if NPARTS != 4 else 0):  # the T parts first: their load latency overlaps the gathers
                    ld = x.t("pl")
                    x(f"{ld} = {'vldb' if os.environ.get('DQ_TLDB') == '1' else 'vlda'}.512.bf16x32 {ppt}, {(hh * NPARTS + p) * 64}")
                    lds.append(ld)

                def build_xs():
                    for half in range(2):
                        for mode, dst in ((m2c, lo_p), (m3c, hi_p)):
                            pv = x.t("xp")
                            x(f"{pv} = vshuffle {ents[(half, 0)]}, {ents[(half, 1)]}, {mode}")
                            dst.append(pv)
                    xs_ = []
                    for ps in (lo_p, hi_p):
                        x4 = x.t("X")
                        x(f"{x4} = concat({ps[0]}, {ps[1]}) : ({V2}, {V2}) -> {V4}")
                        xs_.append(x4)
                    return xs_
                if NPARTS != 4:
                    xs = build_xs()
                accs = [None, None]

                def rowb(ld, idx=None):
                    bc, ra, rb, t4 = x.t("bc"), x.t("ra"), x.t("rb"), x.t("TB")
                    x(f"{bc} = vbroadcast.bf16x8.to.bf16x32 {ld}, {2 * t_ + u if idx is None else idx}")
                    x(f"{ra} = vshuffle {bc}, {bc}, {m52}")
                    x(f"{rb} = vshuffle {bc}, {bc}, {m53}")
                    x(f"{t4} = concat({ra}, {rb}) : ({V2}, {V2}) -> {V4}")
                    return t4
                if NPARTS == 4:
                    # Q4_K: a byte's low nibbles are sub-block 2 t, its high nibbles 2 t + 1 (separate scales). The -dmin m
                    # term first (32-lane vmul / vmac against 1.0 per broadcast half, accumulator halves concatenated), each
                    # scale line loaded right before its broadcasts; then the fragments; then T (lines loaded late)
                    one2 = x.t("one")
                    x(f"{one2} = vbcst.16 {k(0x3F80)}")
                    halves_acc = [[None, None], [None, None]]          # [n_][broadcast half]
                    for p in (2, 3):
                        ld = x.t("pl")
                        x(f"{ld} = vlda.512.bf16x32 {ppt}, {(hh * NPARTS + p) * 64}")
                        for n_ in range(2):
                            bc, ra, rb = x.t("bc"), x.t("ra"), x.t("rb")
                            x(f"{bc} = vbroadcast.bf16x8.to.bf16x32 {ld}, {2 * t_ + n_}")
                            x(f"{ra} = vshuffle {bc}, {bc}, {m52}")
                            x(f"{rb} = vshuffle {bc}, {bc}, {m53}")
                            for hb_, v_ in enumerate((ra, rb)):
                                a0 = halves_acc[n_][hb_]
                                r_ = x.t("ma")
                                x(f"{r_} = vmul.bf16x32 {v_}, {one2}, {conf}" if a0 is None else
                                  f"{r_} = vmac.bf16x32 {a0}, {v_}, {one2}, {conf}")
                                halves_acc[n_][hb_] = r_
                    for n_ in range(2):
                        sl4 = []
                        for hb_ in range(2):
                            for q_ in range(2):
                                q1 = x.t("mq")
                                x(f"{q1} = slice {halves_acc[n_][hb_]}[{q_}] : {M4} -> {M1}")
                                sl4.append(q1)
                        m4 = x.t("m4")
                        x(f"{m4} = concat({sl4[0]}, {sl4[1]}, {sl4[2]}, {sl4[3]}) : ({M1}, {M1}, {M1}, {M1}) -> {M4}")
                        accs[n_] = m4 if os.environ.get("DQ_DBG") != "noM" else None
                    xs = build_xs()
                    if os.environ.get("DQ_DBG") == "tdump":   # X = 1.0, no -M: the fragments are the broadcast T0 + T1
                        o3, o4, ones = x.t("one"), x.t("one"), x.t("one4")
                        x(f"{o3} = vbcst.16 {k(0x3F80)}")
                        x(f"{o4} = vbcst.16 {k(0x3F80)}")
                        x(f"{ones} = concat({o3}, {o4}) : ({V2}, {V2}) -> {V4}")
                        xs = [ones, ones]
                        accs = [None, None]
                    for p in range(2):
                        ld = x.t("pl")
                        x(f"{ld} = vlda.512.bf16x32 {ppt}, {(hh * NPARTS + p) * 64}")
                        for n_, x4 in enumerate(xs):
                            ix = 2 * t_ + (0 if os.environ.get("DQ_DBG") == "hi0" else n_)
                            accs[n_] = mac(x, accs[n_], rowb(ld, ix), x4)
                if NPARTS != 4:
                    for p in range(2):
                        ld = lds[p]
                        bc, ra, rb, t4 = x.t("bc"), x.t("ra"), x.t("rb"), x.t("TB")
                        if os.environ.get("DQ_ABL") == "5":          # timing ablation: T rows loaded, not broadcast
                            x(f"{rb} = vlda.512.bf16x32 {ppt}, {512 * hh - 512 + 256 * p + 64}")
                            x(f"{t4} = concat({ld}, {rb}) : ({V2}, {V2}) -> {V4}")
                        else:
                            x(f"{bc} = vbroadcast.bf16x8.to.bf16x32 {ld}, {2 * t_ + u}")
                            x(f"{ra} = vshuffle {bc}, {bc}, {m52}")
                            x(f"{rb} = vshuffle {bc}, {bc}, {m53}")
                            x(f"{t4} = concat({ra}, {rb}) : ({V2}, {V2}) -> {V4}")
                        for n_, x4 in enumerate(xs):
                            accs[n_] = mac(x, accs[n_], t4, x4)
                hi_acc[(cc, hh)] = accs[1]
                pt = e.stream()
                # panel k-block order per sub-block: chunk c's low nibbles then its high nibbles, each [h0, h1]:
                # (4s, 4s+2, 4s+1, 4s+3) -- the activations use the same k-block permutation
                push(pt, accs[0])
                if hh == 1:
                    push(pt, hi_acc[(cc, 0)])
                    push(pt, hi_acc[(cc, 1)])
                    if cc == 1 and (u == 0 or SLU):
                        flush(pt)                          # every 8 pushes (SLU: the iteration-end flush too)
                tails.append(pt.L)
    run(frags, int(os.environ.get("DQ_WINH", "1" if Q3K else "6")), tails,   # Q3_K: 4 x4 operands per stream
        hv=int(os.environ.get("DQ_HHV", "20" if GRID else "24")))   # grid: X (4 vec256) per G stream, 8 T rows per C
    if SLU:
        e(f"%spc0 = copy {sp0} : reg<aie2p.ep> -> reg<aie2p.ep>")
        if 2 * NPARTS * 64 <= 448:
            e(f"%spn0 = padda %spc0, {2 * NPARTS * 64}")
        else:                                            # padda immediates reach 448
            e(f"%spm0 = padda %spc0, 256")
            e(f"%spn0 = padda %spm0, {2 * NPARTS * 64 - 256}")
        lockop("rel", "%k1r")
        loop_end("rl", [("%spn0", "reg<aie2p.ep>")])
    ff = st["fifo"]
    if not SLU:
      e(f"%yf, %yp, %yq = vst.flush.512 {ff[0]}, {ff[1]}, {ff[2]}")
      e("%sqc = copy %sqb : reg<aie2p.ep> -> reg<aie2p.ep>")
      e("%sqm = padda %sqc, 448")
      e("%sqk = padda %sqm, 448")
      e("%sqn = padda %sqk, 128")
      e(f"%spc0 = copy {sp0} : reg<aie2p.ep> -> reg<aie2p.ep>")
      e("%spn0 = padda %spc0, 64")
      loop_end("sl", [("%yf", "reg<aie2p.mstfifo>"), ("%yp", "reg<aie2p.mpfs>"), ("%yq", "reg<aie2p.mr26_fifo_st>"),
                      ("%sqn", "reg<aie2p.ep>"), ("%spn0", "reg<aie2p.ep>")])
      lockop("rel", "%k1r")
      loop_end("rl", [(sp0, "reg<aie2p.ep>")])
    e.pre.append("  %k1r = mova.i32 1")
    e.consts["%k1r"] = True
    loop_end("jl")
    sec = e.L[sec0:]
    if PLOOP:
        loop_end("pl")
    for pp_ in range(1, nports if not PLOOP else 1):
        e.L += port_copy(sec, pp_, nports)
    e("return")
    e.L.append("}")
    pre_ = grid_rodata(name) if GRID else []
    return "\n".join(pre_ + e.L[:hdr_end] + place_consts(e.pre, e.L[hdr_end:]))


def leaf_inline(nsb, name="inl"):
    """The leaf's body for inlining into another core function: decodes nsb super-blocks of an input record
    ([16 rows][RECROW]) at %in into OUT_B records at %out (2 per super-block, no output ring), storages at %sp
    (SCR_B, 2048-aligned) and %sp2 (SCR2_B), all defined by the caller (the table is rebuilt each call). Returns the
    lines (not renamed; the caller prefixes its values and labels). Grid formats gather from module rodata
    (inline_rodata(name), defined once by the caller's module)."""
    INLINE[0] = nsb
    try:
        txt = leaf(name, 1)
    finally:
        INLINE[0] = 0
    out = []
    for ln in txt.split("\n"):
        t = ln.strip()
        if (t.startswith("global.rodata.def ") or t.startswith("low.func.def ") or " = resource<" in t or
                " = storage " in t or " = storage_address " in t or t in ("return", "}")):
            continue
        out.append(ln)
    return out


def inline_rodata(name="inl"):
    """The module rodata leaf_inline(.., name)'s code reads (grid formats' tables), else []."""
    return grid_rodata(name) if GRID else []


def place_consts(pre, body):
    """Constants used inside loops: a copy per innermost loop body (defined at its start, uses renamed), so no constant
    lives across loops (long-lived constants got rematerialized at every use). Elsewhere: right before the first use,
    and before the table build's branch when used both inside and outside that conditional block."""
    import re
    defs = {d.split("=")[0].strip(): d for d in pre}
    # innermost loop of each line
    stack, owner, nloops = [], [], 0
    for l in body:
        if l == "// LOOP-BEGIN":
            nloops += 1
            stack.append(nloops)
            owner.append(None)
        elif l == "// LOOP-END":
            stack.pop()
            owner.append(None)
        else:
            owner.append(stack[-1] if stack else None)
    begin = {}
    cur = 0
    for n_, l in enumerate(body):
        if l == "// LOOP-BEGIN":
            cur += 1
            begin[cur] = n_
    local = {}                      # loop -> constants used in its own lines
    out_body = []
    for l, o in zip(body, owner):
        if o is not None:
            names = [nm for nm in re.findall(r"%k[\w]+", l) if nm in defs]
            for nm in names:
                local.setdefault(o, []).append(nm) if nm not in local.get(o, []) else None
            l = re.sub(r"%k[\w]+", lambda m: f"{m.group(0)}_L{o}" if m.group(0) in defs else m.group(0), l)
        out_body.append(l)
    body = out_body
    first, cond, region_of = {}, None, {}
    for n_, l in enumerate(body):
        t_ = l.strip()
        if t_.startswith("^tbuild") and not INLINE[0]:          # inline: the table build is unconditional
            cond = (max(i for i in range(n_) if body[i].strip().startswith("low.cond_br")), "^tdone")
        elif cond and t_.startswith(cond[1]):
            cond = None
        if owner[n_] is not None:
            continue
        for nm in re.findall(r"%k[\w]+", l):
            if nm not in defs:
                continue
            region_of.setdefault(nm, set()).add(cond[0] if cond else None)
            first.setdefault(nm, cond[0] if cond else n_)
    for nm, regs in region_of.items():
        if len(regs) > 1:
            first[nm] = min(first[nm], min(r for r in regs if r is not None))
    at = {}
    for nm, n_ in first.items():
        at.setdefault(n_, []).append(defs[nm])
    # a loop-local copy right before its first use among the loop's own lines (blocks of a loop body run in order)
    for lp, names in local.items():
        for nm in names:
            loc = f"{nm}_L{lp}"
            n_ = next(i for i, l in enumerate(body) if owner[i] == lp and re.search(re.escape(loc) + r"\b", l))
            at.setdefault(n_, []).append(defs[nm].replace(nm + " ", f"{loc} ", 1))
    out = []
    for n_, l in enumerate(body):
        if l == "// LOOP-BEGIN":
            out += at.get(n_, [])
            continue
        out += at.get(n_, [])
        if not l.startswith("// LOOP"):
            out.append(l)
    return out

def array(units, nports=1):
    S = "reg<aie2p.array.scalar : index>"
    RS, NS, P = 20 * BLK // 4, 5, units // 5           # row stride (words): K / 256 super-blocks of BLK bytes
    return f"""aie2p.target<array> @array_target
aie2p.target<core> @core_target

low.func.def public retain target<amd.xdna.aie2p.array>(@array_target) abi(array_program) @probe() asm {{
  %n0 = constant.u32 0 : {S}
  %n1 = constant.u32 1 : {S}
  %n2 = constant.u32 2 : {S}
  %rec = constant.u32 {units} : {S}
  %orec = constant.u32 {8 * units} : {S}
  %origin = constant.u64 0 : reg<aie2p.array.offset : offset>
  %workers = group %n1
  %k = worker %workers, %n0, @dec
  constrain.location %k, %n0, %n2
  %ib = binding 0, "read"
  %ob = binding 1, "write"
  %np = constant.u32 {nports} : {S}
  %o_all = receiver %ob, 0 : reg<aie2p.array.receiver : tile<{nports}x{8 * units}x{OUT_B // 4}xi32>>
""" + "".join(f"""  %io{p} = constant.u64 {p * 80 * RS * 4} : reg<aie2p.array.offset : offset>
  %is{p} = sender %ib, 0 : reg<aie2p.array.sender : tile<{P}x{NS}x16x{ROWB // 4}xi32, #encoding.layout.strided<strides=[{BLK}, {16 * RS}, {RS}, 1]>>>
  %iv{p} = view.sender %is{p}, %io{p} : reg<aie2p.array.sender : tile<16x{ROWB // 4}xi32>>
  %ri{p} = receiver %k, {p} : reg<aie2p.array.receiver : tile<16x{ROWB // 4}xi32>>
  %chi{p} = channel %iv{p}, %ri{p}, %n1, %rec : reg<aie2p.array.channel : tile<16x{ROWB // 4}xi32>>
  %orc{p} = partition.receiver %o_all, %origin, %n{p}, %np : reg<aie2p.array.receiver : tile<{OUT_B // 4}xi32>>
  %so{p} = sender %k, {nports + p} : reg<aie2p.array.sender : tile<{OUT_B // 4}xi32>>
  %cho{p} = channel %so{p}, %orc{p}, %n2, %orec : reg<aie2p.array.channel : tile<{OUT_B // 4}xi32>>
  constrain.leaf_sync %cho{p}
""" for p in range(nports)) + """  return
}
"""


# panel position -> k-block within a pass: per sub-block of 4, (0, 2, 1, 3)
KPERM = (np.array([8 * (p // 8) + (0, 4, 1, 5, 2, 6, 3, 7)[p % 8] for p in range(128)])
         if FMT == "Q4_K" or (Q3K and GORD == "PK") else
         np.arange(128) if (GRID or Q3K) and not GP4 else np.array([4 * (p // 4) + (0, 2, 1, 3)[p % 4] for p in range(128)]))
# Q4_K: chunk c of a 32-byte group pushes low nibbles (sub-block 2 g, kb c) then high (2 g + 1, kb 4 + c)


def parts2(v):
    """f32 v -> the kernel's two bf16 parts (RNE, then the RNE of the exact remainder)."""
    p0 = Q.bf16_rne(v.astype(np.float32))
    return p0, Q.bf16_rne((v.astype(np.float32) - p0).astype(np.float32))


def oracle(raw):
    """raw [80 rows][20 super-blocks][BLK] -> panel-order fragments [pass][slab][kb 128][h 2][72]."""
    w = weights(raw)
    w = w.reshape(80, 640, 8)                                     # [row][kb][8]
    frag = w.reshape(5, 2, 8, 5, 128, 8).transpose(3, 0, 4, 1, 2, 5)   # [pass][slab][kb][h][row][8]
    frag = frag[:, :, KPERM]                                      # the panel's k-block order
    return Q.bfp_hw(frag.reshape(-1, 8, 8)).reshape(5, 5, 128, 2, 72)


def q3k_fields(blk):
    """blk [..., 110] u8 -> d (f64), sc [..., 16] (6 bit), q [..., 256] in -4 .. 3 (llama.cpp block_q3_K)."""
    d = blk[..., 108:110].copy().view(np.float16)[..., 0].astype(np.float64)
    s = blk[..., 96:108].astype(np.int64)
    sc = np.stack([(s[..., i] & 15 if i < 8 else s[..., i - 8] >> 4) | (((s[..., 8 + i % 4] >> (2 * (i // 4))) & 3) << 4)
                   for i in range(16)], -1)
    hm, qs = blk[..., 0:32].astype(np.int64), blk[..., 32:96].astype(np.int64)
    q = np.stack([((qs[..., 32 * (e // 128) + e % 32] >> (2 * ((e % 128) // 32))) & 3)
                  + 4 * ((hm[..., e % 32] >> (4 * (e // 128) + (e % 128) // 32)) & 1) - 4 for e in range(256)], -1)
    return d, sc, q


def weights(raw):
    """raw [rows][super-blocks][BLK] -> the decoder's f32 weights [rows][super-blocks][256] (its exact op order)."""
    if XS2:                                             # w = d (2 ls + 1) (grid / 8)[q & 511][j] ksign(q >> 9): exact
        grid = np.fromfile(os.path.join(TABLES, "grid_iq2xs.bin"), np.uint8).reshape(512, 8).astype(np.float64) / 8
        d = raw[..., 0:2].copy().view(np.float16)[..., 0].astype(np.float64)
        q = raw[..., 2:66].copy().view(np.uint16).astype(np.int64)
        sb = raw[..., 66:74].astype(np.int64)
        ls = np.stack([(sb[..., i // 2] >> (4 * (i % 2))) & 15 for i in range(16)], -1)
        f = q >> 9
        par = np.zeros_like(f)
        for b in range(7):
            par ^= (f >> b) & 1
        ks = f | (par << 7)
        sign = np.stack([1 - 2 * ((ks[..., v // 8] >> (v % 8)) & 1) for v in range(256)], -1)
        g = grid[q & 511].reshape(raw.shape[:-1] + (256,))
        return (np.repeat(d[..., None] * (1 + 2 * ls), 16, -1) * g * sign).astype(np.float32)
    if X2:                                              # w = d (2 s + 1) (grid / 8)[qs][j] ksign: exact in f32
        grid = np.fromfile(os.path.join(TABLES, "grid_iq2xxs.bin"), np.uint8).reshape(256, 8).astype(np.float64) / 8
        d = raw[..., 0:2].copy().view(np.float16)[..., 0].astype(np.float64)
        sb = raw[..., 2:66].reshape(raw.shape[:-1] + (8, 8))
        qs = sb[..., 0:4].reshape(raw.shape[:-1] + (32,)).astype(np.int64)
        aux = sb[..., 4:8].copy().view(np.uint32)[..., 0].astype(np.int64)
        sc = aux >> 28
        f = np.stack([(aux >> (7 * l)) & 127 for l in range(4)], -1)
        par = np.zeros_like(f)
        for b in range(7):
            par ^= (f >> b) & 1
        ks = f | (par << 7)
        sign = np.stack([1 - 2 * ((ks[..., v // 32, (v // 8) % 4] >> (v % 8)) & 1) for v in range(256)], -1)
        g = grid[qs].reshape(raw.shape[:-1] + (256,))
        return (np.repeat(d[..., None] * (1 + 2 * sc), 32, -1) * g * sign).astype(np.float32)
    if XXS:                                             # w = d (2 s + 1) (grid / 4)[qs][j] ksign: exact in f32
        grid = grid_table("grid_iq3xxs.bin", 256) / 4
        d = raw[..., 0:2].copy().view(np.float16)[..., 0].astype(np.float64)
        qs = raw[..., 2:66].astype(np.int64)
        aux = raw[..., 66:98].copy().view(np.uint32).astype(np.int64)
        sc = aux >> 28
        f = np.stack([(aux >> (7 * l)) & 127 for l in range(4)], -1)
        par = np.zeros_like(f)
        for b in range(7):
            par ^= (f >> b) & 1
        ks = f | (par << 7)
        sign = np.stack([1 - 2 * ((ks[..., v // 32, (v // 8) % 4] >> (v % 8)) & 1) for v in range(256)], -1)
        g = grid[qs].reshape(raw.shape[:-1] + (256,))
        return (np.repeat(d[..., None] * (1 + 2 * sc), 32, -1) * g * sign).astype(np.float32)
    if FMT == "IQ3_S":                                  # w = d (1 + 2 s) grid[qs | qh bit << 8][j] sign: exact in f32
        grid = grid_table("grid_iq3s.bin", 512)
        d = raw[..., 0:2].copy().view(np.float16)[..., 0].astype(np.float64)
        qs = raw[..., 2:66].astype(np.int64)
        qh = raw[..., 66:74].astype(np.int64)
        sg = raw[..., 74:106].astype(np.int64)
        scb = raw[..., 106:110].astype(np.int64)
        sc = np.stack([(scb[..., i // 2] >> (4 * (i % 2))) & 15 for i in range(8)], -1)
        hb = np.stack([(qh[..., n // 8] >> (n % 8)) & 1 for n in range(64)], -1)
        g = grid[qs | (hb << 8)].reshape(raw.shape[:-1] + (256,))
        sign = np.stack([1 - 2 * ((sg[..., v // 8] >> (v % 8)) & 1) for v in range(256)], -1)
        return (np.repeat(d[..., None] * (1 + 2 * sc), 32, -1) * g * sign).astype(np.float32)
    if Q3K:                                             # w = d (sc - 32) (crumb + 4 hbit - 4): exact in f32
        d, sc, q = q3k_fields(raw)
        return (np.repeat(d[..., None] * (sc - 32), 16, -1) * q).astype(np.float32)
    if FMT == "Q4_K":
        # T = d sc, Mn = -dmin m (each two bf16 parts); X = q; f32 accumulation in the kernel's order:
        # m = Mn0 + Mn1, w = (m + T0 X) + T1 X
        d, dm, sc, mn, q = Q.q4k_fields(raw)
        f = np.float32
        T0, T1 = parts2((d[..., None] * sc).astype(f))
        M0, M1 = parts2((-dm[..., None] * mn).astype(f))
        rep_ = lambda v: np.repeat(v, 32, -1)
        X = q.astype(f)
        m = (rep_(M0) + rep_(M1)).astype(f)
        w = (m + rep_(T0) * X).astype(f)
        w = (w + rep_(T1) * X).astype(f)
    else:
        d, ls, q = I.iq4xs_fields(raw)
        w = (np.repeat(d[..., None] * (ls - 32), 32, -1) * np.array(KV, np.float64)[q]).astype(np.float32)
    return w


def main():
    work = sys.argv[1]
    units = 25
    nports = int(os.environ.get("DQ_PORTS", "1"))
    os.makedirs(work, exist_ok=True)
    src = array(units, nports) + "\n" + leaf("dec", nports) + "\n"
    open(f"{work}/p.loom", "w").write(src)
    LC = os.environ.get("LC", hrx_paths.LOOM_COMPILE)
    env = dict(hrx_paths.env(), LOOM_EXP_LOCKED_PACK="1")
    r = subprocess.run([LC, f"{work}/p.loom", "--root=@probe", "--target=amd.xdna.aie2p:amd.xdna.strix_halo.17f0_11",
                        f"--output={work}/p.xdna", "--compile-report=text", f"--compile-report-output={work}/report.txt"],
                       capture_output=True, text=True, env=env)
    if r.returncode:
        sys.exit("compile failed\n" + r.stderr[-4000:])
    for l in open(f"{work}/report.txt"):
        if l.startswith("COMPILE-REPORT: emission"):
            print("compiled:", " ".join(t for t in l.split() if t.split("=")[0] in ("instructions", "code_bytes")))
    if os.environ.get("DQ_NORUN"):
        return
    import gguf   # llama.cpp's gguf-py on PYTHONPATH
    rd = gguf.GGUFReader(sys.argv[2])
    t = next(x for x in rd.tensors if x.tensor_type.name == FMT and len(x.shape) == 2 and int(x.shape[0]) == 5120
             and int(x.shape[1]) >= 1024)
    raw = np.asarray(t.data).view(np.uint8).reshape(-1, 20, BLK)[160:160 + 80 * nports].copy()   # 80 rows per port
    np.concatenate([raw.reshape(-1), np.zeros(1024, np.uint8)]).tofile(f"{work}/in.bin")   # + slack: records over-read
    np.zeros(nports * 8 * units * OUT_B, np.uint8).tofile(f"{work}/o0.bin")
    r = subprocess.run([hrx_paths.XDNA_RUN, "--columns=1", f"--image={work}/p.xdna", "--entry=probe", "--binding_memory=system",
                        f"--binding={work}/in.bin", f"--binding={work}/o0.bin", f"--output=1={work}/out.bin"],
                       capture_output=True, text=True, env=env, timeout=60)
    if r.returncode:
        sys.exit("run failed\n" + r.stdout[-1500:] + r.stderr[-1500:])
    got = np.fromfile(f"{work}/out.bin", np.uint8).reshape(nports, 5, 5, 128, 2, 72)
    ref = np.stack([oracle(raw[80 * p:80 * p + 80]) for p in range(nports)])
    bad = np.any(got != ref, axis=-1)
    print(f"fragments differing: {int(bad.sum())} of {bad.size}")
    if bad.any():
        i = np.argwhere(bad)[0]
        print("first bad [pass, slab, kb, h]:", i.tolist(), "\n got", got[tuple(i)][:18].tolist(), "\n ref", ref[tuple(i)][:18].tolist())




def port_copy(sec, p, nports):
    """Port p's section: port 0's with every value / label defined in it suffixed, port 0's resources swapped for
    port p's, and the output ring's acquire / release index moved."""
    import re
    defined = set()
    for l in sec:
        t = l.strip()
        m = re.match(r"((?:%[\w.]+)(?:, %[\w.]+)*) = ", t)
        if m:
            defined.update(re.findall(r"%[\w.]+", m.group(1)))
        m = re.match(r"\^([\w]+)\((.*)\):", t)
        if m:
            defined.update(re.findall(r"(%[\w.]+):", m.group(2)))
    labels = {m for l in sec for m in re.findall(r"\^([\w]+)", l)}
    swap = {"%in": f"%in{p}", "%out": f"%out{p}", "%ina": f"%ina{p}", "%outa": f"%outa{p}"}
    sfx = f"_p{p}"

    def fix(l):
        l = re.sub(r"%[\w.]+", lambda m: swap.get(m.group(0), m.group(0) + sfx if m.group(0) in defined else m.group(0)), l)
        l = re.sub(r"\^([\w]+)", lambda m: "^" + m.group(1) + (sfx if m.group(1) in labels else ""), l)
        l = re.sub(r"^(\s*(?:acq|rel) %\w+, )" + str(nports) + r"$", lambda m: m.group(1) + str(nports + p), l)
        l = l.replace("{index = 0, source_type", f"{{index = {p}, source_type").replace(
            f"{{index = {nports}, source_type", f"{{index = {nports + p}, source_type")
        return l
    return [fix(l) for l in sec]


if __name__ == "__main__":
    main()
