#!/usr/bin/env python3
"""Emit the NPU Q4_K weight decoder leaf (XDNA2 core, Loom low asm): raw GGUF blocks in, the cascade GEMM's BFP16 out.

One call decodes an 8-row group of one 256-value super-block.
Input record (1152 B): headers [row][16 B] (d, dmin, 12 scale bytes) then qs [chunk 16][row][8 B] (the DMA transposes rows
into 8-byte chunks, so one 64-byte load holds 8 rows x 8 k-values of two sub-blocks: lo and hi nibbles).
Output record (2304 B): 32 bfp16ebs8 fragments (8 rows x 8 k, [E, m0..m7] per row) in k-block order, the record format
of gen_bfp16_encode's weight stream.

Numerics: w = (C' + T1 X) + T2 X in f32 with X = 128 + q (bf16 0x43qq), T = d sc split into two bf16 parts and
C' = f32(-128 (T1 + T2) - M), M = dmin m; every product is exact, so one f32 rounding per add, then the hardware
bfp16 conversion (E = exponent of the block max, +1 only if a rounded mantissa leaves int8; subnormals flush).

Scale side, per call: the scalar unit splits the 12 scale bytes per row (SWAR) into sc / m bytes [row][f][s]; three byte
deinterleaves transpose them to [s % 4][row][f], one 64-lane block per 4 sub-blocks. The f16 d / dmin become, in 16-bit
lanes, U = 128 + u, V = 128 + v (significand 128 u + v, subnormals included) and an offset (e' - 25) << 7 | sign that
scales a bf16 operand by P = +-2^(e' - 25). Six exact bf16 MACs give T and M per block:
T = U (128 P S) + V (P S) - U (16384 P) - V (128 P) - S (16512 P) + (2^21 + 2^14) P, S = 128 + sc.
Per sub-block, Tb_1, Tb_2 (row-broadcast parts) and C' go to scratch; each fragment then loads them and runs 2 MACs.
Ops are merged by xdna_sched (compile with LOOM_EXP_LOCKED_PACK=1 so the locked leaf co-issues).
"""
import sys

import xdna_sched

IN_B, OUT_B = 1152, 2304
V1, V2, V4 = "reg<aie2p.vec256>", "reg<aie2p.vec256 x2>", "reg<aie2p.vec256 x4>"
M1, M2, M4 = "reg<aie2p.mbms>", "reg<aie2p.mbms x2>", "reg<aie2p.mbms x4>"
# scratch: d / dmin tables, d | dmin words, sc / m bytes, T / M bf16 parts, per sub-block Tb_1, Tb_2 (128 B each), C'
TAB, HB, SCM, PARTS, CC = 0, 128, 192, 384, 1920
SCR_B = CC + 8 * 512


class Emit:
    """Low asm lines with SSA temporaries; stream() makes an independent op list for xdna_sched to merge.
    Constants go to the root's preamble; each stream advances its own pointer cursors."""

    def __init__(self, root=None):
        self.L = []
        self.root = root or self
        self.cursors = {}
        if root is None:
            self.n = 0
            self.consts = {}
            self.pre = []

    def __call__(self, s):
        self.L.append("  " + s)

    def t(self, p="t"):
        self.root.n += 1
        return f"%{p}{self.root.n}"

    def const(self, v):
        nm = f"%k{v}".replace("-", "m")
        if nm not in self.root.consts:
            self.root.consts[nm] = True
            self.root.pre.append(f"  {nm} = {'mova.i32' if -1024 <= v <= 1023 else 'mov.i32'} {v}")
        return nm

    def stream(self):
        return Emit(self.root)

    def advance(self, cur, delta):
        """cur + delta; pointer adds take multiples of 64 as immediates, the rest through a modifier."""
        left = delta - delta % 64
        while left:
            st = max(-512, min(448, left))
            nxt = self.t("pb")
            self(f"{nxt} = padda {cur}, {st}")
            cur, left = nxt, left - st
        if delta % 64:
            md, nxt = self.t("md"), self.t("pb")
            self(f"{md} = mov.modifier {self.const(delta % 64)}")
            self(f"{nxt} = padda.modifier {cur}, {md}")
            cur = nxt
        return cur

    def addr(self, base, off, cursor, hi=448, step=64):
        """(pointer, immediate) for base + off through a cursor that only moves forward (padda consumes its input)."""
        if cursor not in self.cursors:
            cur = self.t("pc")
            self(f"{cur} = copy {base} : reg<aie2p.ep> -> reg<aie2p.ep>")
            self.cursors[cursor] = (cur, 0)
        cur, at = self.cursors[cursor]
        if -(hi + step) <= off - at <= hi and (off - at) % step == 0:
            return cur, off - at
        tgt = off - off % step
        self.cursors[cursor] = (self.advance(cur, tgt - at), tgt)
        return self.cursors[cursor][0], off - tgt


def gen_q4k(name="dq_q4k"):
    """The decoder leaf; resources 0 / 1 are the input / output records."""
    e = Emit()
    e.L.append(f"low.func.def schedule(locked) target<amd.xdna.aie2p.core>(@core_target) abi(object_function) @{name}() asm {{")
    e("%in = resource<native_pointer> {index = 0, source_type = buffer} : reg<aie2p.ep>")
    e("%out = resource<native_pointer> {index = 1, source_type = buffer} : reg<aie2p.ep>")
    e(f"%scr = storage {{byte_alignment = 64, byte_length = {SCR_B}}} : low.storage<private>")
    e("%sp = storage_address %scr : low.storage<private> -> reg<aie2p.ep>")
    e("set.unpack-size 0")
    e("set.rounding 12")   # round to nearest even
    hdr_end = len(e.L)
    k = e.const
    conf = k(60)
    m0c, m1c, m20, m21, m52, m53 = k(0), k(1), k(20), k(21), k(52), k(53)
    win = {}

    def splat16(x, v):
        r = x.t("sp")
        x(f"{r} = vbcst.16 {k(v)}")
        return r

    def wide(x, x2):
        """vec256 x2 -> x4 by repeating."""
        c, r4 = x.t("cp"), x.t("w4")
        x(f"{c} = vmov.512 {x2}")
        x(f"{r4} = concat({x2}, {c}) : ({V2}, {V2}) -> {V4}")
        return r4

    def col16(x, off):
        """The [row][f] 16-lane int16 table at TAB + off repeated to 32 lanes."""
        l0, l1, r2 = x.t("c"), x.t("c"), x.t("c2")
        pp, oo = x.addr("%sp", TAB + off, "tab", 224, 32)
        x(f"{l0} = vlda.256.i16x16 {pp}, {oo}")
        x(f"{l1} = vlda.256.i16x16 {pp}, {oo}")
        x(f"{r2} = concat({l0}, {l1}) : ({V1}, {V1}) -> {V2}")
        return r2

    def add16(x, a, b):
        r = x.t("ad")
        x(f"{r} = vadd.16 {a}, {b}")
        return r

    def sub16(x, a, b):
        r = x.t("sb")
        x(f"{r} = vsub.16 {a}, {b}")
        return r

    def band(x, a, v):
        r = x.t("an")
        x(f"{r} = vband {a}, {splat16(x, v)}")
        return r

    def mac(x, acc, s1, s2):
        r = x.t("ac")
        if acc is None:
            x(f"{r} = mmul.bf16bf16.m8n8k1 {s1}, {s2}, {conf}")
        else:
            x(f"{r} = mma.bf16bf16.m8n8k1 {acc}, {s1}, {s2}, {conf}")
        return r

    def conv(x, acc):
        """f32 x64 -> bf16 x64: (x4, (half0, half1))."""
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

    def shr8(x, a):
        """16-bit lanes >> 8: the odd bytes widened with zero."""
        o, r = x.t("o"), x.t("r8")
        x(f"{o} = vshuffle {a}, {a}, {m1c}")
        x(f"{r} = vshuffle {o}, %z8, {m20}")
        return r

    def window(x, base, at, name):
        """A pointer to base + at shared by streams, defined by the first stream that needs it."""
        if (name, at) not in win:
            cur = x.t("pw")
            x(f"{cur} = copy {base} : reg<aie2p.ep> -> reg<aie2p.ep>")
            win[(name, at)] = (x.advance(cur, at), at)
        return win[(name, at)]

    def rel(fp, off, hi=448, step=64):
        o = off - fp[1]
        assert -(hi + step) <= o <= hi and o % step == 0, (off, fp)
        return fp[0], o

    def run(streams, window_, tails=None, ext=(0, 0, 0)):
        e.L.extend(xdna_sched.schedule([x.L for x in streams], window_, tails, ext))

    def rows(x, s, which, p, fp, before=None):
        """Row-broadcast bf16 x64 of part p of T (which 0) or M (1) for sub-block s; before(lo, hi) sees the halves."""
        blk, sl = s // 4, s % 4
        pp, o = rel(fp, PARTS + (((3 * blk + p) * 2 + which) * 2 + sl // 2) * 64)
        ld, bc, ra, rb, r4 = x.t("pl"), x.t("bc"), x.t("ra"), x.t("rb"), x.t("tb")
        x(f"{ld} = vlda.512.bf16x32 {pp}, {o}")
        x(f"{bc} = vbroadcast.bf16x8.to.bf16x32 {ld}, {sl % 2}")
        x(f"{ra} = vshuffle {bc}, {bc}, {m52}")
        x(f"{rb} = vshuffle {bc}, {bc}, {m53}")
        if before:
            before(ra, rb)
        x(f"{r4} = concat({ra}, {rb}) : ({V2}, {V2}) -> {V4}")
        return r4

    # scalar: SWAR scale split per row into SCM [row][f][s], d | dmin words into HB [row]
    sts = []
    for r in range(8):
        x = e.stream()
        sts.append(x)
        fin = window(x, "%in", 32 * (r // 2), "hdr")
        fsc = window(x, "%sp", SCM + 32 * (r // 2), "scm")
        fhb = window(x, "%sp", HB, "hb")
        w = [x.t("w") for _ in range(4)]
        for i in range(4):
            x(f"{w[i]} = lda {fin[0]}, {16 * (r % 2) + 4 * i}")
        x(f"st {w[0]}, {fhb[0]}, {4 * r}")
        sc0, m0, t1, t2, u1, sc1, t3, t4, u2, u3, m1 = (x.t("s") for _ in range(11))
        x(f"{sc0} = and {w[1]}, {k(0x3F3F3F3F)}")
        x(f"{m0} = and {w[2]}, {k(0x3F3F3F3F)}")
        x(f"{t1} = lshl {w[1]}, {k(-2)}")
        x(f"{t2} = and {t1}, {k(0x30303030)}")
        x(f"{u1} = and {w[3]}, {k(0x0F0F0F0F)}")
        x(f"{sc1} = or {u1}, {t2}")
        x(f"{t3} = lshl {w[2]}, {k(-2)}")
        x(f"{t4} = and {t3}, {k(0x30303030)}")
        x(f"{u2} = lshl {w[3]}, {k(-4)}")
        x(f"{u3} = and {u2}, {k(0x0F0F0F0F)}")
        x(f"{m1} = or {u3}, {t4}")
        for i, val in enumerate((sc0, sc1, m0, m1)):
            x(f"st {val}, {fsc[0]}, {16 * (r % 2) + 4 * i}")
    run(sts, 8)

    # transpose (sc, m) [row][f][s] -> [s][row][f]; d / dmin fields in 16-bit lanes [row][f]
    e(f"%k43 = vbcst.8 {k(0x43)}")
    e(f"%z8 = vbcst.8 {k(0)}")
    pp, oo = e.addr("%sp", SCM, "scm_v")
    e(f"%sa = vlda.512.i8x64 {pp}, {oo}")
    e(f"%sb = vlda.512.i8x64 {pp}, {oo + 64}")
    pp, oo = e.addr("%sp", HB, "hb_v")
    e(f"%hv = vlda.512.i16x32 {pp}, {oo}")
    xt, xd = e.stream(), e.stream()
    a, b = "%sa", "%sb"
    for _ in range(3):
        na, nb = xt.t("tr"), xt.t("tr")
        xt(f"{na} = vshuffle {a}, {b}, {m0c}")
        xt(f"{nb} = vshuffle {a}, {b}, {m1c}")
        a, b = na, nb
    e4 = band(xd, shr8(xd, "%hv"), 0x7C)
    e128 = e4
    for _ in range(5):
        e128 = add16(xd, e128, e128)
    ep128 = xd.t("ep")
    xd(f"{ep128} = max.u16x32 {e128}, {splat16(xd, 128)}")
    offv = add16(xd, sub16(xd, ep128, splat16(xd, 3200)), band(xd, "%hv", 0x8000))
    z128 = sub16(xd, ep128, e128)
    z1024 = add16(xd, add16(xd, z128, z128), add16(xd, z128, z128))
    z1024 = add16(xd, z1024, z1024)
    sd = add16(xd, band(xd, "%hv", 0x3FF), sub16(xd, splat16(xd, 1024), z1024))
    uv = add16(xd, shr8(xd, add16(xd, sd, sd)), splat16(xd, 0x4300))
    vv = add16(xd, band(xd, sd, 127), splat16(xd, 0x4300))
    for val, o16 in ((uv, 0), (vv, 32), (offv, 64)):
        lo = xd.t("lo")
        xd(f"{lo} = slice {val}[0] : {V2} -> {V1}")
        pp, oo = xd.addr("%sp", TAB + o16, "tab", 224, 32)
        xd(f"vst.256.i16x16 {lo}, {pp}, {oo}")
    run([xt, xd], 2)

    # T = d sc, M = dmin m per block (lanes [s % 4][row][f]), split into three bf16 parts
    bl = []
    for blk, reg in enumerate((a, b)):
        x = e.stream()
        bl.append(x)

        def sreg(x=x, reg=reg):
            slo, shi = x.t("S"), x.t("S")
            x(f"{slo} = vshuffle {reg}, %k43, {m20}")
            x(f"{shi} = vshuffle {reg}, %k43, {m21}")
            return slo, shi

        def offc(v, x=x):
            return add16(x, col16(x, 64), splat16(x, v))

        def scaled(v, x=x, sreg=sreg, offc=offc):
            slo, shi = sreg()
            ov = offc(v)
            r4 = x.t("A")
            x(f"{r4} = concat({add16(x, slo, ov)}, {add16(x, shi, ov)}) : ({V2}, {V2}) -> {V4}")
            return r4

        acc = mac(x, None, wide(x, col16(x, 0)), scaled(0x380))
        acc = mac(x, acc, wide(x, col16(x, 32)), scaled(0))
        acc = mac(x, acc, wide(x, col16(x, 0)), wide(x, offc(0xC680)))
        acc = mac(x, acc, wide(x, col16(x, 32)), wide(x, offc(0xC300)))
        slo, shi = sreg()
        s4 = x.t("S4")
        x(f"{s4} = concat({slo}, {shi}) : ({V2}, {V2}) -> {V4}")
        acc = mac(x, acc, s4, wide(x, offc(0xC681)))
        acc = mac(x, acc, wide(x, splat16(x, 0x3F80)), wide(x, offc(0x4A01)))
        for p in range(3):
            part, halves = conv(x, acc)
            for h, hh in enumerate(halves):
                for f in range(2):
                    dv = x.t("dv")
                    x(f"{dv} = vshuffle {hh}, {hh}, {k(2 + f)}")
                    pp, o = x.addr("%sp", PARTS + (((3 * blk + p) * 2 + f) * 2 + h) * 64, "parts_w")
                    x(f"vst.512.bf16x32 {dv}, {pp}, {o}")
            if p < 2:
                acc = mac(x, acc, part, wide(x, splat16(x, 0xBF80)))
    run(bl, 1, ext=(6, 0, 0))

    # per sub-block: Tb_1, Tb_2 and C' = f32(-128 (T1 + T2) - M1 - M2 - M3) into scratch
    kcs = [wide(e, splat16(e, 0xC300)), wide(e, splat16(e, 0xBF80))]
    chains = []
    for s in range(8):
        x = e.stream()
        chains.append(x)
        fpp = window(x, "%sp", PARTS + 768 * (s // 4) + 384, "parts")
        fcc = window(x, "%sp", CC + 512 * (2 * (s // 2)) + 512, "cc")
        acc = None
        for which, p in ((0, 0), (0, 1), (1, 0), (1, 1), (1, 2)):
            def st_tb(ra, rb, x=x, s=s, p=p, fcc=fcc):
                pp, o = rel(fcc, CC + 512 * s + 128 * p)
                x(f"vst.512.bf16x32 {ra}, {pp}, {o}")
                x(f"vst.512.bf16x32 {rb}, {pp}, {o + 64}")
            acc = mac(x, acc, rows(x, s, which, p, fpp, st_tb if which == 0 else None), kcs[which])
        for i in range(4):
            qq = x.t("cq")
            x(f"{qq} = slice {acc}[{i}] : {M4} -> {M1}")
            pp, o = rel(fcc, CC + 512 * s + 256 + 64 * i)
            x(f"vst.acc {qq}, {pp}, {o}")
    run(chains, 4, ext=(8, 0, 0))

    # fragments: X = 128 + q, acc = C' + Tb_1 X + Tb_2 X, pushed in k-block order
    e("%op = copy %out : reg<aie2p.ep> -> reg<aie2p.mpfs>")
    e("%sl = vlda.store-fifo.low512 %out, 0")
    e("%f0 = vlda.store-fifo.high512 %out, %sl, 64")
    e("%pos = mova.fifo.store.position 0")
    fifo = ["%f0", "%op", "%pos"]
    frags, tails = [], []
    for s in range(8):
        j = s // 2
        for c2 in range(4):
            x = e.stream()
            frags.append(x)
            qoff = 128 + 256 * j + 64 * c2
            fq = window(x, "%in", 384 + 256 * j, "qs")
            fc = window(x, "%sp", CC + 512 * (2 * (s // 2)) + 512, "cc_hot")
            nib, xl, xh, x4 = (x.t("x") for _ in range(4))
            if s % 2 == 0:
                # lo nibbles: the masked [row][8 B] bytes are already the fragment
                raw, msk = x.t("qr"), x.t("mk")
                pp, o = rel(fq, qoff)
                x(f"{raw} = vlda.512.i8x64 {pp}, {o}")
                x(f"{msk} = vbcst.8 {k(15)}")
                x(f"{nib} = vband {raw}, {msk}")
            else:
                ua, ub = x.t("x"), x.t("x")
                pp, o = rel(fq, qoff, 224, 32)
                x(f"{ua} = vldb.unpack.u4.to.u8x64 {pp}, {o}")
                x(f"{ub} = vldb.unpack.u4.to.u8x64 {pp}, {o + 32}")
                x(f"{nib} = vshuffle {ua}, {ub}, {m1c}")
            cq = []
            for i in range(4):
                qq = x.t("cl")
                pp, o = rel(fc, CC + 512 * s + 256 + 64 * i)
                x(f"{qq} = vlda.acc {pp}, {o}")
                cq.append(qq)
            acc = x.t("C")
            x(f"{acc} = concat({', '.join(cq)}) : ({M1}, {M1}, {M1}, {M1}) -> {M4}")
            k43 = x.t("k43")
            x(f"{k43} = vbcst.8 {k(0x43)}")
            x(f"{xl} = vshuffle {nib}, {k43}, {m20}")
            x(f"{xh} = vshuffle {nib}, {k43}, {m21}")
            x(f"{x4} = concat({xl}, {xh}) : ({V2}, {V2}) -> {V4}")
            for p in range(2):
                pp, o = rel(fc, CC + 512 * s + 128 * p)
                h0, h1, t4 = x.t("bl"), x.t("bl"), x.t("TB")
                x(f"{h0} = vldb.512.bf16x32 {pp}, {o}")
                x(f"{h1} = vldb.512.bf16x32 {pp}, {o + 64}")
                x(f"{t4} = concat({h0}, {h1}) : ({V2}, {V2}) -> {V4}")
                acc = mac(x, acc, t4, x4)
            nf = len(frags)
            t = [f"  %f{nf}, %p{nf}, %q{nf} = vst.push.bfp16ebs8.from.fp32 {fifo[0]}, {acc}, {fifo[1]}, {fifo[2]}"]
            fifo = [f"%f{nf}", f"%p{nf}", f"%q{nf}"]
            if nf % 8 == 0 and nf < 32:
                # 8 pushes of 576 bits leave a whole line in the FIFO: drain it or the next push overflows
                t.append(f"  %g{nf}, %h{nf}, %j{nf} = vst.flush.512 {fifo[0]}, {fifo[1]}, {fifo[2]}")
                fifo = [f"%g{nf}", f"%h{nf}", f"%j{nf}"]
            tails.append(t)
    run(frags, 4, tails)
    e(f"%ff, %fp, %fq = vst.flush.512 {fifo[0]}, {fifo[1]}, {fifo[2]}")
    e("return")
    e.L.append("}")
    return "\n".join(e.L[:hdr_end] + e.root.pre + e.L[hdr_end:])


if __name__ == "__main__":
    sys.stdout.write(gen_q4k(sys.argv[1] if len(sys.argv) > 1 else "dq_q4k") + "\n")
