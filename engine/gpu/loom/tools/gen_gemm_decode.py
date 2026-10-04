#!/usr/bin/env python3
"""Weight decoders for the tile GEMM (tools/gen_gemm_tile.py): per format, the loads and the branch-free f32 decode of
one K phase into f16 LDS rows, and the format table FMTS (block bytes, phase width, extra bindings).
The f32 decode ops (one f16 rounding) match the reference dequant, so the GEMMs are bit-identical to it.
gen_gemm_tile.configure() sets the per-format switches and the geometry below before it emits a kernel.
"""

NW = 2                      # decoding waves per workgroup
LR = 64                     # weight rows per decoded tile
PAD = 0                     # f16 of padding per LDS weight row
# Per-format decode variants; gen_gemm_tile.configure() turns them on per format.
# Q4_HDR: the Q4_K/Q5_K header (d, dmin, scales[12]) as one 16-byte load instead of byte loads that each carry an address clamp.
Q4_HDR = False
# VDEC_W: IQ3/IQ2 sign application on the two grid words instead of i8 vectors.
VDEC_W = False
# IQ3_U8F (IQ3_S / IQ3_XXS, word path): mag bytes XOR 0x80 are u = mag + 128 as unsigned bytes, converted by v_cvt_f32_ubyteN.
# The -128 rides the f32 addend: fptrunc(fma(dsc, u, -128*dsc)).
# (u-128)*dsc has <= 24 significant bits (|mag| <= 127, dsc = d * odd <= 5 bits), so it is exact in f32: bit-identical.
IQ3_U8F = False
# VDECW_FR: the word path's sign spread without quarter-rate v_mul_lo_u32.
# nibble * 0x00204081 becomes shifts and ORs, s1 * 255 becomes (s1 << 8) - s1 (top byte wraps). Same integers: bit-identical.
VDECW_FR = False
# IQ3_SGN2 (with IQ3_U8F + VDECW_FR): u = (g ^ (0x80808080 - s1)) + s1 straight from the grid word, and the sign spread as
# two shift-adds (n | n << 7, t | t << 14: the bits are disjoint, so OR is the add). Per byte, s = 0: g ^ 0x80 = g + 128;
# s = 1: (g ^ 0x7f) + 1 = 128 - g (grid magnitudes 1..127: no borrow, no carry). Same bytes as ((g ^ -s) + s) ^ 0x80.
IQ3_SGN2 = False
# IQ3_SGTAB (with the IQ3_SGN2 path): the sign words come from a 2 KB LDS table built in the prologue from ksigns: entry i is
# (k0, s0, k1, s1) with s_h = the sign nibble h of ksigns[i] spread to one bit per byte and k_h = 0x80808080 - s_h, the same
# integers the per-pair spread computes; one 16-byte load per sign byte replaces the extract / spread / subtract chain.
IQ3_SGTAB = False             # set per tile (gen_gemm_tile Tile.sgtab)
# IQ3_SGTAB_W: words per table entry: 4 = (k0, s0, k1, s1); 2 = (s0, s1) with k = 0x80808080 - s in the loop (fewer registers)
IQ3_SGTAB_W = 2
# IQ3_F16P (IQ3_U8F path): the biased bytes u of each sign-applied word as f16 subnormal pairs (w & 0x00ff00ff,
# (w >> 8) & 0x00ff00ff = u * 2^-24, exact) read by v_fma_mix with dsc * 2^24 (exact) instead of v_cvt_f32_ubyteN:
# fma(dsc * 2^24, u * 2^-24, -128 dsc) has the same exact product as fma(dsc, u, -128 dsc): bit-identical.
IQ3_F16P = False
IQ3_GADDR = False             # set per tile (gen_gemm_tile Tile.gaddr), see the grid / sign lookups in iq3*_compute
IQ3_F16P_PERM = True          # the odd-byte pair as one v_perm (vector.shuffle with a zero word) instead of shift + and
# IQ2_W (IQ2_XXS / IQ2_XS): the IQ3 word path (VDEC_W, IQ3_U8F, VDECW_FR) for the IQ2 grids too. Grid magnitudes are 8 / 25 / 43
# (nonzero, < 128) and d * (2n + 1) / 8 * mag has <= 22 significant bits: the same exactness argument, bit-identical.
IQ2_W = False
# Q4FMIX (Q4_K/Q5_K): narrow as fptrunc(fma(e, 1, -dm)) instead of fptrunc(e - dm). The product by 1 is exact: bit-identical.
# It selects v_fma_mix{lo,hi}. v_cvt_f16_f32 writes only v0..v127: with 128 VGPRs of accumulators live, each result spills one.
# The 1.0 comes from gb & ~gb so the canonicalizer cannot fold the fma back into a subf.
Q4FMIX = False
# Q4DFMA (with Q4FMIX): fptrunc(fma(dsc, q, -dm)) straight from the nibble, without the vector multiply and the opaque 1.0:
# dsc * q has <= 21 significant bits, so it is exact and the fma sees the same value (and zero sign) as fma(dsc * q, 1, -dm).
Q4DFMA = True
# F16PAIR (IQ4_XS): the u bytes (q + 128) as f16 subnormals: w & 0x00ff00ff and (w >> 8) & 0x00ff00ff are pairs c * 2^-24
# (exact; FP16 denormals are on) for 3 ops per 4 values instead of a v_cvt_f32_ubyteN each; narrowed as
# fptrunc(fma(dsc * 2^24, c * 2^-24, -128 * dsc)): the scaled dsc is exact, so the fma sees the same exact product.
F16PAIR = True
# Q4_SEL (Q4_K / Q5_K): the scale / min unpack as both forms + select instead of a divergent scf.if (same integers)
Q4_SEL = True
# Q3_FMIX (Q3_K): narrow dsc * q as fptrunc(fma(dsc, q, -0.0)) instead of fptrunc(dsc * q): d * sc * q has <= 19 significant
# bits, so the product is exact either way, and the -0.0 addend keeps a zero product's sign (q = 0 with dsc < 0 stays -0; the
# MMA output keeps it too: +0 there changed the GEMM hashes). Selects v_fma_mix{lo,hi} instead of mul + v_cvt_f16_f32.
Q3_FMIX = True

KSUB = PH = GPP = GPL = ROWP = None


def configure(fmt):
    """Set the per-format phase geometry: KSUB = FMTS[fmt]["ksub"] K columns decoded per phase.
    A whole 256-wide block is a ~34 KB LDS tile (residency ~1.5 waves/SIMD); a narrower phase costs two barriers per phase."""
    global KSUB, ROWP, PH, GPP, GPL
    KSUB = FMTS[fmt]["ksub"]
    ROWP = KSUB + PAD           # f16 per LDS row
    PH = 256 // KSUB            # phases per 256-wide block
    GPP = KSUB // 32            # 32-element groups per row per phase
    GPL = GPP // NW             # groups decoded per lane per phase
    assert 256 % KSUB == 0 and GPP % NW == 0 and GPL >= 1




def _i8n(ty):
    import re as _re
    m = _re.fullmatch(r"vector<(\d+)xi8>", ty)
    return int(m.group(1)) if m and int(m.group(1)) % 4 == 0 else 0


def pack_vals(e, vals, tag):
    """Bitcast carried vector<Nxi8> values to vector<N/4xi32> and return the new (name, type) list.
    A vector<Nxi8> lowers to one byte per VGPR, so a carried qs[8] would hold 8 registers across the MMA loop."""
    out = []
    for nm, ty in vals:
        n = _i8n(ty)
        if n:
            pk = f"{nm}_{tag}pk"
            e(f"    {pk} = vector.bitcast {nm} : {ty} to vector<{n // 4}xi32>")
            out.append((pk, f"vector<{n // 4}xi32>"))
        else:
            out.append((nm, ty))
    return out


def unpack_vals(e, cur, orig):
    """Inverse of pack_vals for loop-carried names cur with original types orig."""
    names = []
    for (nm, ty), (_, oty) in zip(cur, orig):
        n = _i8n(oty)
        if n:
            e(f"    {nm}_u = vector.bitcast {nm} : {ty} to {oty}")
            names.append(f"{nm}_u")
        else:
            names.append(nm)
    return names


IQ4_KVALUES = [-127, -104, -83, -65, -49, -35, -22, -10,
               1, 13, 25, 38, 53, 69, 89, 113]


def iq4xs_loads(p, blk, gb):
    """Emit this lane's raw-byte loads for one phase of row %drow; return (lines, [(name, type)]) of the loaded values.
    blk: i32 SSA byte offset of the row's current block; gb: i32 SSA index of the lane's first group in the block.
    block_iq4_xs: d f16 @0, scales_h u16 @2, scales_l[4] @4, qs[128] @8. All *_loads share this contract.
    """
    L = []
    e = L.append
    vals = []
    # the 8-byte header (d, scales_h, scales_l[4]) as one VMEM load instead of four byte/half loads
    e(f"    %{p}hd_ix = index.cast {blk} : i32 to index")
    e(f"    %{p}hd_lo = index.max %{p}hd_ix, %c0 : index")
    e(f"    %{p}hd_idx = index.min %{p}hd_lo, %w_lim8 : index")
    e(f"    %{p}hdr = vector.load %w_view[%{p}hd_idx] : view<[%w_bytes]xi8> -> vector<8xi8>")
    vals.append((f"%{p}hdr", "vector<8xi8>"))
    e(f"    %{p}d_h_i = scalar.shrui {blk}, %c1i : i32")
    e(f"    %{p}d_ix = index.cast %{p}d_h_i : i32 to index")
    e(f"    %{p}d_lo = index.max %{p}d_ix, %c0 : index")
    e(f"    %{p}d_idx = index.min %{p}d_lo, %w_half_last : index")
    for u in range(GPL):
        e(f"    %{p}g{u} = scalar.addi {gb}, %c{u}i : i32")
        e(f"    %{p}gh{u} = scalar.shrui %{p}g{u}, %c1i : i32")
        e(f"    %{p}sl_i{u} = scalar.addi {blk}, %c4i : i32")
        e(f"    %{p}sl_j{u} = scalar.addi %{p}sl_i{u}, %{p}gh{u} : i32")
        e(f"    %{p}sl_ix{u} = index.cast %{p}sl_j{u} : i32 to index")
        e(f"    %{p}sl_lo{u} = index.max %{p}sl_ix{u}, %c0 : index")
        e(f"    %{p}sl_idx{u} = index.min %{p}sl_lo{u}, %w_last : index")
        e(f"    %{p}g16_{u} = scalar.shli %{p}g{u}, %c4i : i32")
        e(f"    %{p}qs_a{u} = scalar.addi {blk}, %c8i : i32")
        e(f"    %{p}qs_b{u} = scalar.addi %{p}qs_a{u}, %{p}g16_{u} : i32")
        e(f"    %{p}qs_ix{u} = index.cast %{p}qs_b{u} : i32 to index")
        e(f"    %{p}qs_lo{u} = index.max %{p}qs_ix{u}, %c0 : index")
        e(f"    %{p}qs_idx{u} = index.min %{p}qs_lo{u}, %w_lim : index")
        e(f"    %{p}q{u} = vector.load %w_view[%{p}qs_idx{u}] : view<[%w_bytes]xi8> -> vector<16xi8>")
        vals.append((f"%{p}q{u}", "vector<16xi8>"))
    return L, vals


def iq4xs_compute(v, gb):
    """Decode the loaded values v (iq4xs_loads order) into the LDS tile, in the .loom kernel's f32 op order.
    Element g*32 + w, L = w%16: nib = w<16 ? qs[g*16+L]&15 : qs[g*16+L]>>4
      sc = ((scales_l[g/2] >> 4*(g%2)) & 15 | ((scales_h >> 2g) & 3) << 4) - 32, value = (d*sc) * kvalues[nib]"""
    L = []
    e = L.append
    it = iter(v)
    hdr = next(it)
    e(f"    %hdw = vector.bitcast {hdr} : vector<8xi8> to vector<2xi32>")
    e("    %hdw0 = vector.extract %hdw[0] : vector<2xi32> -> i32")
    e("    %hdw1 = vector.extract %hdw[1] : vector<2xi32> -> i32")
    e("    %hd16 = scalar.trunci %hdw0 : i32 to i16")
    e("    %hdf = scalar.bitcast %hd16 : i16 to f16")
    e("    %d = scalar.extf %hdf : f16 to f32")
    e("    %shv = scalar.shrui %hdw0, %c16i_h : i32")
    for u in range(GPL):
        q = next(it)
        e(f"    %g{u} = scalar.addi {gb}, %c{u}i : i32")
        e(f"    %gl{u} = scalar.addi %gl_i, %c{u}i : i32")
        e(f"    %gp{u} = scalar.andi %g{u}, %c1i : i32")
        e(f"    %sh4_{u} = scalar.shli %gp{u}, %c2i : i32")
        e(f"    %sh2_{u} = scalar.shli %g{u}, %c1i : i32")
        # scales_l[g/2] is byte g/2 of the header's second word
        e(f"    %slg{u} = scalar.shrui %g{u}, %c1i : i32")
        e(f"    %sls{u} = scalar.shli %slg{u}, %c3i : i32")
        e(f"    %slw{u} = scalar.shrui %hdw1, %sls{u} : i32")
        e(f"    %slb{u} = scalar.andi %slw{u}, %c255i_h : i32")
        e(f"    %sc_sh{u} = scalar.shrui %slb{u}, %sh4_{u} : i32")
        e(f"    %sc_l{u} = scalar.andi %sc_sh{u}, %c15i : i32")
        e(f"    %sc_ha{u} = scalar.shrui %shv, %sh2_{u} : i32")
        e(f"    %sc_h{u} = scalar.andi %sc_ha{u}, %c3i : i32")
        e(f"    %sc_h4{u} = scalar.shli %sc_h{u}, %c4i : i32")
        e(f"    %sc6_{u} = scalar.ori %sc_l{u}, %sc_h4{u} : i32")
        e(f"    %sc{u} = scalar.subi %sc6_{u}, %c32i : i32")
        e(f"    %sc_f{u} = scalar.sitofp %sc{u} : i32 to f32")
        e(f"    %dsc{u} = scalar.mulf %d, %sc_f{u} : f32")
        e(f"    %dsc_v{u} = vector.splat %dsc{u} : vector<16xf32>")
        e(f"    %nlo{u} = vector.andi {q}, %m15v : vector<16xi8>")
        e(f"    %nhi{u} = vector.shrui {q}, %s4v : vector<16xi8>")
        # The codebook +128 as unsigned bytes: uitofp of a byte is one v_cvt_f32_ubyteN.
        # fptrunc(fma(s, u, -128*s)) = fptrunc((u-128)*s) exactly (<= 24 significant bits): bit-identical.
        # The bias rides v_fma_mix's f32 addend (no literal).
        for part in ("lo", "hi"):
            e(f"    %cu{part}{u} = vector.table.lookup %kvtu[%n{part}{u}] : vector<16xi8>, vector<16xi8> -> vector<16xi8>")
            if F16PAIR:
                e(f"    %cuw{part}{u} = vector.bitcast %cu{part}{u} : vector<16xi8> to vector<4xi32>")
                for w in range(4):
                    tw = f"{part}{u}_{w}"
                    # bytes 0 / 2 and 1 / 3 in the low bits of the two halves: f16 subnormals c * 2^-24 (exact)
                    e(f"    %cw{tw} = vector.extract %cuw{part}{u}[{w}] : vector<4xi32> -> i32")
                    e(f"    %cwe{tw} = scalar.andi %cw{tw}, %fpm00ff : i32")
                    e(f"    %cws{tw} = scalar.shrui %cw{tw}, %c8i_fp : i32")
                    e(f"    %cwo{tw} = scalar.andi %cws{tw}, %fpm00ff : i32")
                    for hh, src in ((0, f"%cwe{tw}"), (1, f"%cwo{tw}")):
                        e(f"    %cwv{tw}_{hh} = vector.from_elements {src} : vector<1xi32>")
                        e(f"    %cwh{tw}_{hh} = vector.bitcast %cwv{tw}_{hh} : vector<1xi32> to vector<2xf16>")
            else:
                e(f"    %fu{part}{u} = vector.uitofp %cu{part}{u} : vector<16xi8> to vector<16xf32>")
        e(f"    %nb{u} = scalar.mulf %dsc{u}, %cm128f_iq : f32")
        if F16PAIR:
            e(f"    %dsc24_{u} = scalar.mulf %dsc{u}, %c2p24f_iq : f32")
        # scalar form: fptrunc(fma) pairs feeding from_elements select as v_fma_mix{lo,hi}_f16
        for part in ("lo", "hi"):
            hs = []
            for j in range(16):
                if F16PAIR:
                    w, b = divmod(j, 4)
                    e(f"    %yh{part}{u}_{j} = vector.extract %cwh{part}{u}_{w}_{b % 2}[{b // 2}] : vector<2xf16> -> f16")
                    e(f"    %y{part}{u}_{j} = scalar.extf %yh{part}{u}_{j} : f16 to f32")
                else:
                    e(f"    %y{part}{u}_{j} = vector.extract %fu{part}{u}[{j}] : vector<16xf32> -> f32")
                e(f"    %m{part}{u}_{j} = scalar.fmaf {'%dsc24_' if F16PAIR else '%dsc'}{u}, %y{part}{u}_{j}, %nb{u} : f32")
                e(f"    %t{part}{u}_{j} = scalar.fptrunc %m{part}{u}_{j} : f32 to f16")
                hs.append(f"%t{part}{u}_{j}")
            e(f"    %h{part}{u} = vector.from_elements {', '.join(hs)} : vector<16xf16>")
        e(f"    %col_i{u} = scalar.shli %gl{u}, %c5i : i32")
        e(f"    %col_x{u} = index.cast %col_i{u} : i32 to index")
        e(f"    %col_l{u} = index.max %col_x{u}, %c0 : index")
        e(f"    %col{u} = index.min %col_l{u}, %ccolmax : index")
        e(f"    %colh{u} = index.add %col{u}, %c16 : index")
        e(f"    vector.store %hlo{u}, %wl_view[%drow, %col{u}] : vector<16xf16>, view<{LR}x{ROWP}xf16>")
        e(f"    vector.store %hhi{u}, %wl_view[%drow, %colh{u}] : vector<16xf16>, view<{LR}x{ROWP}xf16>")
    return L


def _ld8(e, p, name, off):
    """Load one weight byte at i32 SSA offset `off` into %{p}{name} (i8)."""
    e(f"    %{p}{name}_ix = index.cast {off} : i32 to index")
    e(f"    %{p}{name}_lo = index.max %{p}{name}_ix, %c0 : index")
    e(f"    %{p}{name}_idx = index.min %{p}{name}_lo, %w_last : index")
    e(f"    %{p}{name} = view.load %w_view[%{p}{name}_idx] : view<[%w_bytes]xi8> -> i8")


def _ldv(e, p, name, off, n):
    """Load n consecutive weight bytes at i32 SSA offset `off` as vector<nxi8>.
    The clamp must be w_bytes - n for this n: w_bytes - 16 moves loads near the tensor end (IQ3_S signs) to a wrong address."""
    e(f"    %{p}{name}_ix = index.cast {off} : i32 to index")
    e(f"    %{p}{name}_lo = index.max %{p}{name}_ix, %c0 : index")
    e(f"    %{p}{name}_idx = index.min %{p}{name}_lo, %w_lim{n} : index")
    e(f"    %{p}{name} = view.load %w_view[%{p}{name}_idx] : view<[%w_bytes]xi8> -> vector<{n}xi8>".replace("view.load", "vector.load"))


def _ldd(e, p, blk):
    """Load the block's f16 d (offset 0) into %{p}dh."""
    e(f"    %{p}d_h_i = scalar.shrui {blk}, %c1i : i32")
    e(f"    %{p}d_ix = index.cast %{p}d_h_i : i32 to index")
    e(f"    %{p}d_lo = index.max %{p}d_ix, %c0 : index")
    e(f"    %{p}d_idx = index.min %{p}d_lo, %w_half_last : index")
    e(f"    %{p}dh = view.load %w_f16_view[%{p}d_idx] : view<[%w_halfs]xf16> -> f16")


def _vdec_pair(e, t, gw0, gw1, sgb8, dsc_v8, col, u, p, dsc_s=None, sgt=None):
    """Decode the 8 elements one sign byte covers (grid words lw=2p and 2p+1) and store them to the LDS tile.
    mags = bytes of [gw0, gw1], s = sign bits LSB first, mag = (g ^ -s) + s in i8 (exact: grid magnitudes are < 128)."""
    if VDEC_W:
        # On the two grid words: s1 = sign bit i in byte i (nibble * 0x00204081), m = s1 * 255, mag = (g ^ m) + s1.
        # Per byte that is 256 - g for a set bit, no carry since grid magnitudes are > 0. (i8 vectors lower element by element.)
        sgn2 = IQ3_SGN2 and IQ3_U8F and VDECW_FR and dsc_s is not None
        assert sgt is None or sgn2, "the sign table feeds the IQ3_SGN2 path"
        if sgt is None:
            e(f"    %wsb_{t} = scalar.extui {sgb8} : i8 to i32")
        if sgn2 and sgt is None:
            e(f"    %c128i_sg2_{t} = scalar.constant 128 : i32")
            e(f"    %c16384i_sg2_{t} = scalar.constant 16384 : i32")
        for h, gw in ((0, gw0), (1, gw1)):
            if sgt is not None and IQ3_SGTAB_W == 2:
                e(f"    %ws1{h}_{t} = vector.extract {sgt}[{h}] : vector<2xi32> -> i32")
                e(f"    %wsk{h}_{t} = scalar.subi %c80x4_iq3, %ws1{h}_{t} : i32")
                e(f"    %wsx{h}_{t} = scalar.xori {gw}, %wsk{h}_{t} : i32")
                e(f"    %wu{h}_{t} = scalar.addi %wsx{h}_{t}, %ws1{h}_{t} : i32")
                continue
            if sgt is not None:
                e(f"    %wsk{h}_{t} = vector.extract {sgt}[{2 * h}] : vector<4xi32> -> i32")
                e(f"    %ws1{h}_{t} = vector.extract {sgt}[{2 * h + 1}] : vector<4xi32> -> i32")
                e(f"    %wsx{h}_{t} = scalar.xori {gw}, %wsk{h}_{t} : i32")
                e(f"    %wu{h}_{t} = scalar.addi %wsx{h}_{t}, %ws1{h}_{t} : i32")
                continue
            if h:
                e(f"    %wsn{h}_{t}0 = scalar.shrui %wsb_{t}, %c4i : i32")
            else:
                e(f"    %wsn{h}_{t}0 = scalar.addi %wsb_{t}, %c0i : i32")
            e(f"    %wsn{h}_{t} = scalar.andi %wsn{h}_{t}0, %c15i : i32")
            if sgn2:
                e(f"    %wst{h}_{t} = scalar.fmai %wsn{h}_{t}, %c128i_sg2_{t}, %wsn{h}_{t} : i32")
                e(f"    %wsp{h}_{t} = scalar.fmai %wst{h}_{t}, %c16384i_sg2_{t}, %wst{h}_{t} : i32")
                e(f"    %ws1{h}_{t} = scalar.andi %wsp{h}_{t}, %vdw_ones : i32")
                e(f"    %wsk{h}_{t} = scalar.subi %c80x4_iq3, %ws1{h}_{t} : i32")
                e(f"    %wsx{h}_{t} = scalar.xori {gw}, %wsk{h}_{t} : i32")
                e(f"    %wu{h}_{t} = scalar.addi %wsx{h}_{t}, %ws1{h}_{t} : i32")
                continue
            if VDECW_FR:
                # n * 0x00204081 & 0x01010101 == (t | t << 14) & 0x01010101, t = n | n << 7 (n <= 15: OR cannot carry)
                e(f"    %wst7{h}_{t} = scalar.shli %wsn{h}_{t}, %c7i : i32")
                e(f"    %wst{h}_{t} = scalar.ori %wsn{h}_{t}, %wst7{h}_{t} : i32")
                e(f"    %wst14{h}_{t} = scalar.shli %wst{h}_{t}, %c14i_vdw : i32")
                e(f"    %wsp{h}_{t} = scalar.ori %wst{h}_{t}, %wst14{h}_{t} : i32")
            else:
                e(f"    %wsp{h}_{t} = scalar.muli %wsn{h}_{t}, %vdw_spread : i32")
            e(f"    %ws1{h}_{t} = scalar.andi %wsp{h}_{t}, %vdw_ones : i32")
            if VDECW_FR:
                e(f"    %wsm8{h}_{t} = scalar.shli %ws1{h}_{t}, %c8i : i32")
                e(f"    %wsm{h}_{t} = scalar.subi %wsm8{h}_{t}, %ws1{h}_{t} : i32")
            else:
                e(f"    %wsm{h}_{t} = scalar.muli %ws1{h}_{t}, %vdw_ff : i32")
            e(f"    %wx{h}_{t} = scalar.xori {gw}, %wsm{h}_{t} : i32")
            e(f"    %wm{h}_{t} = scalar.addi %wx{h}_{t}, %ws1{h}_{t} : i32")
        if not sgn2:
            e(f"    %vmw_{t} = vector.from_elements %wm0_{t}, %wm1_{t} : vector<2xi32>")
            e(f"    %vm_{t} = vector.bitcast %vmw_{t} : vector<2xi32> to vector<8xi8>")
    else:
        e(f"    %vg_{t} = vector.from_elements {gw0}, {gw1} : vector<2xi32>")
        e(f"    %vb_{t} = vector.bitcast %vg_{t} : vector<2xi32> to vector<8xi8>")
        e(f"    %vs1_{t} = vector.from_elements {sgb8} : vector<1xi8>")
        e(f"    %vs_{t} = vector.bitunpacku<1> %vs1_{t} : vector<1xi8> -> vector<8xi8>")
        e(f"    %vn_{t} = vector.subi %z8v, %vs_{t} : vector<8xi8>")
        e(f"    %vx_{t} = vector.xori %vb_{t}, %vn_{t} : vector<8xi8>")
        e(f"    %vm_{t} = vector.addi %vx_{t}, %vs_{t} : vector<8xi8>")
    if IQ3_U8F and VDEC_W and dsc_s is not None:
        if not (IQ3_SGN2 and VDECW_FR):
            e(f"    %wu0_{t} = scalar.xori %wm0_{t}, %c80x4_iq3 : i32")
            e(f"    %wu1_{t} = scalar.xori %wm1_{t}, %c80x4_iq3 : i32")
        e(f"    %unb_{t} = scalar.mulf {dsc_s}, %cm128f_iq3 : f32")
        if IQ3_F16P:
            e(f"    %f16pm_{t} = scalar.constant 16711935 : i32")
            e(f"    %f16p8_{t} = scalar.constant 8 : i32")
            e(f"    %f16p24_{t} = scalar.constant 16777216.0 : f32")
            e(f"    %ud24_{t} = scalar.mulf {dsc_s}, %f16p24_{t} : f32")
            e(f"    %upz_{t} = scalar.constant 0 : i32")
            for h in range(2):
                e(f"    %upe{h}_{t} = scalar.andi %wu{h}_{t}, %f16pm_{t} : i32")
                if IQ3_F16P_PERM:
                    # bytes 1 / 3 into the low byte of each half, zero high bytes: one v_perm_b32 with a zero word
                    e(f"    %upw{h}_{t} = vector.from_elements %wu{h}_{t}, %upz_{t} : vector<2xi32>")
                    e(f"    %upb{h}_{t} = vector.bitcast %upw{h}_{t} : vector<2xi32> to vector<8xi8>")
                    e(f"    %upq{h}_{t} = vector.shuffle<[1, 4, 3, 4, 4, 4, 4, 4]> %upb{h}_{t} : vector<8xi8>")
                    e(f"    %upr{h}_{t} = vector.bitcast %upq{h}_{t} : vector<8xi8> to vector<2xi32>")
                    e(f"    %upo{h}_{t} = vector.extract %upr{h}_{t}[0] : vector<2xi32> -> i32")
                else:
                    e(f"    %ups{h}_{t} = scalar.shrui %wu{h}_{t}, %f16p8_{t} : i32")
                    e(f"    %upo{h}_{t} = scalar.andi %ups{h}_{t}, %f16pm_{t} : i32")
                for nm in ("e", "o"):
                    e(f"    %upv{nm}{h}_{t} = vector.from_elements %up{nm}{h}_{t} : vector<1xi32>")
                    e(f"    %uph{nm}{h}_{t} = vector.bitcast %upv{nm}{h}_{t} : vector<1xi32> to vector<2xf16>")
        else:
            e(f"    %vuw_{t} = vector.from_elements %wu0_{t}, %wu1_{t} : vector<2xi32>")
            e(f"    %vub_{t} = vector.bitcast %vuw_{t} : vector<2xi32> to vector<8xi8>")
            e(f"    %vuf_{t} = vector.uitofp %vub_{t} : vector<8xi8> to vector<8xf32>")
        hs = []
        for j in range(8):
            if IQ3_F16P:
                h, b = divmod(j, 4)
                e(f"    %uyh_{t}_{j} = vector.extract %uph{'e' if b % 2 == 0 else 'o'}{h}_{t}[{b // 2}] : vector<2xf16> -> f16")
                e(f"    %uy_{t}_{j} = scalar.extf %uyh_{t}_{j} : f16 to f32")
                e(f"    %um_{t}_{j} = scalar.fmaf %ud24_{t}, %uy_{t}_{j}, %unb_{t} : f32")
            else:
                e(f"    %uy_{t}_{j} = vector.extract %vuf_{t}[{j}] : vector<8xf32> -> f32")
                e(f"    %um_{t}_{j} = scalar.fmaf {dsc_s}, %uy_{t}_{j}, %unb_{t} : f32")
            e(f"    %uh_{t}_{j} = scalar.fptrunc %um_{t}_{j} : f32 to f16")
            hs.append(f"%uh_{t}_{j}")
        e(f"    %vh_{t} = vector.from_elements {', '.join(hs)} : vector<8xf16>")
    else:
        e(f"    %vf_{t} = vector.sitofp %vm_{t} : vector<8xi8> to vector<8xf32>")
        e(f"    %vv_{t} = vector.mulf {dsc_v8}, %vf_{t} : vector<8xf32>")
        e(f"    %vh_{t} = vector.fptrunc %vv_{t} : vector<8xf32> to vector<8xf16>")
    e(f"    %vc_{t} = index.constant {8 * p} : index")
    e(f"    %vco_{t} = index.add {col}, %vc_{t} : index")
    e(f"    vector.store %vh_{t}, %wl_view[%drow, %vco_{t}] : vector<8xf16>, view<{LR}x{ROWP}xf16>")


def _col_of(e, u):
    e(f"    %col_i{u} = scalar.shli %gl{u}, %c5i : i32")
    e(f"    %col_x{u} = index.cast %col_i{u} : i32 to index")
    e(f"    %col_l{u} = index.max %col_x{u}, %c0 : index")
    e(f"    %col{u} = index.min %col_l{u}, %ccolmax : index")


def iq3s_loads(p, blk, gb):
    """block_iq3_s (110 B): d f16 @0, qs[64] @2, qh[8] @66, signs[32] @74, scales[4] @106.
    Group g reads qs[8g..8g+7], qh[g], signs[4g..4g+3] and the scale byte g/2."""
    L = []
    e = L.append
    vals = []
    _ldd(e, p, blk)
    vals.append((f"%{p}dh", "f16"))
    for u in range(GPL):
        e(f"    %{p}g{u} = scalar.addi {gb}, %c{u}i : i32")
        e(f"    %{p}g8_{u} = scalar.shli %{p}g{u}, %c3i : i32")
        e(f"    %{p}qo{u} = scalar.addi {blk}, %{p}g8_{u} : i32")
        e(f"    %{p}qo2_{u} = scalar.addi %{p}qo{u}, %c2i : i32")
        _ldv(e, p, f"qs{u}", f"%{p}qo2_{u}", 8)
        vals.append((f"%{p}qs{u}", "vector<8xi8>"))
        e(f"    %{p}ho{u} = scalar.addi {blk}, %{p}g{u} : i32")
        e(f"    %{p}ho2_{u} = scalar.addi %{p}ho{u}, %c66i : i32")
        _ld8(e, p, f"qh{u}", f"%{p}ho2_{u}")
        vals.append((f"%{p}qh{u}", "i8"))
        e(f"    %{p}g4_{u} = scalar.shli %{p}g{u}, %c2i : i32")
        e(f"    %{p}so{u} = scalar.addi {blk}, %{p}g4_{u} : i32")
        e(f"    %{p}so2_{u} = scalar.addi %{p}so{u}, %c74i : i32")
        _ldv(e, p, f"sg{u}", f"%{p}so2_{u}", 4)
        vals.append((f"%{p}sg{u}", "vector<4xi8>"))
        e(f"    %{p}gd{u} = scalar.shrui %{p}g{u}, %c1i : i32")
        e(f"    %{p}co{u} = scalar.addi {blk}, %{p}gd{u} : i32")
        e(f"    %{p}co2_{u} = scalar.addi %{p}co{u}, %c106i : i32")
        _ld8(e, p, f"sc{u}", f"%{p}co2_{u}")
        vals.append((f"%{p}sc{u}", "i8"))
    return L, vals


def iq3s_compute(v, gb):
    """IQ3_S element decode, one row per lane, in the .loom kernel's op order. Element lw*4 + b of group g (lw = 2*l + which):
      gidx = qs[8g+lw] | ((qh[g] >> lw) & 1) << 8,  gword = grid[gidx]
      sign_nib = (signs[4g+l] >> 4*which) & 15,     s = (sign_nib >> b) & 1
      mag = ((gword >> 8b) & 255 ^ -s) + s
      value = (d * f32(1 + 2*((scales[g/2] >> 4*(g%2)) & 15))) * f32(mag)"""
    L = []
    e = L.append
    it = iter(v)
    dh = next(it)
    e(f"    %d = scalar.extf {dh} : f16 to f32")
    for u in range(GPL):
        qs = next(it); qh = next(it); sg = next(it); sc = next(it)
        e(f"    %g{u} = scalar.addi {gb}, %c{u}i : i32")
        e(f"    %gl{u} = scalar.addi %gl_i, %c{u}i : i32")
        e(f"    %odd{u} = scalar.andi %g{u}, %c1i : i32")
        e(f"    %nb_sh{u} = scalar.shli %odd{u}, %c2i : i32")
        e(f"    %scb{u} = scalar.extui {sc} : i8 to i32")
        e(f"    %nb_t{u} = scalar.shrui %scb{u}, %nb_sh{u} : i32")
        e(f"    %nib{u} = scalar.andi %nb_t{u}, %c15i : i32")
        e(f"    %nib2_{u} = scalar.shli %nib{u}, %c1i : i32")
        e(f"    %onep{u} = scalar.addi %nib2_{u}, %c1i : i32")
        e(f"    %scf{u} = scalar.sitofp %onep{u} : i32 to f32")
        e(f"    %dsc{u} = scalar.mulf %d, %scf{u} : f32")
        e(f"    %qhb{u} = scalar.extui {qh} : i8 to i32")
        e(f"    %dsc_v8_{u} = vector.splat %dsc{u} : vector<8xf32>")
        _col_of(e, u)
        gws = []
        # grid indices from 32-bit words (one v_bfe_u32 each); their range is known, so no clamp
        e(f"    %qsw{u} = vector.bitcast {qs} : vector<8xi8> to vector<2xi32>")
        for lw in range(8):
            t = f"{u}_{lw}"
            e(f"    %qwd_{t} = vector.extract %qsw{u}[{lw // 4}] : vector<2xi32> -> i32")
            if IQ3_GADDR:   # byte offset ((w >> (8k - 2)) & 0x3fc) + (qh bit lw) * 1024 (disjoint bits: one v_lshl_add)
                k8 = 8 * (lw % 4)
                if k8 == 0:
                    e(f"    %qsh_{t} = scalar.shli %qwd_{t}, %c2i_ga : i32")
                else:
                    e(f"    %qshc_{t} = scalar.constant {k8 - 2} : i32")
                    e(f"    %qsh_{t} = scalar.shrui %qwd_{t}, %qshc_{t} : i32")
                e(f"    %qby_{t} = scalar.andi %qsh_{t}, %c1020i_ga : i32")
                e(f"    %hbb_{t} = scalar.bitfield.extractu %qhb{u} {{offset = {lw}, width = 1}} : i32 -> i32")
                e(f"    %gbo_{t} = scalar.fmai %hbb_{t}, %c1024i_ga, %qby_{t} : i32")
                e(f"    %gbx_{t} = index.cast %gbo_{t} : i32 to index")
                e(f"    %gbd_{t} = index.assume %gbx_{t} [range(%gbx_{t}, 0, 2044)] : index")
                e(f"    %gwb_{t} = vector.load %grid_bview[%gbd_{t}] : view<2048xi8> -> vector<4xi8>")
                e(f"    %gwv_{t} = vector.bitcast %gwb_{t} : vector<4xi8> to vector<1xi32>")
                e(f"    %gw_{t} = vector.extract %gwv_{t}[0] : vector<1xi32> -> i32")
                gws.append(f"%gw_{t}")
                continue
            e(f"    %qsh_{t} = scalar.shrui %qwd_{t}, %c{8 * (lw % 4)}i : i32")
            e(f"    %qlo_{t} = scalar.andi %qsh_{t}, %c255i : i32")
            e(f"    %hb0_{t} = scalar.shrui %qhb{u}, %c{lw}i : i32")
            e(f"    %hb_{t} = scalar.andi %hb0_{t}, %c1i : i32")
            e(f"    %hb8_{t} = scalar.shli %hb_{t}, %c8i : i32")
            e(f"    %gi_{t} = scalar.ori %qlo_{t}, %hb8_{t} : i32")
            e(f"    %gix_{t} = index.cast %gi_{t} : i32 to index")
            e(f"    %gid_{t} = index.assume %gix_{t} [range(%gix_{t}, 0, 511)] : index")
            e(f"    %gw_{t} = view.load %grid_view[%gid_{t}] : view<512xi32> -> i32")
            gws.append(f"%gw_{t}")
        if _sgtab() and IQ3_GADDR and IQ3_SGTAB_W == 2:
            e(f"    %sgw1_{u} = vector.bitcast {sg} : vector<4xi8> to vector<1xi32>")
            e(f"    %sgw_{u} = vector.extract %sgw1_{u}[0] : vector<1xi32> -> i32")
        for pp in range(4):
            if _sgtab() and IQ3_GADDR and IQ3_SGTAB_W == 2:   # byte offset (word >> (8 pp - 3)) & 0x7f8
                if pp == 0:
                    e(f"    %sbs_{u}_{pp} = scalar.shli %sgw_{u}, %c3i_ga : i32")
                else:
                    e(f"    %sbc_{u}_{pp} = scalar.constant {8 * pp - 3} : i32")
                    e(f"    %sbs_{u}_{pp} = scalar.shrui %sgw_{u}, %sbc_{u}_{pp} : i32")
                e(f"    %sby_{u}_{pp} = scalar.andi %sbs_{u}_{pp}, %c2040i_ga : i32")
                e(f"    %sbx_{u}_{pp} = index.cast %sby_{u}_{pp} : i32 to index")
                e(f"    %sbd_{u}_{pp} = index.assume %sbx_{u}_{pp} [range(%sbx_{u}_{pp}, 0, 2040)] : index")
                e(f"    %sgb_{u}_{pp} = vector.load %sgt_bview[%sbd_{u}_{pp}] : view<2048xi8> -> vector<8xi8>")
                e(f"    %sgt_{u}_{pp} = vector.bitcast %sgb_{u}_{pp} : vector<8xi8> to vector<2xi32>")
                _vdec_pair(e, f"{u}_{pp}", gws[2 * pp], gws[2 * pp + 1], None, f"%dsc_v8_{u}", f"%col{u}", u, pp, f"%dsc{u}",
                           sgt=f"%sgt_{u}_{pp}")
                continue
            e(f"    %sgb8_{u}_{pp} = vector.extract {sg}[{pp}] : vector<4xi8> -> i8")
            if _sgtab():
                W = IQ3_SGTAB_W
                e(f"    %sgbi_{u}_{pp} = scalar.extui %sgb8_{u}_{pp} : i8 to i32")
                e(f"    %sgbx_{u}_{pp} = index.cast %sgbi_{u}_{pp} : i32 to index")
                e(f"    %sgbc_{u}_{pp} = index.assume %sgbx_{u}_{pp} [range(%sgbx_{u}_{pp}, 0, 255)] : index")
                e(f"    %sgi_{u}_{pp} = index.mul %sgbc_{u}_{pp}, %sgt_cw : index")
                e(f"    %sgt_{u}_{pp} = vector.load %sgt_view[%sgi_{u}_{pp}] : view<{256 * W}xi32> -> vector<{W}xi32>")
                _vdec_pair(e, f"{u}_{pp}", gws[2 * pp], gws[2 * pp + 1], None, f"%dsc_v8_{u}", f"%col{u}", u, pp, f"%dsc{u}",
                           sgt=f"%sgt_{u}_{pp}")
                continue
            _vdec_pair(e, f"{u}_{pp}", gws[2 * pp], gws[2 * pp + 1], f"%sgb8_{u}_{pp}", f"%dsc_v8_{u}", f"%col{u}", u, pp, f"%dsc{u}")
    return L


def iq3s_setup():
    """Copy the 512-word grid into 2 KiB of LDS (%grid_view): the eight lookups per group are ds_reads, not global gathers.
    The first K phase starts with a barrier, which publishes the copy."""
    return ["  %grid_g = buffer.view %grid_na[%base] : buffer -> view<512xi32>",
            "  %c511 = index.constant 511 : index",
            "  %vdw_spread = scalar.constant 2113665 : i32", "  %vdw_ones = scalar.constant 16843009 : i32",
            "  %vdw_ff = scalar.constant 255 : i32",
            "  %c80x4_iq3 = scalar.constant -2139062144 : i32", "  %cm128f_iq3 = scalar.constant -128.0 : f32",
            "  %c14i_vdw = scalar.constant 14 : i32",
            "  %c2i_ga = scalar.constant 2 : i32", "  %c1020i_ga = scalar.constant 1020 : i32",
            "  %c3i_ga = scalar.constant 3 : i32", "  %c1024i_ga = scalar.constant 1024 : i32",
            "  %c2040i_ga = scalar.constant 2040 : i32",
            "  %grid_bytes = index.constant 2048 : offset",
            "  %grid_l = buffer.alloca<workgroup> align(16) %grid_bytes : buffer",
            "  %grid_view = buffer.view %grid_l[%base] : buffer -> view<512xi32>",
            *(["  %grid_bview = buffer.view %grid_l[%base] : buffer -> view<2048xi8>"] if IQ3_GADDR else []),
            f"  %cgstep = index.constant {64 * NW} : index",
            "  %gsink = scf.for %gi = [%c0 to %c512 step %cgstep](%gm = %c0 : index) -> (index) {",
            "    %gidx0 = index.add %gi, %tid : index",
            "    %gidx = index.min %gidx0, %c511 : index",
            "    %gv = view.load %grid_g[%gidx] : view<512xi32> -> i32",
            "    view.store %gv, %grid_view[%gidx] : i32, view<512xi32>",
            "    scf.yield %gm : index",
            "  }"] + (_sign_table(256, from_ksigns=False) if _sgtab() else [])


def iq3xxs_loads(p, blk, gb):
    """block_iq3_xxs (98 B): d f16 @0, qs[64] @2 (grid indices), aux[32] @66 (one LE32 word per 32-element group).
    Group g reads qs[8g..8g+7] and aux bytes 66+4g..69+4g."""
    L = []
    e = L.append
    vals = []
    _ldd(e, p, blk)
    vals.append((f"%{p}dh", "f16"))
    for u in range(GPL):
        e(f"    %{p}g{u} = scalar.addi {gb}, %c{u}i : i32")
        e(f"    %{p}g8_{u} = scalar.shli %{p}g{u}, %c3i : i32")
        e(f"    %{p}qo{u} = scalar.addi {blk}, %{p}g8_{u} : i32")
        e(f"    %{p}qo2_{u} = scalar.addi %{p}qo{u}, %c2i : i32")
        _ldv(e, p, f"qs{u}", f"%{p}qo2_{u}", 8)
        vals.append((f"%{p}qs{u}", "vector<8xi8>"))
        e(f"    %{p}g4_{u} = scalar.shli %{p}g{u}, %c2i : i32")
        e(f"    %{p}ao{u} = scalar.addi {blk}, %{p}g4_{u} : i32")
        e(f"    %{p}ao2_{u} = scalar.addi %{p}ao{u}, %c66i : i32")
        _ldv(e, p, f"ax{u}", f"%{p}ao2_{u}", 4)
        vals.append((f"%{p}ax{u}", "vector<4xi8>"))
    return L, vals


def iq3xxs_compute(v, gb):
    """IQ3_XXS element decode, one row per lane, in the .loom kernel's op order. Element lw*4 + b of group g (lw = 2*l + which):
      gword = grid[qs[8g+lw]],  aux = LE32(aux bytes of g)
      sign_nib = (ksigns[(aux >> 7l) & 127] >> 4*which) & 15, s = (sign_nib >> b) & 1
      mag = ((gword >> 8b) & 255 ^ -s) + s
      value = (d * ((f32(aux >> 28) + 0.5) * 0.5)) * f32(mag)"""
    L = []
    e = L.append
    it = iter(v)
    dh = next(it)
    e(f"    %d = scalar.extf {dh} : f16 to f32")
    for u in range(GPL):
        qs = next(it); ax = next(it)
        e(f"    %gl{u} = scalar.addi %gl_i, %c{u}i : i32")
        e(f"    %axw{u} = vector.bitcast {ax} : vector<4xi8> to vector<1xi32>")
        e(f"    %aux{u} = vector.extract %axw{u}[0] : vector<1xi32> -> i32")
        e(f"    %n4_{u} = scalar.shrui %aux{u}, %c28i : i32")
        e(f"    %n4f_{u} = scalar.sitofp %n4_{u} : i32 to f32")
        e(f"    %hp_{u} = scalar.addf %n4f_{u}, %fhalf : f32")
        e(f"    %hp2_{u} = scalar.mulf %hp_{u}, %fhalf : f32")
        e(f"    %dsc{u} = scalar.mulf %d, %hp2_{u} : f32")
        e(f"    %dsc_v8_{u} = vector.splat %dsc{u} : vector<8xf32>")
        _col_of(e, u)
        gws = []
        # grid indices from 32-bit words (one v_bfe_u32 each); their range is known, so no clamp
        e(f"    %qsw{u} = vector.bitcast {qs} : vector<8xi8> to vector<2xi32>")
        for lw in range(8):
            t = f"{u}_{lw}"
            e(f"    %qwd_{t} = vector.extract %qsw{u}[{lw // 4}] : vector<2xi32> -> i32")
            if IQ3_GADDR:   # byte offset (w >> (8k - 2)) & 0x3fc, then / 4: the load's * 4 can fold into the mask
                k8 = 8 * (lw % 4)
                if k8 == 0:
                    e(f"    %qsh_{t} = scalar.shli %qwd_{t}, %c2i_ga : i32")
                else:
                    e(f"    %qshc_{t} = scalar.constant {k8 - 2} : i32")
                    e(f"    %qsh_{t} = scalar.shrui %qwd_{t}, %qshc_{t} : i32")
                e(f"    %qby_{t} = scalar.andi %qsh_{t}, %c1020i_ga : i32")
                e(f"    %gbx_{t} = index.cast %qby_{t} : i32 to index")
                e(f"    %gbd_{t} = index.assume %gbx_{t} [range(%gbx_{t}, 0, 1020)] : index")
                e(f"    %gwb_{t} = vector.load %grid_bview[%gbd_{t}] : view<1024xi8> -> vector<4xi8>")
                e(f"    %gwv_{t} = vector.bitcast %gwb_{t} : vector<4xi8> to vector<1xi32>")
                e(f"    %gw_{t} = vector.extract %gwv_{t}[0] : vector<1xi32> -> i32")
                gws.append(f"%gw_{t}")
                continue
            else:
                e(f"    %qsh_{t} = scalar.shrui %qwd_{t}, %c{8 * (lw % 4)}i : i32")
                e(f"    %qlo_{t} = scalar.andi %qsh_{t}, %c255i : i32")
            e(f"    %gix_{t} = index.cast %qlo_{t} : i32 to index")
            e(f"    %gid_{t} = index.assume %gix_{t} [range(%gix_{t}, 0, 255)] : index")
            e(f"    %gw_{t} = view.load %grid_view[%gid_{t}] : view<256xi32> -> i32")
            gws.append(f"%gw_{t}")
        for pp in range(4):
            if _sgtab() and IQ3_GADDR and IQ3_SGTAB_W == 2:
                if pp == 0:
                    e(f"    %sbs_{u}_{pp} = scalar.shli %aux{u}, %c3i_ga : i32")
                else:
                    e(f"    %sbc_{u}_{pp} = scalar.constant {7 * pp - 3} : i32")
                    e(f"    %sbs_{u}_{pp} = scalar.shrui %aux{u}, %sbc_{u}_{pp} : i32")
                e(f"    %sby_{u}_{pp} = scalar.andi %sbs_{u}_{pp}, %c1016i_ga : i32")
                e(f"    %sbx_{u}_{pp} = index.cast %sby_{u}_{pp} : i32 to index")
                e(f"    %sbd_{u}_{pp} = index.assume %sbx_{u}_{pp} [range(%sbx_{u}_{pp}, 0, 1016)] : index")
                e(f"    %sgb_{u}_{pp} = vector.load %sgt_bview[%sbd_{u}_{pp}] : view<1024xi8> -> vector<8xi8>")
                e(f"    %sgt_{u}_{pp} = vector.bitcast %sgb_{u}_{pp} : vector<8xi8> to vector<2xi32>")
                _vdec_pair(e, f"{u}_{pp}", gws[2 * pp], gws[2 * pp + 1], None, f"%dsc_v8_{u}", f"%col{u}", u, pp, f"%dsc{u}",
                           sgt=f"%sgt_{u}_{pp}")
                continue
            e(f"    %sid0_{u}_{pp} = scalar.shrui %aux{u}, %c{7 * pp}i : i32")
            e(f"    %sid_{u}_{pp} = scalar.andi %sid0_{u}_{pp}, %c127i : i32")
            e(f"    %sidx_{u}_{pp} = index.cast %sid_{u}_{pp} : i32 to index")
            e(f"    %sidc_{u}_{pp} = index.assume %sidx_{u}_{pp} [range(%sidx_{u}_{pp}, 0, 127)] : index")
            if _sgtab():
                W = IQ3_SGTAB_W
                e(f"    %sgi_{u}_{pp} = index.mul %sidc_{u}_{pp}, %sgt_cw : index")
                e(f"    %sgt_{u}_{pp} = vector.load %sgt_view[%sgi_{u}_{pp}] : view<{128 * W}xi32> -> vector<{W}xi32>")
                _vdec_pair(e, f"{u}_{pp}", gws[2 * pp], gws[2 * pp + 1], None, f"%dsc_v8_{u}", f"%col{u}", u, pp, f"%dsc{u}",
                           sgt=f"%sgt_{u}_{pp}")
                continue
            e(f"    %ks8_{u}_{pp} = view.load %ksigns_view[%sidc_{u}_{pp}] : view<128xi8> -> i8")
            _vdec_pair(e, f"{u}_{pp}", gws[2 * pp], gws[2 * pp + 1], f"%ks8_{u}_{pp}", f"%dsc_v8_{u}", f"%col{u}", u, pp, f"%dsc{u}")
    return L


def _stage_table(name, src, n, ty, bytes_per):
    """Copy an n-entry read-only table into LDS once; the first K phase's leading barrier publishes it."""
    return [f"  %{name}_g = buffer.view {src}[%base] : buffer -> view<{n}x{ty}>",
            f"  %{name}_bytes = index.constant {n * bytes_per} : offset",
            f"  %{name}_l = buffer.alloca<workgroup> align(16) %{name}_bytes : buffer",
            f"  %{name}_view = buffer.view %{name}_l[%base] : buffer -> view<{n}x{ty}>",
            f"  %{name}_n1 = index.constant {n - 1} : index",
            f"  %{name}_step = index.constant {64 * NW} : index",
            f"  %{name}_cnt = index.constant {n} : index",
            f"  %{name}_sink = scf.for %{name}_i = [%c0 to %{name}_cnt step %{name}_step](%{name}_m = %c0 : index) -> (index) {{",
            f"    %{name}_x0 = index.add %{name}_i, %tid : index",
            f"    %{name}_x = index.min %{name}_x0, %{name}_n1 : index",
            f"    %{name}_v = view.load %{name}_g[%{name}_x] : view<{n}x{ty}> -> {ty}",
            f"    view.store %{name}_v, %{name}_view[%{name}_x] : {ty}, view<{n}x{ty}>",
            f"    scf.yield %{name}_m : index",
            "  }"]


def iq3xxs_setup():
    return ["  %vdw_spread = scalar.constant 2113665 : i32", "  %vdw_ones = scalar.constant 16843009 : i32", "  %vdw_ff = scalar.constant 255 : i32",
            "  %c80x4_iq3 = scalar.constant -2139062144 : i32", "  %cm128f_iq3 = scalar.constant -128.0 : f32",
            "  %c14i_vdw = scalar.constant 14 : i32", "  %c2i_ga = scalar.constant 2 : i32", "  %c1020i_ga = scalar.constant 1020 : i32",
            "  %c3i_ga = scalar.constant 3 : i32", "  %c1016i_ga = scalar.constant 1016 : i32"] + ((["  %fhalf = scalar.constant 0.5 : f32"])
            + _stage_table("grid", "%grid_na", 256, "i32", 4)
            + (["  %grid_bview = buffer.view %grid_l[%base] : buffer -> view<1024xi8>"] if IQ3_GADDR else [])
            + (_sign_table() if _sgtab() else _stage_table("ksigns", "%ksigns_na", 128, "i8", 1)))


def _sgtab():
    return IQ3_SGTAB and IQ3_SGN2 and IQ3_U8F and VDECW_FR and VDEC_W


def _sign_table(n=128, from_ksigns=True):
    """The IQ3_SGTAB table: n entries of (s0, s1) or (k0, s0, k1, s1) of sign byte ksigns[i] (IQ3_XXS) or i itself (IQ3_S),
    see IQ3_SGTAB; written in a loop of
    32-entry steps so any workgroup of at least one wave covers it (entry min(i + tid, 127): duplicates write the same words);
    the first K phase's leading barrier publishes it."""
    W = IQ3_SGTAB_W
    L = [f"  %sgt_cw = index.constant {W} : index",
         *(["  %sgt_g = buffer.view %ksigns_na[%base] : buffer -> view<128xi8>"] if from_ksigns else []),
         f"  %sgt_bytes = index.constant {4 * n * W} : offset",
         "  %sgt_l = buffer.alloca<workgroup> align(16) %sgt_bytes : buffer",
         f"  %sgt_view = buffer.view %sgt_l[%base] : buffer -> view<{n * W}xi32>",
         *([f"  %sgt_bview = buffer.view %sgt_l[%base] : buffer -> view<{4 * n * W}xi8>"] if IQ3_GADDR else []),
         f"  %sgt_max = index.constant {n - 1} : index",
         "  %sgt_step = index.constant 32 : index",
         f"  %sgt_cnt = index.constant {n} : index",
         "  %sgt_c4i = scalar.constant 4 : i32", "  %sgt_c7i = scalar.constant 7 : i32", "  %sgt_c14i = scalar.constant 14 : i32",
         "  %sgt_c15i = scalar.constant 15 : i32", "  %sgt_ones = scalar.constant 16843009 : i32",
         "  %sgt_k80 = scalar.constant -2139062144 : i32",
         "  %sgt_sink = scf.for %sgt_i = [%c0 to %sgt_cnt step %sgt_step](%sgt_m = %c0 : index) -> (index) {",
         "    %sgt_x0 = index.add %sgt_i, %tid : index",
         "    %sgt_x = index.min %sgt_x0, %sgt_max : index",
         *(["    %sgt_b8 = view.load %sgt_g[%sgt_x] : view<128xi8> -> i8",
            "    %sgt_b = scalar.extui %sgt_b8 : i8 to i32"] if from_ksigns else
           ["    %sgt_b = index.cast %sgt_x : index to i32"])]
    ws = []
    for h in range(2):
        src = "%sgt_b" if h == 0 else "%sgt_bh"
        if h:
            L.append("    %sgt_bh = scalar.shrui %sgt_b, %sgt_c4i : i32")
        L += [f"    %sgt_n{h} = scalar.andi {src}, %sgt_c15i : i32",
              f"    %sgt_t7{h} = scalar.shli %sgt_n{h}, %sgt_c7i : i32",
              f"    %sgt_t{h} = scalar.ori %sgt_n{h}, %sgt_t7{h} : i32",
              f"    %sgt_p14{h} = scalar.shli %sgt_t{h}, %sgt_c14i : i32",
              f"    %sgt_p{h} = scalar.ori %sgt_t{h}, %sgt_p14{h} : i32",
              f"    %sgt_s{h} = scalar.andi %sgt_p{h}, %sgt_ones : i32",
              f"    %sgt_k{h} = scalar.subi %sgt_k80, %sgt_s{h} : i32"]
        ws += [f"%sgt_k{h}", f"%sgt_s{h}"] if W == 4 else [f"%sgt_s{h}"]
    L += [f"    %sgt_e = vector.from_elements {', '.join(ws)} : vector<{W}xi32>",
          "    %sgt_xi = index.mul %sgt_x, %sgt_cw : index",
          f"    vector.store %sgt_e, %sgt_view[%sgt_xi] : vector<{W}xi32>, view<{n * W}xi32>",
          "    scf.yield %sgt_m : index",
          "  }"]
    return L


def q4k_loads(p, blk, gb, q5=False):
    """block_q4_K (144 B): d f16 @0, dmin f16 @2, scales[12] @4, qs[128] @16.
    Sub-block g reads qs[32*(g/2) .. +31] (low nibbles for even g, high for odd) and scale bytes 4+g, 8+g, g (get_scale_min_k4).
    With GPL even a lane's groups are even/odd pairs that share qs; with GPL odd, q4k_compute picks the nibble at run time."""
    L = []
    e = L.append
    vals = []
    if Q4_HDR:
        # d, dmin and scales[12] as one 16-byte load (blocks are 16-aligned)
        e(f"    %{p}hd_ix = index.cast {blk} : i32 to index")
        e(f"    %{p}hd_lo = index.max %{p}hd_ix, %c0 : index")
        e(f"    %{p}hd_idx = index.min %{p}hd_lo, %w_lim16 : index")
        e(f"    %{p}hdr = vector.load %w_view[%{p}hd_idx] : view<[%w_bytes]xi8> -> vector<16xi8>")
        vals.append((f"%{p}hdr", "vector<16xi8>"))
    else:
        _ldd(e, p, blk)
        vals.append((f"%{p}dh", "f16"))
    if not Q4_HDR:
        e(f"    %{p}dm_h_i = scalar.addi %{p}d_h_i, %c1i : i32")
    if not Q4_HDR:
        e(f"    %{p}dm_ix = index.cast %{p}dm_h_i : i32 to index")
        e(f"    %{p}dm_lo = index.max %{p}dm_ix, %c0 : index")
        e(f"    %{p}dm_idx = index.min %{p}dm_lo, %w_half_last : index")
    if not Q4_HDR:
        e(f"    %{p}dmh = view.load %w_f16_view[%{p}dm_idx] : view<[%w_halfs]xf16> -> f16")
        vals.append((f"%{p}dmh", "f16"))
    if q5:
        # the 32-byte qh plane (offset 16) is shared by every group of the block
        e(f"    %{p}qh_a = scalar.addi {blk}, %c16i : i32")
        e(f"    %{p}qh_b = scalar.addi {blk}, %c32i : i32")
        _ldv(e, p, "qha_v", f"%{p}qh_a", 16)
        _ldv(e, p, "qhb_v", f"%{p}qh_b", 16)
        vals.append((f"%{p}qha_v", "vector<16xi8>"))
        vals.append((f"%{p}qhb_v", "vector<16xi8>"))
    for u in range(GPL):
        e(f"    %{p}g{u} = scalar.addi {gb}, %c{u}i : i32")
        if u % 2 == 0:
            e(f"    %{p}gp{u} = scalar.shrui %{p}g{u}, %c1i : i32")
            e(f"    %{p}g32_{u} = scalar.shli %{p}gp{u}, %c5i : i32")
            e(f"    %{p}qo{u} = scalar.addi {blk}, %{p}g32_{u} : i32")
            qsb = 48 if q5 else 16
            e(f"    %{p}qa{u} = scalar.addi %{p}qo{u}, %c{qsb}i : i32")
            e(f"    %{p}qb{u} = scalar.addi %{p}qo{u}, %c{qsb + 16}i : i32")
            _ldv(e, p, f"qa_v{u}", f"%{p}qa{u}", 16)
            _ldv(e, p, f"qb_v{u}", f"%{p}qb{u}", 16)
            vals.append((f"%{p}qa_v{u}", "vector<16xi8>"))
            vals.append((f"%{p}qb_v{u}", "vector<16xi8>"))
        if Q4_HDR:
            continue
        e(f"    %{p}ga{u} = scalar.addi {blk}, %{p}g{u} : i32")
        e(f"    %{p}la_o{u} = scalar.addi %{p}ga{u}, %c4i : i32")
        e(f"    %{p}lb_o{u} = scalar.addi %{p}ga{u}, %c8i : i32")
        _ld8(e, p, f"la{u}", f"%{p}la_o{u}")
        _ld8(e, p, f"lb{u}", f"%{p}lb_o{u}")
        _ld8(e, p, f"lc{u}", f"%{p}ga{u}")
        vals += [(f"%{p}la{u}", "i8"), (f"%{p}lb{u}", "i8"), (f"%{p}lc{u}", "i8")]
    return L, vals


def q4k_compute(v, gb, q5=False):
    """Q4_K / Q5_K element decode, one row per lane, in the .loom kernel's op order:
      value = (d * f32(sc)) * f32(q) - dmin * f32(m)"""
    L = []
    e = L.append
    it = iter(v)
    if Q4FMIX and not Q4DFMA:
        # an opaque 1.0 (see Q4FMIX): gb & ~gb is 0, unprovable to the folder
        e(f"    %q4nb = scalar.xori {gb}, %q4m1 : i32")
        e(f"    %q4z = scalar.andi {gb}, %q4nb : i32")
        e("    %q4ob = scalar.ori %q4z, %q4one_b : i32")
        e("    %q4one = scalar.bitcast %q4ob : i32 to f32")
    if Q4_HDR:
        hdr = next(it)
        e(f"    %hdw = vector.bitcast {hdr} : vector<16xi8> to vector<4xi32>")
        for w in range(4):
            e(f"    %hdw{w} = vector.extract %hdw[{w}] : vector<4xi32> -> i32")
        e("    %hdd16 = scalar.trunci %hdw0 : i32 to i16")
        e("    %hdm0 = scalar.shrui %hdw0, %c16i_q : i32")
        e("    %hdm16 = scalar.trunci %hdm0 : i32 to i16")
        e("    %hddf = scalar.bitcast %hdd16 : i16 to f16")
        e("    %hdmf = scalar.bitcast %hdm16 : i16 to f16")
        dh, dmh = "%hddf", "%hdmf"
    else:
        dh = next(it); dmh = next(it)
    if q5:
        qha = next(it); qhb = next(it)
    e(f"    %d = scalar.extf {dh} : f16 to f32")
    e(f"    %dmin = scalar.extf {dmh} : f16 to f32")
    for u in range(GPL):
        if u % 2 == 0:
            qa = next(it); qb = next(it)
        if not Q4_HDR:
            la = next(it); lb = next(it); lc = next(it)
        e(f"    %g{u} = scalar.addi {gb}, %c{u}i : i32")
        e(f"    %gl{u} = scalar.addi %gl_i, %c{u}i : i32")
        if Q4_HDR:
            # header byte k = 4 + g (la), 8 + g (lb), g (lc) is word k/4, byte k%4; g < 8: two adjacent words, picked on g/4
            e(f"    %hq{u} = scalar.shrui %g{u}, %c2i : i32")
            e(f"    %hq1_{u} = scalar.cmpi eq, %hq{u}, %c1i : i32")
            e(f"    %hr{u} = scalar.andi %g{u}, %c3i : i32")
            e(f"    %hs{u} = scalar.shli %hr{u}, %c3i : i32")
            for nm, w0 in (("la", 1), ("lb", 2), ("lc", 0)):
                e(f"    %{nm}w{u} = scf.select %hq1_{u}, %hdw{w0 + 1}, %hdw{w0} : i32")
                e(f"    %{nm}s{u} = scalar.shrui %{nm}w{u}, %hs{u} : i32")
                e(f"    %{nm}_{u} = scalar.andi %{nm}s{u}, %c255i_q : i32")
        else:
            e(f"    %la_{u} = scalar.extui {la} : i8 to i32")
            e(f"    %lb_{u} = scalar.extui {lb} : i8 to i32")
            e(f"    %lc_{u} = scalar.extui {lc} : i8 to i32")
        e(f"    %slo{u} = scalar.cmpi slt, %g{u}, %c4i : i32")
        if Q4_SEL:
            # both scale / min forms and a select: the same integers without the divergent branch (exec save / restore)
            e(f"    %s_a{u} = scalar.andi %la_{u}, %c63i : i32")
            e(f"    %m_a{u} = scalar.andi %lb_{u}, %c63i : i32")
        else:
            e(f"    %sc{u}, %mn{u} = scf.if %slo{u} -> (i32, i32) {{")
            e(f"      %s_a{u} = scalar.andi %la_{u}, %c63i : i32")
            e(f"      %m_a{u} = scalar.andi %lb_{u}, %c63i : i32")
            e(f"      scf.yield %s_a{u}, %m_a{u} : i32, i32")
            e("    } else {")
        e(f"      %lbl{u} = scalar.andi %lb_{u}, %c15i : i32")
        e(f"      %lch{u} = scalar.shrui %lc_{u}, %c6i : i32")
        e(f"      %lch4{u} = scalar.shli %lch{u}, %c4i : i32")
        e(f"      %s_b{u} = scalar.ori %lbl{u}, %lch4{u} : i32")
        e(f"      %lbh{u} = scalar.shrui %lb_{u}, %c4i : i32")
        e(f"      %lah{u} = scalar.shrui %la_{u}, %c6i : i32")
        e(f"      %lah4{u} = scalar.shli %lah{u}, %c4i : i32")
        e(f"      %m_b{u} = scalar.ori %lbh{u}, %lah4{u} : i32")
        if Q4_SEL:
            e(f"    %sc{u} = scf.select %slo{u}, %s_a{u}, %s_b{u} : i32")
            e(f"    %mn{u} = scf.select %slo{u}, %m_a{u}, %m_b{u} : i32")
        else:
            e(f"      scf.yield %s_b{u}, %m_b{u} : i32, i32")
            e("    }")
        e(f"    %scf{u} = scalar.sitofp %sc{u} : i32 to f32")
        e(f"    %mf{u} = scalar.sitofp %mn{u} : i32 to f32")
        e(f"    %dsc{u} = scalar.mulf %d, %scf{u} : f32")
        e(f"    %dm{u} = scalar.mulf %dmin, %mf{u} : f32")
        e(f"    %dsc_v{u} = vector.splat %dsc{u} : vector<16xf32>")
        e(f"    %dm_v{u} = vector.splat %dm{u} : vector<16xf32>")
        rt = GPL % 2 == 1
        if rt:
            # odd GPL: g's parity is known only at run time, so the nibble is (q >> 4*(g & 1)) & 15
            e(f"    %gpar{u} = scalar.andi %g{u}, %c1i : i32")
            e(f"    %gsh{u} = scalar.shli %gpar{u}, %c2i : i32")
            e(f"    %gsh8_{u} = scalar.trunci %gsh{u} : i32 to i8")
            e(f"    %gshv{u} = vector.splat %gsh8_{u} : vector<16xi8>")
        if q5:
            # fifth bit: quant = nibble + ((qh[lane] >> g) & 1) * 16
            e(f"    %g8_{u} = scalar.trunci %g{u} : i32 to i8")
            e(f"    %g8v_{u} = vector.splat %g8_{u} : vector<16xi8>")
        # nibbles on whole 32-bit words, (w >> s) & 0x0f0f0f0f: per-byte i8 shifts and masks lower element by element
        if rt:
            e(f"    %gshw{u} = vector.splat %gsh{u} : vector<4xi32>")
        else:
            e(f"    %gshw{u} = vector.splat %q4sh{0 if u % 2 == 0 else 4} : vector<4xi32>")
        for half, q, qh in (("lo", qa, qha if q5 else None), ("hi", qb, qhb if q5 else None)):
            e(f"    %qw{half}{u} = vector.bitcast {q} : vector<16xi8> to vector<4xi32>")
            e(f"    %qws{half}{u} = vector.shrui %qw{half}{u}, %gshw{u} : vector<4xi32>")
            e(f"    %qwm{half}{u} = vector.andi %qws{half}{u}, %m0f4 : vector<4xi32>")
            e(f"    %nq{half}{u} = vector.bitcast %qwm{half}{u} : vector<4xi32> to vector<16xi8>")
            src = f"%nq{half}{u}"
            if q5:
                # the fifth bit on words too: ((qh >> g) & 0x01010101) << 4, ORed in (the bits do not overlap, so OR is the add)
                e(f"    %hw{half}{u} = vector.bitcast {qh} : vector<16xi8> to vector<4xi32>")
                e(f"    %hgw{half}{u} = vector.splat %g{u} : vector<4xi32>")
                e(f"    %hs{half}{u} = vector.shrui %hw{half}{u}, %hgw{half}{u} : vector<4xi32>")
                e(f"    %hb{half}{u} = vector.andi %hs{half}{u}, %m014 : vector<4xi32>")
                e(f"    %h16{half}{u} = vector.shli %hb{half}{u}, %s44 : vector<4xi32>")
                e(f"    %n5w{half}{u} = vector.ori %qwm{half}{u}, %h16{half}{u} : vector<4xi32>")
                e(f"    %n5{half}{u} = vector.bitcast %n5w{half}{u} : vector<4xi32> to vector<16xi8>")
                src = f"%n5{half}{u}"
            # nibbles are 0..15 (0..31 with q5's bit): uitofp gives the same value and selects v_cvt_f32_ubyteN
            e(f"    %fq{half}{u} = vector.uitofp {src} : vector<16xi8> to vector<16xf32>")
            if not (Q4FMIX and Q4DFMA):
                e(f"    %sq{half}{u} = vector.mulf %dsc_v{u}, %fq{half}{u} : vector<16xf32>")
            if Q4FMIX:
                if half == "lo":
                    e(f"    %ndm{u} = scalar.negf %dm{u} : f32")
                hs = []
                for j in range(16):
                    t = f"{half}{u}_{j}"
                    if Q4DFMA:
                        e(f"    %qe{t} = vector.extract %fq{half}{u}[{j}] : vector<16xf32> -> f32")
                        e(f"    %qm{t} = scalar.fmaf %dsc{u}, %qe{t}, %ndm{u} : f32")
                    else:
                        e(f"    %qe{t} = vector.extract %sq{half}{u}[{j}] : vector<16xf32> -> f32")
                        e(f"    %qm{t} = scalar.fmaf %qe{t}, %q4one, %ndm{u} : f32")
                    e(f"    %qt{t} = scalar.fptrunc %qm{t} : f32 to f16")
                    hs.append(f"%qt{t}")
                e(f"    %h{half}{u} = vector.from_elements {', '.join(hs)} : vector<16xf16>")
            else:
                e(f"    %vq{half}{u} = vector.subf %sq{half}{u}, %dm_v{u} : vector<16xf32>")
                e(f"    %h{half}{u} = vector.fptrunc %vq{half}{u} : vector<16xf32> to vector<16xf16>")
        e(f"    %col_i{u} = scalar.shli %gl{u}, %c5i : i32")
        e(f"    %col_x{u} = index.cast %col_i{u} : i32 to index")
        e(f"    %col_l{u} = index.max %col_x{u}, %c0 : index")
        e(f"    %col{u} = index.min %col_l{u}, %ccolmax : index")
        e(f"    %colh{u} = index.add %col{u}, %c16 : index")
        e(f"    vector.store %hlo{u}, %wl_view[%drow, %col{u}] : vector<16xf16>, view<{LR}x{ROWP}xf16>")
        e(f"    vector.store %hhi{u}, %wl_view[%drow, %colh{u}] : vector<16xf16>, view<{LR}x{ROWP}xf16>")
    return L


def q4k_setup():
    return ["  %c15b = scalar.constant 15 : i8", "  %c4b = scalar.constant 4 : i8", "  %c1b = scalar.constant 1 : i8",
            "  %m15v = vector.splat %c15b : vector<16xi8>", "  %s4v = vector.splat %c4b : vector<16xi8>",
            "  %one8v = vector.splat %c1b : vector<16xi8>",
            "  %c0f4 = scalar.constant 252645135 : i32", "  %m0f4 = vector.splat %c0f4 : vector<4xi32>",
            "  %q4sh0 = scalar.constant 0 : i32", "  %q4sh4 = scalar.constant 4 : i32",
            "  %c16i_q = scalar.constant 16 : i32", "  %c255i_q = scalar.constant 255 : i32",
            "  %c014 = scalar.constant 16843009 : i32", "  %m014 = vector.splat %c014 : vector<4xi32>",
            "  %s44 = vector.splat %q4sh4 : vector<4xi32>",
            "  %q4m1 = scalar.constant -1 : i32", "  %q4one_b = scalar.constant 1065353216 : i32"]


def iq2xxs_loads(p, blk, gb):
    """block_iq2_xxs (66 B): d f16 @0, then per 32-element group g eight bytes at 2 + 8g.
    The eight bytes are four grid codes (bytes 0..3) and one LE32 aux word (bytes 4..7)."""
    L = []
    e = L.append
    vals = []
    _ldd(e, p, blk)
    vals.append((f"%{p}dh", "f16"))
    for u in range(GPL):
        e(f"    %{p}g{u} = scalar.addi {gb}, %c{u}i : i32")
        e(f"    %{p}g8_{u} = scalar.shli %{p}g{u}, %c3i : i32")
        e(f"    %{p}qo{u} = scalar.addi {blk}, %{p}g8_{u} : i32")
        e(f"    %{p}qo2_{u} = scalar.addi %{p}qo{u}, %c2i : i32")
        _ldv(e, p, f"qs{u}", f"%{p}qo2_{u}", 8)
        vals.append((f"%{p}qs{u}", "vector<8xi8>"))
    return L, vals


def iq2xxs_compute(v, gb):
    """IQ2_XXS decode in the .loom kernel's op order, 8 elements per grid code li:
      gw = grid words (2*code, 2*code+1), s = bits of ksigns[(aux >> 7li) & 127]
      mag = (g ^ -s) + s, value = (d * ((f32(aux >> 28) + 0.5) * 0.25)) * f32(mag)"""
    L = []
    e = L.append
    it = iter(v)
    dh = next(it)
    e(f"    %d = scalar.extf {dh} : f16 to f32")
    for u in range(GPL):
        qs = next(it)
        e(f"    %gl{u} = scalar.addi %gl_i, %c{u}i : i32")
        e(f"    %qw{u} = vector.bitcast {qs} : vector<8xi8> to vector<2xi32>")
        e(f"    %aux{u} = vector.extract %qw{u}[1] : vector<2xi32> -> i32")
        e(f"    %n4_{u} = scalar.shrui %aux{u}, %c28i : i32")
        e(f"    %n4f_{u} = scalar.sitofp %n4_{u} : i32 to f32")
        e(f"    %hp_{u} = scalar.addf %n4f_{u}, %fhalf : f32")
        e(f"    %hp2_{u} = scalar.mulf %hp_{u}, %fquarter : f32")
        e(f"    %dsc{u} = scalar.mulf %d, %hp2_{u} : f32")
        e(f"    %dsc_v8_{u} = vector.splat %dsc{u} : vector<8xf32>")
        _col_of(e, u)
        for li in range(4):
            t = f"{u}_{li}"
            e(f"    %cd8_{t} = vector.extract {qs}[{li}] : vector<8xi8> -> i8")
            e(f"    %cd_{t} = scalar.extui %cd8_{t} : i8 to i32")
            e(f"    %w0_{t} = scalar.shli %cd_{t}, %c1i : i32")
            e(f"    %w1_{t} = scalar.addi %w0_{t}, %c1i : i32")
            for w in (0, 1):
                e(f"    %wx{w}_{t} = index.cast %w{w}_{t} : i32 to index")
                e(f"    %wl{w}_{t} = index.max %wx{w}_{t}, %c0 : index")
                e(f"    %wc{w}_{t} = index.min %wl{w}_{t}, %c511 : index")
                e(f"    %gw{w}_{t} = view.load %grid_view[%wc{w}_{t}] : view<512xi32> -> i32")
            e(f"    %sid0_{t} = scalar.shrui %aux{u}, %c{7 * li}i : i32")
            e(f"    %sid_{t} = scalar.andi %sid0_{t}, %c127i : i32")
            e(f"    %sidx_{t} = index.cast %sid_{t} : i32 to index")
            e(f"    %sidl_{t} = index.max %sidx_{t}, %c0 : index")
            e(f"    %sidc_{t} = index.min %sidl_{t}, %c127 : index")
            e(f"    %ks8_{t} = view.load %ksigns_view[%sidc_{t}] : view<128xi8> -> i8")
            _vdec_pair(e, t, f"%gw0_{t}", f"%gw1_{t}", f"%ks8_{t}", f"%dsc_v8_{u}", f"%col{u}", u, li, f"%dsc{u}" if IQ2_W else None)
    return L


def iq2xs_loads(p, blk, gb):
    """block_iq2_xs (74 B): d f16 @0, qs[32] u16 @2, scales[8] @66.
    Group g (32 elements) reads the four codes qs[4g..4g+3] (bytes 2 + 8g) and scale byte 66 + g."""
    L = []
    e = L.append
    vals = []
    _ldd(e, p, blk)
    vals.append((f"%{p}dh", "f16"))
    for u in range(GPL):
        e(f"    %{p}g{u} = scalar.addi {gb}, %c{u}i : i32")
        e(f"    %{p}g8_{u} = scalar.shli %{p}g{u}, %c3i : i32")
        e(f"    %{p}qo{u} = scalar.addi {blk}, %{p}g8_{u} : i32")
        e(f"    %{p}qo2_{u} = scalar.addi %{p}qo{u}, %c2i : i32")
        _ldv(e, p, f"qs{u}", f"%{p}qo2_{u}", 8)
        vals.append((f"%{p}qs{u}", "vector<8xi8>"))
        e(f"    %{p}so{u} = scalar.addi {blk}, %{p}g{u} : i32")
        e(f"    %{p}so2_{u} = scalar.addi %{p}so{u}, %c66i : i32")
        _ld8(e, p, f"sc{u}", f"%{p}so2_{u}")
        vals.append((f"%{p}sc{u}", "i8"))
    return L, vals


def iq2xs_compute(v, gb):
    """IQ2_XS decode in yah_ffn_gemm_iq2xs_f32.loom's op order, 8 elements per code l (elements 8l..8l+7 of the group):
      gw = grid words (2*(code & 511), +1), s = bits of ksigns[code >> 9],
      mag = (g ^ -s) + s, nib = scales[g] low nibble for l < 2, high for l >= 2,
      value = (d * ((f32(nib) + 0.5) * 0.25)) * f32(mag)"""
    L = []
    e = L.append
    it = iter(v)
    dh = next(it)
    e(f"    %d = scalar.extf {dh} : f16 to f32")
    for u in range(GPL):
        qs = next(it); sc = next(it)
        e(f"    %gl{u} = scalar.addi %gl_i, %c{u}i : i32")
        e(f"    %qw{u} = vector.bitcast {qs} : vector<8xi8> to vector<2xi32>")
        e(f"    %scb{u} = scalar.extui {sc} : i8 to i32")
        for hv in (0, 1):
            if hv:
                e(f"    %nb{hv}_{u}0 = scalar.shrui %scb{u}, %c4i : i32")
            else:
                e(f"    %nb{hv}_{u}0 = scalar.andi %scb{u}, %c15i : i32")
            e(f"    %nbf{hv}_{u} = scalar.sitofp %nb{hv}_{u}0 : i32 to f32")
            e(f"    %hp{hv}_{u} = scalar.addf %nbf{hv}_{u}, %fhalf : f32")
            e(f"    %hq{hv}_{u} = scalar.mulf %hp{hv}_{u}, %fquarter : f32")
            e(f"    %dsc{hv}_{u} = scalar.mulf %d, %hq{hv}_{u} : f32")
            e(f"    %dsc_v8_{hv}_{u} = vector.splat %dsc{hv}_{u} : vector<8xf32>")
        _col_of(e, u)
        for l in range(4):
            t = f"{u}_{l}"
            e(f"    %cw_{t} = vector.extract %qw{u}[{l // 2}] : vector<2xi32> -> i32")
            if l % 2:
                e(f"    %cd_{t} = scalar.shrui %cw_{t}, %c16i_2 : i32")
            else:
                e(f"    %cd_{t} = scalar.andi %cw_{t}, %c65535i_2 : i32")
            e(f"    %gi_{t} = scalar.andi %cd_{t}, %c511i_2 : i32")
            e(f"    %w0_{t} = scalar.shli %gi_{t}, %c1i : i32")
            e(f"    %w1_{t} = scalar.addi %w0_{t}, %c1i : i32")
            for w in (0, 1):
                e(f"    %wx{w}_{t} = index.cast %w{w}_{t} : i32 to index")
                e(f"    %wl{w}_{t} = index.max %wx{w}_{t}, %c0 : index")
                e(f"    %wc{w}_{t} = index.min %wl{w}_{t}, %c1023 : index")
                e(f"    %gw{w}_{t} = view.load %grid_view[%wc{w}_{t}] : view<1024xi32> -> i32")
            e(f"    %sid0_{t} = scalar.shrui %cd_{t}, %c9i_2 : i32")
            e(f"    %sid_{t} = scalar.andi %sid0_{t}, %c127i : i32")
            e(f"    %sidx_{t} = index.cast %sid_{t} : i32 to index")
            e(f"    %sidl_{t} = index.max %sidx_{t}, %c0 : index")
            e(f"    %sidc_{t} = index.min %sidl_{t}, %c127 : index")
            e(f"    %ks8_{t} = view.load %ksigns_view[%sidc_{t}] : view<128xi8> -> i8")
            _vdec_pair(e, t, f"%gw0_{t}", f"%gw1_{t}", f"%ks8_{t}", f"%dsc_v8_{l // 2}_{u}", f"%col{u}", u, l,
                       f"%dsc{l // 2}_{u}" if IQ2_W else None)
    return L


def iq2xs_setup():
    return (["  %c14i_vdw = scalar.constant 14 : i32", "  %c80x4_iq3 = scalar.constant -2139062144 : i32", "  %cm128f_iq3 = scalar.constant -128.0 : f32",
             "  %fhalf = scalar.constant 0.5 : f32", "  %fquarter = scalar.constant 0.25 : f32",
             "  %c1023 = index.constant 1023 : index",
             "  %c16i_2 = scalar.constant 16 : i32", "  %c65535i_2 = scalar.constant 65535 : i32",
             "  %c511i_2 = scalar.constant 511 : i32", "  %c9i_2 = scalar.constant 9 : i32",
             "  %vdw_spread = scalar.constant 2113665 : i32", "  %vdw_ones = scalar.constant 16843009 : i32",
             "  %vdw_ff = scalar.constant 255 : i32"]
            + _stage_table("grid", "%grid_na", 1024, "i32", 4)
            + _stage_table("ksigns", "%ksigns_na", 128, "i8", 1))


def iq2xxs_setup():
    return (["  %c14i_vdw = scalar.constant 14 : i32", "  %vdw_spread = scalar.constant 2113665 : i32", "  %vdw_ones = scalar.constant 16843009 : i32", "  %vdw_ff = scalar.constant 255 : i32",
             "  %c80x4_iq3 = scalar.constant -2139062144 : i32", "  %cm128f_iq3 = scalar.constant -128.0 : f32",
             "  %fhalf = scalar.constant 0.5 : f32", "  %fquarter = scalar.constant 0.25 : f32",
             "  %c511 = index.constant 511 : index"]
            + _stage_table("grid", "%grid_na", 512, "i32", 4)
            + _stage_table("ksigns", "%ksigns_na", 128, "i8", 1))


def q6k_loads(p, blk, gb):
    """block_q6_K (210 B): ql[128] @0, qh[64] @128, int8 scales[16] @192, d @208.
    Group g = 4*half + seg reads 32 ql bytes at 64*half + 32*(seg&1) and 32 qh bytes at 128 + 32*half.
    Its two scales are at 192 + 8*half + 2*seg (+1)."""
    L = []
    e = L.append
    vals = []
    e(f"    %{p}d_h0 = scalar.shrui {blk}, %c1i : i32")
    e(f"    %{p}d_h_i = scalar.addi %{p}d_h0, %c104i : i32")
    e(f"    %{p}d_ix = index.cast %{p}d_h_i : i32 to index")
    e(f"    %{p}d_lo = index.max %{p}d_ix, %c0 : index")
    e(f"    %{p}d_idx = index.min %{p}d_lo, %w_half_last : index")
    e(f"    %{p}dh = view.load %w_f16_view[%{p}d_idx] : view<[%w_halfs]xf16> -> f16")
    vals.append((f"%{p}dh", "f16"))
    for u in range(GPL):
        e(f"    %{p}g{u} = scalar.addi {gb}, %c{u}i : i32")
        e(f"    %{p}hf{u} = scalar.shrui %{p}g{u}, %c2i : i32")
        e(f"    %{p}sg{u} = scalar.andi %{p}g{u}, %c3i : i32")
        e(f"    %{p}so{u} = scalar.andi %{p}sg{u}, %c1i : i32")
        e(f"    %{p}h64_{u} = scalar.shli %{p}hf{u}, %c6i : i32")
        e(f"    %{p}s32_{u} = scalar.shli %{p}so{u}, %c5i : i32")
        e(f"    %{p}qlo{u} = scalar.addi %{p}h64_{u}, %{p}s32_{u} : i32")
        e(f"    %{p}qla{u} = scalar.addi {blk}, %{p}qlo{u} : i32")
        e(f"    %{p}qlb{u} = scalar.addi %{p}qla{u}, %c16i : i32")
        _ldv(e, p, f"qla_v{u}", f"%{p}qla{u}", 16)
        _ldv(e, p, f"qlb_v{u}", f"%{p}qlb{u}", 16)
        e(f"    %{p}h32_{u} = scalar.shli %{p}hf{u}, %c5i : i32")
        e(f"    %{p}qh0_{u} = scalar.addi {blk}, %{p}h32_{u} : i32")
        e(f"    %{p}qha{u} = scalar.addi %{p}qh0_{u}, %c128i : i32")
        e(f"    %{p}qhb{u} = scalar.addi %{p}qha{u}, %c16i : i32")
        _ldv(e, p, f"qha_v{u}", f"%{p}qha{u}", 16)
        _ldv(e, p, f"qhb_v{u}", f"%{p}qhb{u}", 16)
        e(f"    %{p}h8_{u} = scalar.shli %{p}hf{u}, %c3i : i32")
        e(f"    %{p}s2_{u} = scalar.shli %{p}sg{u}, %c1i : i32")
        e(f"    %{p}sc0_{u} = scalar.addi %{p}h8_{u}, %{p}s2_{u} : i32")
        e(f"    %{p}sca{u} = scalar.addi {blk}, %{p}sc0_{u} : i32")
        e(f"    %{p}sca2_{u} = scalar.addi %{p}sca{u}, %c192i : i32")
        e(f"    %{p}scb2_{u} = scalar.addi %{p}sca2_{u}, %c1i : i32")
        _ld8(e, p, f"sa{u}", f"%{p}sca2_{u}")
        _ld8(e, p, f"sb{u}", f"%{p}scb2_{u}")
        vals += [(f"%{p}qla_v{u}", "vector<16xi8>"), (f"%{p}qlb_v{u}", "vector<16xi8>"),
                 (f"%{p}qha_v{u}", "vector<16xi8>"), (f"%{p}qhb_v{u}", "vector<16xi8>"),
                 (f"%{p}sa{u}", "i8"), (f"%{p}sb{u}", "i8")]
    return L, vals


def q6k_compute(v, gb):
    """Q6_K element decode, one row per lane, in the .loom kernel's op order:
      code = low4 | ((qh >> 2*seg) & 3) << 4
      value = (d * f32(int8 scale)) * f32(code - 32)"""
    L = []
    e = L.append
    it = iter(v)
    dh = next(it)
    e(f"    %d = scalar.extf {dh} : f16 to f32")
    for u in range(GPL):
        qla = next(it); qlb = next(it); qha = next(it); qhb = next(it); sa = next(it); sb = next(it)
        e(f"    %g{u} = scalar.addi {gb}, %c{u}i : i32")
        e(f"    %gl{u} = scalar.addi %gl_i, %c{u}i : i32")
        e(f"    %sg{u} = scalar.andi %g{u}, %c3i : i32")
        e(f"    %hi{u} = scalar.shrui %sg{u}, %c1i : i32")
        e(f"    %qls{u} = scalar.shli %hi{u}, %c2i : i32")
        e(f"    %qhs{u} = scalar.shli %sg{u}, %c1i : i32")
        e(f"    %qls8_{u} = scalar.trunci %qls{u} : i32 to i8")
        e(f"    %qhs8_{u} = scalar.trunci %qhs{u} : i32 to i8")
        e(f"    %qlsv{u} = vector.splat %qls8_{u} : vector<16xi8>")
        e(f"    %qhsv{u} = vector.splat %qhs8_{u} : vector<16xi8>")
        _col_of(e, u)
        e(f"    %colh{u} = index.add %col{u}, %c16 : index")
        for half, ql, qh, sc, col in (("lo", qla, qha, sa, f"%col{u}"), ("hi", qlb, qhb, sb, f"%colh{u}")):
            t = f"{half}{u}"
            e(f"    %scf_{t} = scalar.sitofp {sc} : i8 to f32")
            e(f"    %dsc_{t} = scalar.mulf %d, %scf_{t} : f32")
            e(f"    %dsv_{t} = vector.splat %dsc_{t} : vector<16xf32>")
            e(f"    %l0_{t} = vector.shrui {ql}, %qlsv{u} : vector<16xi8>")
            e(f"    %l_{t} = vector.andi %l0_{t}, %m15v : vector<16xi8>")
            e(f"    %h0_{t} = vector.shrui {qh}, %qhsv{u} : vector<16xi8>")
            e(f"    %h1_{t} = vector.andi %h0_{t}, %m3v : vector<16xi8>")
            e(f"    %h4_{t} = vector.shli %h1_{t}, %s4v : vector<16xi8>")
            e(f"    %cd_{t} = vector.ori %l_{t}, %h4_{t} : vector<16xi8>")
            e(f"    %bs_{t} = vector.subi %cd_{t}, %m32v : vector<16xi8>")
            e(f"    %bf_{t} = vector.sitofp %bs_{t} : vector<16xi8> to vector<16xf32>")
            e(f"    %vv_{t} = vector.mulf %dsv_{t}, %bf_{t} : vector<16xf32>")
            e(f"    %hv_{t} = vector.fptrunc %vv_{t} : vector<16xf32> to vector<16xf16>")
            e(f"    vector.store %hv_{t}, %wl_view[%drow, {col}] : vector<16xf16>, view<{LR}x{ROWP}xf16>")
    return L


def q6k_setup():
    return ["  %c15b = scalar.constant 15 : i8", "  %c4b = scalar.constant 4 : i8",
            "  %c3b = scalar.constant 3 : i8", "  %c32b = scalar.constant 32 : i8",
            "  %m15v = vector.splat %c15b : vector<16xi8>", "  %s4v = vector.splat %c4b : vector<16xi8>",
            "  %m3v = vector.splat %c3b : vector<16xi8>", "  %m32v = vector.splat %c32b : vector<16xi8>"]


def q8_0_loads(p, blk, gb):
    """Q8_0 in 256-element super-blocks of 8 blocks (272 B): group g is block g, d f16 at 34g, qs int8[32] at 34g + 2."""
    L = []
    e = L.append
    vals = []
    for u in range(GPL):
        e(f"    %{p}g{u} = scalar.addi {gb}, %c{u}i : i32")
        e(f"    %{p}g34_{u} = scalar.muli %{p}g{u}, %c34i : i32")
        e(f"    %{p}bo{u} = scalar.addi {blk}, %{p}g34_{u} : i32")
        e(f"    %{p}dh_i{u} = scalar.shrui %{p}bo{u}, %c1i : i32")
        e(f"    %{p}dx{u} = index.cast %{p}dh_i{u} : i32 to index")
        e(f"    %{p}dl{u} = index.max %{p}dx{u}, %c0 : index")
        e(f"    %{p}dc{u} = index.min %{p}dl{u}, %w_half_last : index")
        e(f"    %{p}dh{u} = view.load %w_f16_view[%{p}dc{u}] : view<[%w_halfs]xf16> -> f16")
        e(f"    %{p}qa{u} = scalar.addi %{p}bo{u}, %c2i : i32")
        e(f"    %{p}qb{u} = scalar.addi %{p}bo{u}, %c18i : i32")
        _ldv(e, p, f"qa_v{u}", f"%{p}qa{u}", 16)
        _ldv(e, p, f"qb_v{u}", f"%{p}qb{u}", 16)
        vals += [(f"%{p}dh{u}", "f16"), (f"%{p}qa_v{u}", "vector<16xi8>"), (f"%{p}qb_v{u}", "vector<16xi8>")]
    return L, vals


def q8_0_compute(v, gb):
    """Q8_0 decode in the .loom kernel's op order: value = d * f32(q)."""
    L = []
    e = L.append
    it = iter(v)
    for u in range(GPL):
        dh = next(it); qa = next(it); qb = next(it)
        e(f"    %gl{u} = scalar.addi %gl_i, %c{u}i : i32")
        e(f"    %d{u} = scalar.extf {dh} : f16 to f32")
        e(f"    %dv{u} = vector.splat %d{u} : vector<16xf32>")
        _col_of(e, u)
        e(f"    %colh{u} = index.add %col{u}, %c16 : index")
        for half, q, col in (("lo", qa, f"%col{u}"), ("hi", qb, f"%colh{u}")):
            t = f"{half}{u}"
            e(f"    %qf_{t} = vector.sitofp {q} : vector<16xi8> to vector<16xf32>")
            e(f"    %vv_{t} = vector.mulf %dv{u}, %qf_{t} : vector<16xf32>")
            e(f"    %hv_{t} = vector.fptrunc %vv_{t} : vector<16xf32> to vector<16xf16>")
            e(f"    vector.store %hv_{t}, %wl_view[%drow, {col}] : vector<16xf16>, view<{LR}x{ROWP}xf16>")
    return L


def f16_loads(p, blk, gb):
    """Pre-dequantized f16 weights (probe for decode-free GEMMs): a 256-element block is 512 B; group g is 32 halves at 64g."""
    L = []
    e = L.append
    vals = []
    for u in range(GPL):
        e(f"    %{p}g{u} = scalar.addi {gb}, %c{u}i : i32")
        e(f"    %{p}g64_{u} = scalar.muli %{p}g{u}, %c64i : i32")
        e(f"    %{p}bo{u} = scalar.addi {blk}, %{p}g64_{u} : i32")
        e(f"    %{p}hb{u} = scalar.shrui %{p}bo{u}, %c1i : i32")
        e(f"    %{p}hx{u} = index.cast %{p}hb{u} : i32 to index")
        e(f"    %{p}hl{u} = index.max %{p}hx{u}, %c0 : index")
        e(f"    %{p}hlim{u} = index.sub %w_halfs, %c32 : index")
        e(f"    %{p}ha{u} = index.min %{p}hl{u}, %{p}hlim{u} : index")
        e(f"    %{p}hb16_{u} = index.add %{p}ha{u}, %c16 : index")
        e(f"    %{p}fa{u} = vector.load %w_f16_view[%{p}ha{u}] : view<[%w_halfs]xf16> -> vector<16xf16>")
        e(f"    %{p}fb{u} = vector.load %w_f16_view[%{p}hb16_{u}] : view<[%w_halfs]xf16> -> vector<16xf16>")
        vals += [(f"%{p}fa{u}", "vector<16xf16>"), (f"%{p}fb{u}", "vector<16xf16>")]
    return L, vals


def f16_compute(v, gb):
    """f16 weights: copy into the LDS weight tile (no decode)."""
    L = []
    e = L.append
    it = iter(v)
    for u in range(GPL):
        fa = next(it); fb = next(it)
        e(f"    %gl{u} = scalar.addi %gl_i, %c{u}i : i32")
        _col_of(e, u)
        e(f"    %colh{u} = index.add %col{u}, %c16 : index")
        e(f"    vector.store {fa}, %wl_view[%drow, %col{u}] : vector<16xf16>, view<{LR}x{ROWP}xf16>")
        e(f"    vector.store {fb}, %wl_view[%drow, %colh{u}] : vector<16xf16>, view<{LR}x{ROWP}xf16>")
    return L


def iq4xs_setup():
    L = [f"  %kv{i} = scalar.constant {v} : i8" for i, v in enumerate(IQ4_KVALUES)]
    L += ["  %c15b_iq = scalar.constant 15 : i8", "  %c4b_iq = scalar.constant 4 : i8",
          "  %c16i_h = scalar.constant 16 : i32", "  %c255i_h = scalar.constant 255 : i32",
          "  %c0f4_iq = scalar.constant 252645135 : i32", "  %m0f4_iq = vector.splat %c0f4_iq : vector<4xi32>",
          "  %c4w_iq = scalar.constant 4 : i32", "  %s4w_iq = vector.splat %c4w_iq : vector<4xi32>"]
    L.append("  %kvt = vector.from_elements " + ", ".join(f"%kv{i}" for i in range(16)) + " : vector<16xi8>")
    L += [f"  %kvu{i} = scalar.constant {(v + 128) - 256 if v + 128 > 127 else v + 128} : i8" for i, v in enumerate(IQ4_KVALUES)]
    L.append("  %kvtu = vector.from_elements " + ", ".join(f"%kvu{i}" for i in range(16)) + " : vector<16xi8>")
    L += ["  %c128f_iq = scalar.constant 128.0 : f32", "  %c128v_iq = vector.splat %c128f_iq : vector<16xf32>",
          "  %cm128f_iq = scalar.constant -128.0 : f32", "  %c2p24f_iq = scalar.constant 16777216.0 : f32",
          "  %fpm00ff = scalar.constant 16711935 : i32", "  %c8i_fp = scalar.constant 8 : i32"]
    L += ["  %c15b = scalar.constant 15 : i8", "  %c4b = scalar.constant 4 : i8",
          "  %m15v = vector.splat %c15b : vector<16xi8>", "  %s4v = vector.splat %c4b : vector<16xi8>"]
    return L


def q3k_loads(p, blk, gb):
    """block_q3_K (110 B): hmask[32] @0, qs[64] @32, scales[12] @96, d f16 @108.
    Group g, half = g/4, sp = g%4: element h16*16 + j reads bits 2sp, 2sp+1 of qs[32*half + 16*h16 + j], bit g of hmask[16*h16 + j].
    The group's two 16-element halves use scales si = 2g and 2g+1."""
    L = []
    e = L.append
    vals = []
    e(f"    %{p}dq_h = scalar.shrui {blk}, %c1i : i32")
    e(f"    %{p}dq_i = scalar.addi %{p}dq_h, %q3c54i : i32")
    e(f"    %{p}dq_ix = index.cast %{p}dq_i : i32 to index")
    e(f"    %{p}dq_lo = index.max %{p}dq_ix, %c0 : index")
    e(f"    %{p}dq_idx = index.min %{p}dq_lo, %w_half_last : index")
    # scales[12] @96 and d @108 as one 16-byte load at @94 (inside the block)
    e(f"    %{p}q3h_o = scalar.addi {blk}, %q3c94i : i32")
    _ldv(e, p, "q3hdr", f"%{p}q3h_o", 16)
    vals.append((f"%{p}q3hdr", "vector<16xi8>"))
    e(f"    %{p}hm_b = scalar.addi {blk}, %c16i : i32")
    _ldv(e, p, "hma", blk, 16)
    _ldv(e, p, "hmb", f"%{p}hm_b", 16)
    vals += [(f"%{p}hma", "vector<16xi8>"), (f"%{p}hmb", "vector<16xi8>")]
    for u in range(GPL):
        e(f"    %{p}g{u} = scalar.addi {gb}, %c{u}i : i32")
        e(f"    %{p}hf{u} = scalar.shrui %{p}g{u}, %c2i : i32")
        e(f"    %{p}hf32_{u} = scalar.shli %{p}hf{u}, %c5i : i32")
        e(f"    %{p}qo{u} = scalar.addi {blk}, %{p}hf32_{u} : i32")
        e(f"    %{p}qa_o{u} = scalar.addi %{p}qo{u}, %c32i : i32")
        e(f"    %{p}qb_o{u} = scalar.addi %{p}qo{u}, %c48i : i32")
        _ldv(e, p, f"qa{u}", f"%{p}qa_o{u}", 16)
        _ldv(e, p, f"qb{u}", f"%{p}qb_o{u}", 16)
        vals += [(f"%{p}qa{u}", "vector<16xi8>"), (f"%{p}qb{u}", "vector<16xi8>")]
        # low-nibble scale bytes scales[2*(g&3)] and +1, high bytes scales[8+2*(g&1)] and +1
        e(f"    %{p}g3_{u} = scalar.andi %{p}g{u}, %c3i : i32")
        e(f"    %{p}g3x2_{u} = scalar.shli %{p}g3_{u}, %c1i : i32")
        e(f"    %{p}sl0_{u} = scalar.addi {blk}, %q3c96i : i32")
        e(f"    %{p}sla_o{u} = scalar.addi %{p}sl0_{u}, %{p}g3x2_{u} : i32")
        e(f"    %{p}slb_o{u} = scalar.addi %{p}sla_o{u}, %c1i : i32")
        e(f"    %{p}g1_{u} = scalar.andi %{p}g{u}, %c1i : i32")
        e(f"    %{p}g1x2_{u} = scalar.shli %{p}g1_{u}, %c1i : i32")
        e(f"    %{p}sh0_{u} = scalar.addi {blk}, %c104i : i32")
        e(f"    %{p}sha_o{u} = scalar.addi %{p}sh0_{u}, %{p}g1x2_{u} : i32")
        e(f"    %{p}shb_o{u} = scalar.addi %{p}sha_o{u}, %c1i : i32")
    return L, vals


def q3k_compute(v, gb):
    """Q3_K element decode in yah_ffn_gemm_q3k_f32.loom's op order:
      low = (qs >> 2sp) & 3, bit = (hmask >> g) & 1, quant = (low | bit<<2) - 4
      low4 = (scales[si&7] >> 4*(si>>3)) & 15, high2 = (scales[8+si%4] >> 2*(si>>2)) & 3
      scale = (low4 | high2<<4) - 32, value = (f32(d) * f32(scale)) * f32(quant)
    The integer steps are exact, so 16 at a time in a vector gives the same values; the f32 products keep the .loom order."""
    L = []
    e = L.append
    it = iter(v)
    hdr = next(it); hma = next(it); hmb = next(it)
    # window bytes 94..109: word i = bytes 94+4i..97+4i; d = word 3 >> 16
    e(f"    %q3w = vector.bitcast {hdr} : vector<16xi8> to vector<4xi32>")
    for w in range(4):
        e(f"    %q3w{w} = vector.extract %q3w[{w}] : vector<4xi32> -> i32")
    e("    %q3dw = scalar.shrui %q3w3, %q3c16i : i32")
    e("    %q3d16 = scalar.trunci %q3dw : i32 to i16")
    e("    %q3dh = scalar.bitcast %q3d16 : i16 to f16")
    e("    %d = scalar.extf %q3dh : f16 to f32")
    for u in range(GPL):
        qa = next(it); qb = next(it)
        e(f"    %g{u} = scalar.addi {gb}, %c{u}i : i32")
        # scales[2*(g&3)], +1 at window bytes 2+2*(g&3); scales[8+2*(g&1)], +1 at 10+2*(g&1): 16-bit pairs from words 0..3
        e(f"    %q3g3_{u} = scalar.andi %g{u}, %c3i : i32")
        e(f"    %q3g1_{u} = scalar.andi %g{u}, %c1i : i32")
        e(f"    %q3z_{u} = scalar.cmpi eq, %q3g3_{u}, %c0i : i32")
        e(f"    %q3t_{u} = scalar.cmpi eq, %q3g3_{u}, %c3i : i32")
        e(f"    %q3wa0_{u} = scf.select %q3t_{u}, %q3w2, %q3w1 : i32")
        e(f"    %q3wa_{u} = scf.select %q3z_{u}, %q3w0, %q3wa0_{u} : i32")
        e(f"    %q3odd_{u} = scalar.cmpi eq, %q3g1_{u}, %c1i : i32")
        e(f"    %q3wah_{u} = scalar.shrui %q3wa_{u}, %q3c16i : i32")
        e(f"    %q3pa_{u} = scf.select %q3odd_{u}, %q3wa_{u}, %q3wah_{u} : i32")
        e(f"    %q3wb_{u} = scf.select %q3odd_{u}, %q3w3, %q3w2 : i32")
        e(f"    %q3wbh_{u} = scalar.shrui %q3wb_{u}, %q3c16i : i32")
        e(f"    %q3pb_{u} = scf.select %q3odd_{u}, %q3wb_{u}, %q3wbh_{u} : i32")
        e(f"    %q3la_{u} = scalar.andi %q3pa_{u}, %c255i_3 : i32")
        e(f"    %q3lb0_{u} = scalar.shrui %q3pa_{u}, %c8i : i32")
        e(f"    %q3lb_{u} = scalar.andi %q3lb0_{u}, %c255i_3 : i32")
        e(f"    %q3ha_{u} = scalar.andi %q3pb_{u}, %c255i_3 : i32")
        e(f"    %q3hb0_{u} = scalar.shrui %q3pb_{u}, %c8i : i32")
        e(f"    %q3hb_{u} = scalar.andi %q3hb0_{u}, %c255i_3 : i32")
        la, lb, ha, hb = f"%q3la_{u}", f"%q3lb_{u}", f"%q3ha_{u}", f"%q3hb_{u}"
        e(f"    %gl{u} = scalar.addi %gl_i, %c{u}i : i32")
        e(f"    %q3sp{u} = scalar.andi %g{u}, %c3i : i32")
        e(f"    %q3ls{u} = scalar.shli %q3sp{u}, %c1i : i32")
        e(f"    %q3ls8_{u} = scalar.trunci %q3ls{u} : i32 to i8")
        e(f"    %q3lsv{u} = vector.splat %q3ls8_{u} : vector<16xi8>")
        e(f"    %q3bs8_{u} = scalar.trunci %g{u} : i32 to i8")
        e(f"    %q3bsv{u} = vector.splat %q3bs8_{u} : vector<16xi8>")
        e(f"    %q3lsw{u} = vector.splat %q3ls{u} : vector<4xi32>")
        e(f"    %q3bsw{u} = vector.splat %g{u} : vector<4xi32>")
        e(f"    %q3hf{u} = scalar.shrui %g{u}, %c2i : i32")
        e(f"    %q3s4{u} = scalar.shli %q3hf{u}, %c2i : i32")
        e(f"    %q3g2{u} = scalar.shrui %g{u}, %c1i : i32")
        e(f"    %q3s2{u} = scalar.shli %q3g2{u}, %c1i : i32")
        e(f"    %col_i{u} = scalar.shli %gl{u}, %c5i : i32")
        e(f"    %col_x{u} = index.cast %col_i{u} : i32 to index")
        e(f"    %col_l{u} = index.max %col_x{u}, %c0 : index")
        e(f"    %col{u} = index.min %col_l{u}, %ccolmax : index")
        e(f"    %colh{u} = index.add %col{u}, %c16 : index")
        for hn, q, hm, lo8, hi8, col in (("lo", qa, hma, la, ha, f"%col{u}"), ("hi", qb, hmb, lb, hb, f"%colh{u}")):
            t = f"{hn}{u}"
            # on 32-bit words: per byte (q >> 2sp) & 3 | ((hm >> g) & 1) << 2, minus 4 as (x | 0x80) - 4 ^ 0x80 (no borrow)
            e(f"    %q3qw{t} = vector.bitcast {q} : vector<16xi8> to vector<4xi32>")
            e(f"    %q3hw{t} = vector.bitcast {hm} : vector<16xi8> to vector<4xi32>")
            e(f"    %q3lw{t} = vector.shrui %q3qw{t}, %q3lsw{u} : vector<4xi32>")
            e(f"    %q3low{t} = vector.andi %q3lw{t}, %q3m3w : vector<4xi32>")
            e(f"    %q3bw{t} = vector.shrui %q3hw{t}, %q3bsw{u} : vector<4xi32>")
            e(f"    %q3bit{t} = vector.andi %q3bw{t}, %q3m1w : vector<4xi32>")
            e(f"    %q3b2{t} = vector.shli %q3bit{t}, %q3s2w : vector<4xi32>")
            e(f"    %q3lb{t} = vector.ori %q3low{t}, %q3b2{t} : vector<4xi32>")
            e(f"    %q3o8{t} = vector.ori %q3lb{t}, %q3m80w : vector<4xi32>")
            e(f"    %q3s4w{t} = vector.subi %q3o8{t}, %q3m4w : vector<4xi32>")
            e(f"    %q3x8{t} = vector.xori %q3s4w{t}, %q3m80w : vector<4xi32>")
            e(f"    %q3qn{t} = vector.bitcast %q3x8{t} : vector<4xi32> to vector<16xi8>")
            e(f"    %q3qf{t} = vector.sitofp %q3qn{t} : vector<16xi8> to vector<16xf32>")
            e(f"    %q3l8{t} = scalar.addi {lo8}, %c0i : i32")
            e(f"    %q3l4s{t} = scalar.shrui %q3l8{t}, %q3s4{u} : i32")
            e(f"    %q3l4{t} = scalar.andi %q3l4s{t}, %c15i : i32")
            e(f"    %q3h8{t} = scalar.addi {hi8}, %c0i : i32")
            e(f"    %q3h2s{t} = scalar.shrui %q3h8{t}, %q3s2{u} : i32")
            e(f"    %q3h2{t} = scalar.andi %q3h2s{t}, %c3i : i32")
            e(f"    %q3h4{t} = scalar.shli %q3h2{t}, %c4i : i32")
            e(f"    %q3s6{t} = scalar.ori %q3l4{t}, %q3h4{t} : i32")
            e(f"    %q3sc{t} = scalar.subi %q3s6{t}, %c32i : i32")
            e(f"    %q3scf{t} = scalar.sitofp %q3sc{t} : i32 to f32")
            e(f"    %q3dsc{t} = scalar.mulf %d, %q3scf{t} : f32")
            if Q3_FMIX:
                hs = []
                for j in range(16):
                    e(f"    %q3y{t}_{j} = vector.extract %q3qf{t}[{j}] : vector<16xf32> -> f32")
                    e(f"    %q3m{t}_{j} = scalar.fmaf %q3dsc{t}, %q3y{t}_{j}, %q3cm0f : f32")
                    e(f"    %q3h{t}_{j} = scalar.fptrunc %q3m{t}_{j} : f32 to f16")
                    hs.append(f"%q3h{t}_{j}")
                e(f"    %h{hn}{u} = vector.from_elements {', '.join(hs)} : vector<16xf16>")
            else:
                e(f"    %q3dv{t} = vector.splat %q3dsc{t} : vector<16xf32>")
                e(f"    %q3v{t} = vector.mulf %q3dv{t}, %q3qf{t} : vector<16xf32>")
                e(f"    %h{hn}{u} = vector.fptrunc %q3v{t} : vector<16xf32> to vector<16xf16>")
        e(f"    vector.store %hlo{u}, %wl_view[%drow, %col{u}] : vector<16xf16>, view<{LR}x{ROWP}xf16>")
        e(f"    vector.store %hhi{u}, %wl_view[%drow, %colh{u}] : vector<16xf16>, view<{LR}x{ROWP}xf16>")
    return L


def q3k_setup():
    return ["  %q3cm0f = scalar.constant -0.0 : f32", "  %q3c54i = scalar.constant 54 : i32", "  %q3c96i = scalar.constant 96 : i32",
            "  %q3c3b = scalar.constant 3 : i8", "  %q3c1b = scalar.constant 1 : i8",
            "  %q3c2b = scalar.constant 2 : i8", "  %q3c4b = scalar.constant 4 : i8",
            "  %q3m3v = vector.splat %q3c3b : vector<16xi8>", "  %q3m1v = vector.splat %q3c1b : vector<16xi8>",
            "  %q3s2v = vector.splat %q3c2b : vector<16xi8>", "  %q3f4v = vector.splat %q3c4b : vector<16xi8>",
            "  %q3c94i = scalar.constant 94 : i32", "  %q3c16i = scalar.constant 16 : i32",
            "  %c255i_3 = scalar.constant 255 : i32",
            "  %q3k3 = scalar.constant 50529027 : i32", "  %q3m3w = vector.splat %q3k3 : vector<4xi32>",
            "  %q3k1 = scalar.constant 16843009 : i32", "  %q3m1w = vector.splat %q3k1 : vector<4xi32>",
            "  %q3k2 = scalar.constant 2 : i32", "  %q3s2w = vector.splat %q3k2 : vector<4xi32>",
            "  %q3k80 = scalar.constant -2139062144 : i32", "  %q3m80w = vector.splat %q3k80 : vector<4xi32>",
            "  %q3k4 = scalar.constant 67372036 : i32", "  %q3m4w = vector.splat %q3k4 : vector<4xi32>"]


FMTS = {
    # bb: block bytes; kdiv: format blocks per 256-element super-block; decode: (loads, compute); extra: bindings after %weight
    # ksub: best measured phase width at pp2048 (NW=2). At 128 the IQ3_S decode pushes 8 accumulators into scratch.
    "iq4xs": dict(bb=136, ksub=128, decode=(iq4xs_loads, iq4xs_compute), extra=[], setup=iq4xs_setup),
    "iq3s": dict(bb=110, ksub=64, decode=(iq3s_loads, iq3s_compute), extra=["grid"], setup=iq3s_setup),
    "q4k": dict(bb=144, ksub=128, decode=(q4k_loads, q4k_compute), extra=[], setup=q4k_setup),
    # Q5_K: Q4_K plus a fifth bit from the qh plane; qs at 48 instead of 16
    "q5k": dict(bb=176, ksub=128, decode=(lambda p, b, g: q4k_loads(p, b, g, q5=True),
                                          lambda v, g: q4k_compute(v, g, q5=True)),
                extra=[], setup=q4k_setup),
    "q8_0": dict(bb=272, kdiv=8, ksub=64, decode=(q8_0_loads, q8_0_compute), extra=[], setup=lambda: []),
    # probe only: pre-dequantized f16 weights, decode-free
    "f16": dict(bb=512, ksub=64, decode=(f16_loads, f16_compute), extra=[], setup=lambda: []),
    "q6k": dict(bb=210, ksub=64, decode=(q6k_loads, q6k_compute), extra=[], setup=q6k_setup),
    "q3k": dict(bb=110, ksub=64, decode=(q3k_loads, q3k_compute), extra=[], setup=q3k_setup),
    "iq2xxs": dict(bb=66, ksub=64, decode=(iq2xxs_loads, iq2xxs_compute), extra=["grid", "ksigns"], setup=iq2xxs_setup),
    "iq2xs": dict(bb=74, ksub=64, decode=(iq2xs_loads, iq2xs_compute), extra=["grid", "ksigns"], setup=iq2xs_setup),
    "iq3xxs": dict(bb=98, ksub=64, decode=(iq3xxs_loads, iq3xxs_compute), extra=["grid", "ksigns"], setup=iq3xxs_setup),
}

