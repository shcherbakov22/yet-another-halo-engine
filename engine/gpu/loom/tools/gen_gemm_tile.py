#!/usr/bin/env python3
"""Tile GEMM yah_ffn_gemm_<fmt>[_swiglu|_kres|_kqg]: wave32, many waves, decoded weights and activations both in LDS.

Workgroup: BM rows x BN tokens (128 x 256) over WM x WN wave32 waves (4 x 4, or 4 x 2 for WAVE_FMTS), each owning a TM x TN tile.
Grid: (m_tiles / ROWGRP, token_tiles); geometry() gives the dispatch.txt tile.
Bindings and layouts as gen_gemm_decode.py, plus gate_out for kqg.
The decode arithmetic and per-accumulator MMA order match gen_gemm_decode.py, so the output is bit-identical to it.
Per K phase the decoded weight tile (BM x KSUB) and the activation tile (BN x KSUB) sit in LDS, so the MMA loop reads only LDS.
The next phase's weight bytes and activation rows load into registers during this phase and go to LDS after it.
Many waves per SIMD hide the load latency that the wave64 shared kernel (about 2 waves per SIMD) leaves exposed.
"""
import os
import sys
import dataclasses
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_bfp16_encode as GE  # noqa: E402
import gen_gemm_decode as G  # noqa: E402

# Default workgroup: 128 x 256 over 4 x 4 waves (WAVE_FMTS: 4 x 2).
# 256 x 256 does not fit: 64 KB of tiles plus the IQ grid table in LDS is over gfx11's 64 KB per workgroup.
WS = 32
APAD = 8                      # f16 of padding per LDS activation row
# Decode-ahead: after the barrier the decoding waves decode the next phase into a second weight tile.
# So their VALU overlaps every wave's MMAs of the current phase. Needs two weight tiles in LDS: KSUB=32 at 128 x 256.
# Off for the IQ3 formats: their LDS grid lookups contend with the MMA fragment loads.
# Off for Q5_K: its read-ahead loads meet s_waitcnt vmcnt(0) drains inside the K loop, which serialize the prefetch.
DECAHEAD_FMTS = ("iq4xs", "q4k", "q6k")
# Short-K residual GEMMs keep the plain schedule: with decode-ahead, 45% of wave time is s_waitcnt vmcnt(0) in the K loop.
# These full drains serialize the read-ahead.
DECAHEAD_SKIP = {("iq4xs", "kres", 24), ("q4k", "kres", 24)}
# The swiglu epilogue through lds_epilogue (one barrier, wave-private slabs, 4-row vector loads/stores).
# swiglu_epilogue issues one dependent gate load per element in a rolled loop; with one workgroup per WGP nothing hides it.
# IQ4_XS keeps swiglu_epilogue: neutral at 2x the code.
SWEPI_FMTS = ("iq3s", "iq3xxs", "f16")
# STAGGER: in the first round the second workgroup on each WGP runs STAGGER barriers before it starts.
# So co-resident workgroups drift out of lockstep; else on short-K kres they all hit the residual epilogue at once.
# 8000 barriers is ~380k cycles of offset, past the ~150-200k cycle epilogue burst. Results are unchanged.
# Only grids of >= STG_MINWG workgroups (8 rounds at 2 per WGP) stagger, so the delay is a one-time cost.
STAGGER = 8000
STG_MINWG = 320
STG_NWGP = 20                 # WGPs on gfx1151
# rhs-outer MMA order with a fence every n rhs groups, per format (fewer live fragments at 4 x 2)
RHSO_FMTS = {"q4k": 1, "q5k": 1}

# f16 of padding per decoded weight row (APAD likewise per activation row).
# Unpadded rows are 128 B apart at KSUB=64, so a 16-lane lhs fragment load hits 2 bank groups (8-way conflicts).
WPAD = 8
# KSL: the k steps of a phase straight-line in program order (no read-ahead).
# Low CSE then shares the fragment loads' address math (13 VALU per 8 WMMAs in the rolled loop).
# A fence keeps step s+1's loads after step s's MMAs.
# Off for Q5_K: the latch copies' vmcnt(0) sits between the steps; with DECLOAD, at the 144-VGPR cap, ~48 moves appear.
# Q6_K, Q2_K, Q8_0 and IQ2_* are not measured.
KSL_FMTS = ("iq4xs", "q4k", "iq3s", "iq3xxs", "q3k", "f16")
# DECLOAD (on where KSL is): under decode-ahead only the decoding waves issue the phase's weight loads, not every wave.
# With KSL it also moves the prefetch's latch copies (and their vmcnt(0)) from between the k steps to after the last MMA.
# EPAD: pad of the LDS epilogue slab's token pitch (f32).
# With pitch TM the 16 lanes storing a fragment row are 128 B apart (one or two banks).
EPAD = 4
# Formats that run 4 x 2 waves (32 x 128 per wave) on the 128 x 256 geometry: fewer fragment loads per WMMA (1.5 -> 1.25).
# 2 x 4 (64 x 64) has fewer instructions still but drops off the issue bound (exposed latency).
# Q4_K / Q5_K fit in VGPRs at 4 x 2 only with Q4FMIX, IQ3_S only with the word-path decode (w3).
# IQ3_XXS swiglu at 4 x 2 needs the LDS epilogue (SWEPI_FMTS).
WAVE_FMTS = ("iq4xs", "iq3xxs", "q3k", "iq3s", "q4k", "q5k", "f16")


@dataclass(frozen=True)
class Tile:
    """The knobs of one tile GEMM. default_tile() gives the shipped choice; a tuner may pick another legal Tile.
    No knob changes the per-accumulator MMA order or the decode arithmetic, so every legal Tile gives bit-identical output."""
    bm: int = 128              # weight rows per workgroup
    bn: int = 256              # tokens per workgroup
    wm: int = 4                # waves along rows
    wn: int = 4                # waves along tokens
    ksub: int = 64             # K per phase
    decahead: bool = False     # decode the next phase during this phase's MMAs (two weight tiles)
    ksl: bool = False          # straight-line k steps (KSL_FMTS)
    decload: bool = False      # under decode-ahead only the decoding waves load weights (DECLOAD)
    rhs_outer: bool = False    # rhs-outer MMA order
    rhs_fence: int = 0         # fence every n rhs groups under rhs_outer
    w3: bool = False           # IQ3 word-path decode (gen_gemm_decode VDEC_W, IQ3_U8F, VDECW_FR)
    q4fmix: bool = False       # Q4_K/Q5_K subtract-and-narrow through v_fma_mix (gen_gemm_decode Q4FMIX)
    swepi: bool = False        # swiglu through lds_epilogue
    stagger: int = STAGGER     # barriers the second workgroup per WGP waits in the first round
    gstage: bool = False       # stage activations through the general segment map even where the row map fits
    dbuf: bool = False         # double-buffered LDS tiles, one barrier per phase (decode / stage phase k+1 while multiplying k)
    tokfast: bool = False      # launch order token tiles fastest: the token tiles of a row block run together (L2 shares weights)
    afrag: bool = False        # activations not staged in LDS: the MMA's B fragments load straight from the input (needs dbuf)
    atiled: bool = False       # afrag input in fragment-major tiles: tile (t/16, k/16) is 256 contiguous halves (k fastest)
    ecoal: bool = False        # tall epilogue: 4 lanes per token (16 rows, 64 contiguous bytes), 8 tokens per access, instead of
                               # 2 lanes per token 32 bytes apart (fewer, larger global requests for the stores and residual loads)
    f16p: bool = False         # IQ3: biased decode bytes as f16 subnormal pairs (gen_gemm_decode.IQ3_F16P) instead of
                               # v_cvt_f32_ubyteN (-11..-15 loop VALU)
    sgtab: bool = False        # IQ3_XXS: sign words from the gen_gemm_decode.IQ3_SGTAB LDS table (with lhs_stream=1; not
                               # the fused ffn / swiglu, whose fragment-major-output forms lose +1.3%)
    f16s: bool = False         # IQ3: signed f16 subnormal pairs from one v_perm per grid word (gen_gemm_decode.IQ3_F16S)
    sgtw: int = 2              # IQ3 sign-table words per entry (gen_gemm_decode.IQ3_SGTAB_W; 1: both nibbles in one word)
    gaddr: bool = False        # IQ3: grid / sign-table lookups at pre-scaled byte offsets on byte views
                               # (gen_gemm_decode.IQ3_GADDR; -15..-23 loop VALU)
    esr: int = 16              # tall epilogue slab rows (32 with ecoal: 8 lanes per token, full 128-byte rows per access)
    respre: int = 0            # kres tall epilogue: issue each slab group's residual loads this many groups ahead (0: at use)
    ffn_inload: bool = False   # mixed ffn: each wave loads its own format's weights at its decode (no carried union)
    ffn: bool = False          # kind "ffn": gate and up fused, weight tile rows 0-63 gate / 64-127 up of the same 64 rows
                               # (the weight binding spans ffn_gate and the ffn_up right after it: up row r is row m_rows + r)
    tallepi: bool = False      # afrag kstore / kres: the LDS epilogue in 16-row slabs in the weight tile (else fragment stores)
    tout: bool = False         # swiglu: store the output fragment-major (atiled layout over K = rows) for an afrag consumer
    stg_minwg: int = STG_MINWG  # smallest grid that staggers (afrag: every grid; lockstep costs it 6-9%)
    wlate: bool = False        # afrag: next phase's weight loads in the last k step (registers free in the MMA section)
    b0early: bool = False      # afrag: issue k step 0's B fragments at the top of the phase, before the decode hides their latency
    bpre: int = 0              # afrag: each k step loads the first bpre of the next step's B fragments before its MMAs (in-phase)
    lhs_stream: int = 0        # straight-line k steps: load the B fragments first, then each A fragment just before its MMAs,
                               # with a scheduling fence every lhs_stream A fragments (few A fragments live: tall per-wave tiles)

    @property
    def tm(self):
        return self.bm // self.wm

    @property
    def tn(self):
        return self.bn // self.wn

    @property
    def nwave(self):
        return self.wm * self.wn

    @property
    def lanes(self):
        return WS * self.nwave

    @property
    def rowgrp(self):
        """m_tiles per workgroup (ffn: output rows, half the weight tile)."""
        return self.bm // (32 if self.ffn else 16)

    @property
    def apl(self):
        """Lanes staging one token row of the activation tile."""
        return self.lanes // self.bn


# set per emitted shape by emit_prefill_pp._emit_gen (O16_MT): kstore outputs stored as f16 for f16 consumers
OUT16 = False

def default_tile(fmt, kind, kb, geom=None):
    """The shipped Tile for fmt / kind at k_blocks kb; geom=(BM, BN, WM, WN) overrides the geometry (16-row tiles)."""
    if geom:
        bm, bn, wm, wn = geom
    else:
        bm, bn, wm, wn = (128, 256, 4, 2) if fmt in WAVE_FMTS else (128, 256, 4, 4)
    # the decoding lanes must be whole waves (a wave-uniform branch): not so for the 16-row tiles, which keep the plain schedule
    decahead = fmt in DECAHEAD_FMTS and (fmt, kind, kb) not in DECAHEAD_SKIP and bm % WS == 0
    # IQ3 word-path decode, bit-identical: IQ3_XXS, and IQ3_S at 4 x 2.
    # At 4 x 4 IQ3_S gains nothing: the longer dependent chain is exposed between barriers, where every wave decodes at once.
    w3 = fmt == "iq3xxs" or (fmt == "iq3s" and (wm, wn) == (4, 2))
    return Tile(bm, bn, wm, wn,
                # KSUB=64 (32 for decode-ahead's two weight tiles): at 128 the 128 x 256 tiles need ~104 KB of LDS
                ksub=32 if decahead else 64,
                decahead=decahead, ksl=fmt in KSL_FMTS, decload=fmt in KSL_FMTS,
                rhs_outer=fmt in RHSO_FMTS, rhs_fence=RHSO_FMTS.get(fmt, 0), w3=w3,
                # Q4_K: this lets Q4_K run at 4 x 2 without spills
                q4fmix=fmt in ("q4k", "q5k"),
                swepi=kind == "swiglu" and fmt in SWEPI_FMTS and bm // wm == 32,
                # decode-free f16 GEMMs read 2 B per weight: the token tiles of a row block run together so L2 serves 7 of 8
                tokfast=fmt == "f16")


def check(t):
    """Raise ValueError if t cannot be emitted (shape rules only; the compiler rejects LDS or VGPR overflow)."""
    rules = [
        (t.bm % (16 * t.wm) == 0 and t.bn % (16 * t.wn) == 0, "per-wave tile is not whole 16 x 16 fragments"),
        (t.lanes >= t.bm, "lanes do not cover the weight rows"),
        (t.ksub in (32, 64, 128) and (not t.decahead or t.bm % WS == 0), "KSUB or decode-ahead geometry"),
        (t.ksub // 32 * t.bm <= t.lanes, "not enough lanes to decode a phase in one pass"),
        (not t.swepi or t.tm == 32, "the LDS swiglu epilogue needs 32 rows per wave"),
        (not (t.dbuf and t.decahead), "double buffering replaces decode-ahead"),
        (not t.afrag or t.ksl, "activation fragments from global need the straight-line k steps"),
        (not t.bpre or (t.afrag and not t.rhs_outer), "B prefetch is for afrag without rhs-outer"),
        (not t.b0early or (t.afrag and not t.dbuf and not t.decahead), "early step-0 B is for afrag without dbuf/decode-ahead"),
        (not t.tout or (t.afrag and not t.swepi), "tiled swiglu output is the plain swiglu epilogue of an afrag tile"),
        (not t.tallepi or t.tm > 32, "the tall LDS epilogue is for waves over 32 rows"),
        (not t.ffn or (t.afrag and t.tallepi and t.bm == 128 and t.wm == 1 and not t.dbuf and not t.decahead),
         "the fused gate / up GEMM is an afrag 128-row tile (64 gate + 64 up rows) with the tall epilogue"),
        (not t.wlate or (t.afrag and not t.dbuf and not t.decahead and not t.rhs_outer), "late weight loads are for afrag without dbuf/decode-ahead/rhs-outer"),
    ]
    for ok, why in rules:
        if not ok:
            raise ValueError(f"{t}: {why}")


def configure(fmt, t):
    """Set gen_gemm_decode's decode geometry and switches for fmt under tile t."""
    G.KSUB = t.ksub
    G.PAD = WPAD
    G.ROWP = t.ksub + G.PAD
    G.PH = 256 // t.ksub
    G.GPP = t.ksub // 32
    # one group per decoding lane (q4k/q5k pick the nibble at run time)
    G.GPL = 1
    G.Q4_HDR = fmt in ("q4k", "q5k")
    G.VDEC_W = G.IQ3_U8F = G.VDECW_FR = t.w3
    # the shorter sign chain: IQ3_XXS -0.6% (ffn, kres, swiglu); IQ3_S -0.6..-0.9% with HRX patch 0008 (+0.7..2.5% before it:
    # the allocator's concat fallback copied an A fragment once per phase)
    G.IQ3_SGN2 = t.w3
    G.IQ3_SGTAB = t.sgtab
    G.IQ3_F16P = t.f16p
    G.IQ3_GADDR = t.gaddr
    G.IQ3_SGTAB_W = t.sgtw
    G.IQ3_F16S = t.f16s
    # IQ2: the same word path and sign chain (IQ2_W): -5.6% clock-free
    G.IQ2_W = fmt in ("iq2xxs", "iq2xs")
    if G.IQ2_W:
        G.VDEC_W = G.IQ3_U8F = G.VDECW_FR = G.IQ3_SGN2 = True
    G.Q4FMIX = t.q4fmix
    G.LR = t.bm
    G.NW = t.nwave // 2        # table-staging stride 64*NW = LANES


def gen(fmt, kind="kstore", tile=None, masked=False):
    """Return the kernel text for fmt; kind as gen_gemm_decode.gen() plus "kqg". tile defaults to default_tile().
    masked: the token count (config "tokens") need not be a multiple of BN. The last token tile clamps its activation loads
    and skips every per-token load and store past it, so the valid tokens are computed exactly as unmasked."""
    t = tile or default_tile(fmt, kind, 0)
    check(t)
    # Measured 2026-10-03: decode-ahead outside DECAHEAD_FMTS gives nondeterministic output (IQ3_XXS kres differed on 1 of
    # 4 runs; IQ3_S, IQ2_XS also seen); the cause is not traced. Within DECAHEAD_FMTS 448 hashed runs agreed.
    if t.decahead and fmt not in DECAHEAD_FMTS:
        raise ValueError("decode-ahead is only verified for " + ", ".join(DECAHEAD_FMTS))
    if masked and t.tm != 32:
        raise ValueError(f"{t}: a masked token tile needs the LDS epilogue (32 rows per wave)")
    # ffn of mixed formats: fmt "<gate fmt>:<up fmt>"
    fmt, fmt_up = fmt.split(":") if ":" in fmt else (fmt, None)
    if fmt_up is not None and (kind != "ffn" or fmt_up == fmt):
        raise ValueError("a gate:up format pair is for kind ffn with two formats")
    configure(fmt, t)
    return _gen(fmt, kind, t, masked, fmt_up)


# dequant kernels: 256-wide K blocks per work item (the table staging and pipeline fill amortize over them), and the
# persistent grid (one workgroup per WGP: 20 KB of LDS fits beside two GEMM workgroups)
DQ_BLOCKS = 4
DQ_WGS = 20

# Column split with the NPU (emit_prefill_pp NPU_SPLIT): the GPU computes output rows [0, m_tiles * 16) of a wider
# matrix; OSTRIDE (> 0) is the full row count, the output's (and kres residual's) token stride. kstore / kres.
OSTRIDE = 0


# NPU weights (emit_prefill_pp NPU_SPLIT): with DQ_BFP = (ks, passes), the dequant kind writes the NPU's BFP16 weight
# stream (gen_bfp16_encode "wgt", 64-row columns) instead of f16 rows; K must be 8 * passes * sum(ks).
# Its grid is dq_wgs(): one workgroup per item.
DQ_BFP = None


def dq_wgs(m_tiles, k_blocks, t, fmt):
    """Workgroups of the DQ_BFP dequant: (row groups) x (K groups of DQ_BLOCKS 256-wide blocks)."""
    return m_tiles // t.rowgrp * (k_blocks // G.FMTS[fmt].get("kdiv", 1) // DQ_BLOCKS)


def orw():
    return "%o_rows" if OSTRIDE else "%m_rows"


def _gen(fmt, kind, t, masked, fmt_up=None):
    BM, BN, WM, WN, TM, TN = t.bm, t.bn, t.wm, t.wn, t.tm, t.tn
    FM, FN, NWAVE, LANES, ROWGRP, APL = TM // 16, TN // 16, t.nwave, t.lanes, t.rowgrp, t.apl
    DECAHEAD, KSL, DECLOAD, RHS_OUTER, RHS_FENCE = t.decahead, t.ksl, t.decload, t.rhs_outer, t.rhs_fence
    DBUF = t.dbuf
    AFRAG = t.afrag
    F = G.FMTS[fmt]
    ksub = t.ksub
    bb, (loads, compute) = F["bb"], F["decode"]
    # mixed ffn: the up rows decode with fmt_up's decoder, generated under its own configuration (its w3 / q4fmix
    # switches from its own default tile); every lane loads its row of both (no branch around the loads: a branch drains
    # vmcnt(0) at its join) and carries both, each wave decodes the one its rows are
    MX = fmt_up is not None
    if MX:
        Fu = G.FMTS[fmt_up]
        bbu, (loads_u, compute_u) = Fu["bb"], Fu["decode"]
        du = default_tile(fmt_up, "swiglu", 20)
        tu = dataclasses.replace(t, w3=du.w3, q4fmix=du.q4fmix)
        if set(F["extra"]) & set(Fu["extra"]):
            raise ValueError(f"{fmt}:{fmt_up}: both formats need the same table binding")

        def as_up(fn):
            configure(fmt_up, tu)
            try:
                return fn()
            finally:
                configure(fmt, t)
    kr = kind == "kres"
    sw = kind == "swiglu"
    # kqg: the attention q projection (rows = heads x [256 q | 256 gate]) writes q and gate to [tokens][heads*256] buffers.
    # Same values as the separate yah_unpack_qg pass, so bit-identical.
    qg = kind == "kqg"
    # dequant: decode the weights to f16 [rows][K] once with this tile's decode (the decode-free GEMMs read that)
    dq = kind == "dequant"
    # ffn: ffn_gate and ffn_up (same format) in one GEMM, out = f16(silu(gate) * up), the swiglu GEMM's scalar ops
    ff = kind == "ffn"
    assert ff == t.ffn, "kind ffn needs Tile.ffn"
    bufs = (["weight"] + F["extra"] + (Fu["extra"] if MX else []) + ["input"] + (["gate"] if sw else []) + (["resid"] if kr else [])
            + ["wstage", "ostage", "output"] + (["gate_out"] if qg else []))
    sym = (f"yah_ffn_gemm_{fmt}" + (f"_{fmt_up}" if MX else "") + ("_swiglu" if sw else "") + ("_kres" if kr else "") + ("_kqg" if qg else "")
           + ("_ffn" if ff else ""))
    if dq:
        assert not (DECAHEAD or DBUF or masked)
        bufs = ["weight"] + F["extra"] + ["output"]
        sym = f"yah_dequant_{fmt}" + ("_bfp16" if DQ_BFP else "")
    slots = G.GPP                   # decoding lane groups of BM per phase
    arow = ksub + APAD              # f16 per LDS activation row
    aseg = ksub // 8                # 16-byte segments per token row
    # Activation staging. Legacy map (every shipped tile): each lane loads aspl segments of one token row.
    # General map (any other BN): lanes walk the tile's BN * aseg segments in order, nsl per lane, so neighbouring lanes
    # read neighbouring 16-byte segments of a row; lanes past the end repeat the last segment (identical stores).
    legacy = LANES % BN == 0 and aseg % (LANES // BN) == 0 and not t.gstage
    aspl = aseg // APL if legacy else 0      # segments one lane loads (legacy map)
    nsl = aspl if legacy else -(-BN * aseg // LANES)
    V8 = "vector<8xf32>"
    VF = "vector<16xf16>"
    L = []
    e = L.append
    e(f"// GENERATED by tools/gen_gemm_tile.py {fmt} {kind} (KSUB={ksub}) -- edit the generator.")
    e("//")
    e(f"// Tile GEMM for {fmt}: {NWAVE} wave32 waves over a {BM} x {BN} tile, {TM} x {TN} per wave,")
    e("// decoded weights and staged activations both in LDS. See the generator.")
    e(f"amdgpu.target<gfx1151> @yah_tile_w32 {{subgroup_size = {WS}}}")
    e("")
    for c in ("m_tiles", "k_blocks", "token_tiles") + (("tokens",) if masked else ()):
        e(f"config.decl @{sym}.{c} : %value: index where [range(%value, 1, 4096)]")
    e("")
    e(f"kernel.def target(@yah_tile_w32) @{sym}() {{")
    e("  %unit = index.constant 1 : index")
    e(f"  %m_tiles = config.get @{sym}.m_tiles : index")
    e(f"  %token_tiles = config.get @{sym}.token_tiles : index")
    e(f"  %wgs = index.constant {LANES} : index")
    e(f"  %rowgrp = index.constant {ROWGRP} : index")
    e("  %m_groups = index.div %m_tiles, %rowgrp : index")
    if dq:
        # one workgroup per (row group, K phase)
        e(f"  %kbl0 = config.get @{sym}.k_blocks : index")
        e(f"  %kdivl = index.constant {F.get('kdiv', 1)} : index")
        e("  %kbl = index.div %kbl0, %kdivl : index")
        # DQ_BLOCKS 256-wide blocks per workgroup (k_blocks is a multiple of it: 20 and 68 here)
        e(f"  %cdqb = index.constant {DQ_BLOCKS} : index")
        e("  %kgl = index.div %kbl, %cdqb : index")
        if DQ_BFP:
            # one workgroup per (row group, K group) item: the NPU weights are encoded with the GPU otherwise idle
            e("  %dqw = index.mul %m_groups, %kgl : index")
        else:
            # persistent: DQ_WGS workgroups (one per WGP) walk the (row group, K group) items, so the dequant fits beside a
            # running GEMM (whose workgroups would otherwise all launch first) and finishes within it
            e(f"  %dqw = index.constant {DQ_WGS} : index")
        e("  kernel.launch.config workgroups(%dqw, %unit, %unit) workgroup_size(%wgs, %unit, %unit) : index")
    else:
        e("  kernel.launch.config workgroups(%m_groups, %token_tiles, %unit) workgroup_size(%wgs, %unit, %unit) : index")
    e("} launch(" + ", ".join(f"%{b}: buffer" for b in bufs) + ") {")
    e("  %base = index.constant 0 : offset")
    for v in sorted({0, 1, 2, 4, 6, 7, 8, 16, 32, 48, 63, 64, 80, 96, 112, 127, 128, 224, 255, 256, 512, BM, BM - 1, BN}):
        e(f"  %c{v} = index.constant {v} : index")
    # the same i32 constants gen_gemm_decode defines (q8_0 needs 18 and 34)
    for v in (0, 1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 18, 21, 24, 28, 32, 34, 48, 63, 64, 66, 74, 104, 106, 127, 128, 192, 255):
        e(f"  %c{v}i = scalar.constant {v} : i32")
    e(f"  %cbb = index.constant {bb} : index")
    e(f"  %cbbh = index.constant {bb // 2} : index")
    e(f"  %cbbi = scalar.constant {bb} : i32")
    e(f"  %cwtok = index.constant {BN} : index")
    e(f"  %cksub = index.constant {ksub} : index")
    e(f"  %cph = index.constant {G.PH} : index")
    e(f"  %ccolmax = index.constant {ksub - 32} : index")
    e(f"  %cgppi = scalar.constant {G.GPP} : i32")
    e("  %m = index.constant 16 : index")
    e("  %n = index.constant 16 : index")
    e("  %k = index.constant 16 : index")
    e(f"  %m_tiles = config.get @{sym}.m_tiles : index")
    # q8_0's k_blocks counts 32-wide blocks; the decode walks 256-wide ones (bb=272 = 8 x 34), as in gen_gemm_decode.
    # Without the division the kernel reads 8x past the weights and hangs the ring.
    kdiv = F.get("kdiv", 1)
    if kdiv == 1:
        e(f"  %k_blocks = config.get @{sym}.k_blocks : index")
    else:
        e(f"  %k_blocks_cfg = config.get @{sym}.k_blocks : index")
        e(f"  %ckdiv = index.constant {kdiv} : index")
        e("  %k_blocks = index.div %k_blocks_cfg, %ckdiv : index")
    e(f"  %token_tiles = config.get @{sym}.token_tiles : index")
    e("  %ktot = index.mul %k_blocks, %c256 : index")
    if masked:
        e(f"  %tokens = config.get @{sym}.tokens : index")
    else:
        e("  %tokens = index.mul %token_tiles, %cwtok : index")
    e("  %m_rows = index.mul %m_tiles, %c16 : index")
    if OSTRIDE:
        assert kind in ("kstore", "kres") and not t.tout, "the NPU column split covers kstore / kres"
        e(f"  %o_rows = index.constant {OSTRIDE} : index")
    e("  %bpr = index.mul %k_blocks, %cbb : index")
    e("  %hpr = index.mul %k_blocks, %cbbh : index")
    if MX:
        # gate rows (bb bytes per block) then up rows (bbu)
        e(f"  %cbbs = index.constant {bb + bbu} : index")
        e("  %bprs = index.mul %k_blocks, %cbbs : index")
        e("  %w_bytes = index.mul %m_rows, %bprs : index")
        e("  %w_halfs = index.div %w_bytes, %c2 : index")
    elif ff:
        # gate rows then up rows
        e("  %w_rows = index.mul %m_rows, %c2 : index")
        e("  %w_bytes = index.mul %w_rows, %bpr : index")
        e("  %w_halfs = index.mul %w_rows, %hpr : index")
    else:
        e("  %w_bytes = index.mul %m_rows, %bpr : index")
        e("  %w_halfs = index.mul %m_rows, %hpr : index")
    e("  %w_last = index.sub %w_bytes, %c1 : index")
    e("  %w_lim = index.sub %w_bytes, %c16 : index")
    for nb in (4, 8, 16):
        e(f"  %cw{nb} = index.constant {nb} : index")
        e(f"  %w_lim{nb} = index.sub %w_bytes, %cw{nb} : index")
    e("  %w_half_last = index.sub %w_halfs, %c1 : index")
    e(f"  %out_total = index.mul {orw()}, %tokens : index")
    e("  %cagpad = index.constant 0 : index")
    e("  %apitch = index.add %ktot, %cagpad : index")
    e("  %a_total = index.mul %tokens, %apitch : index")
    e("  %a_last8 = index.sub %a_total, %c8 : index")
    e("  %a_layout = encoding.layout.strided [%c1, %ktot] : encoding<layout>")
    e("  " + ", ".join(f"%{b}_na" for b in bufs) + " = buffer.assume.noalias "
      + ", ".join(f"%{b}" for b in bufs) + " : " + ", ".join(["buffer"] * len(bufs)))
    e("  %w_view = buffer.view %weight_na[%base] : buffer -> view<[%w_bytes]xi8>")
    e("  %w_f16_view = buffer.view %weight_na[%base] : buffer -> view<[%w_halfs]xf16>")
    if not dq:
        e("  %a_flat = buffer.view %input_na[%base] : buffer -> view<[%a_total]xf16>")
    if AFRAG:
        e("  %a_rhs = buffer.view %input_na[%base] : buffer -> view<[%ktot]x[%tokens]xf16, %a_layout>")
        if t.atiled:
            e("  %a_ktiles = index.div %ktot, %c16 : index")
            e("  %a_tlay = encoding.layout.strided [%c1, %c16] : encoding<layout>")
    # LDS: decoded weight tile and staged activation tile
    wl_b = BM * G.ROWP * 2 * (2 if DECAHEAD or DBUF or dq else 1)
    # the tall epilogue's slab rows (kstore / kres; coalesced only): its slabs live in the weight tile
    esr = t.esr if t.tallepi and t.ecoal and not (ff or sw or qg or masked) else 16
    wl_b = max(wl_b, NWAVE * (esr + EPAD) * 16 * 4) if esr != 16 else wl_b
    e(f"  %wl_bytes = index.constant {wl_b} : offset")
    e(f"  %wl_tb = index.constant {BM * G.ROWP * 2} : index")
    e("  %wl = buffer.alloca<workgroup> align(16) %wl_bytes : buffer")
    e(f"  %wl_view = buffer.view %wl[%base] : buffer -> view<{BM}x{G.ROWP}xf16>")
    if not dq:
        # the LDS epilogue's wave-private TM x 16 f32 slabs live in the activation tile
        al_bytes = 256 if AFRAG else BN * arow * 2 * (2 if DBUF else 1)
        slabs = NWAVE * (TM + EPAD) * 16 * 4
        if TM == 32:
            al_bytes = max(al_bytes, slabs)
        if sw and not t.swepi:
            # swiglu_epilogue's per-wave slabs; narrow tiles need more than the activation tile holds
            al_bytes = max(al_bytes, NWAVE * 16 * swiglu_slab(t, arow) * 4)
        e(f"  %al_bytes = index.constant {al_bytes} : offset")
        e("  %al = buffer.alloca<workgroup> align(16) %al_bytes : buffer")
        e(f"  %al_rows = buffer.view %al[%base] : buffer -> view<{BN}x{arow}xf16>")
        e(f"  %carow = index.constant {arow} : index")
        e(f"  %cksubi = index.constant {ksub} : index")
        e("  %al_layout = encoding.layout.strided [%c1, %carow] : encoding<layout>")
        e(f"  %al_t = buffer.view %al[%base] : buffer -> view<{ksub}x{BN}xf16, %al_layout>")
        e(f"  %al_tb = index.constant {BN * arow * 2} : index")
    if t.tokfast:
        # linear id -> (row block, token tile) with the token tile fastest; the same grid, a different order
        e("  %wg_rx = kernel.workgroup.id<x> : index")
        e("  %wg_ry = kernel.workgroup.id<y> : index")
        e("  %wg_gx = kernel.workgroup.count<x> : index")
        e("  %wg_gy = kernel.workgroup.count<y> : index")
        e("  %wg_l0 = index.mul %wg_ry, %wg_gx : index")
        e("  %wg_lin = index.add %wg_l0, %wg_rx : index")
        e("  %wg_x = index.div %wg_lin, %wg_gy : index")
        e("  %wg_y = index.rem %wg_lin, %wg_gy : index")
    elif dq:
        e("  %dq_wid = kernel.workgroup.id<x> : index")
        e("  %dq_nwg = kernel.workgroup.count<x> : index")
        e(f"  %dq_cdqb = index.constant {DQ_BLOCKS} : index")
        e("  %dq_kg = index.div %k_blocks, %dq_cdqb : index")
        e(f"  %rowgrp_k = index.constant {ROWGRP} : index")
        e("  %dq_mg = index.div %m_tiles, %rowgrp_k : index")
        e("  %dq_nit = index.mul %dq_mg, %dq_kg : index")
        e("  scf.for %dq_item = [%dq_wid to %dq_nit step %dq_nwg] {")
        e("  %wg_x = index.div %dq_item, %dq_kg : index")
        e("  %wg_y = index.rem %dq_item, %dq_kg : index")
    else:
        e("  %wg_x = kernel.workgroup.id<x> : index")
        e("  %wg_y = kernel.workgroup.id<y> : index")
    e("  %tid = kernel.workitem.id<x> : index")
    e(f"  %wave = index.div %tid, %c{WS} : index")
    e(f"  %cwn = index.constant {WN} : index")
    e("  %wr = index.div %wave, %cwn : index")
    e("  %wt = index.rem %wave, %cwn : index")
    e("  %stg_row0 = index.cmp eq, %wg_y, %c0 : index")
    # the second workgroup on each WGP in the first round: linear dispatch ids [NWGP, 2*NWGP) (co-resident pairs are (i, i + 20))
    e("  %stg_gx = kernel.workgroup.count<x> : index")
    e("  %stg_rx = kernel.workgroup.id<x> : index")
    e("  %stg_ry = kernel.workgroup.id<y> : index")
    e("  %stg_l0 = index.mul %stg_ry, %stg_gx : index")
    e("  %stg_lin = index.add %stg_l0, %stg_rx : index")
    e("  %stg_gy = kernel.workgroup.count<y> : index")
    e("  %stg_tot = index.mul %stg_gx, %stg_gy : index")
    e(f"  %stg_minwg = index.constant {t.stg_minwg} : index")
    e("  %stg_big = index.cmp uge, %stg_tot, %stg_minwg : index")
    e(f"  %stg_w = index.constant {STG_NWGP} : index")
    e(f"  %stg_w2 = index.constant {2 * STG_NWGP} : index")
    e("  %stg_ge = index.cmp uge, %stg_lin, %stg_w : index")
    e("  %stg_lt = index.cmp ult, %stg_lin, %stg_w2 : index")
    e(f"  %stg_n = index.constant {0 if dq else t.stagger} : index")
    e("  %stg_n1 = scf.select %stg_ge, %stg_n, %c0 : index")
    e("  %stg_n2 = scf.select %stg_lt, %stg_n1, %c0 : index")
    e("  %stg_iters = scf.select %stg_big, %stg_n2, %c0 : index")
    # workgroup-uniform trip count: every wave of the workgroup takes it
    e("  scf.for %stg_i = [%c0 to %stg_iters step %c1] {")
    e("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    e("  }")
    e(f"  %wg_row = index.mul %wg_x, %c{BM // 2 if ff else BM} : index")
    e(f"  %ctm = index.constant {TM} : index")
    e(f"  %ctn = index.constant {TN} : index")
    e("  %wr_off = index.mul %wr, %ctm : index")
    e("  %wt_off = index.mul %wt, %ctn : index")
    e("  %m_origin = index.add %wg_row, %wr_off : index")
    e("  %wtb = index.mul %wg_y, %cwtok : index")
    e("  %token_base = index.add %wtb, %wt_off : index")
    # decode lane map: lane tid decodes weight row tid % BM, group slot tid / BM of the phase; slots >= GPP idle
    e(f"  %l64 = index.rem %tid, %c{BM} : index")
    e(f"  %slot = index.div %tid, %c{BM} : index")
    e(f"  %drow = index.min %l64, %c{BM - 1} : index")
    e("  %drow_i = index.cast %drow : index to i32")
    e("  %wg_row_i = index.cast %wg_row : index to i32")
    if MX:
        # both rows: gate row drow % 64 at row_off_i, up row drow % 64 at row_off_u (after the gate tensor)
        e("  %drow_g = index.rem %drow, %c64 : index")
        e("  %ff_up = index.cmp uge, %drow, %c64 : index")
        e("  %drow_gi = index.cast %drow_g : index to i32")
        e("  %grow_i = scalar.addi %wg_row_i, %drow_gi : i32")
    elif ff:
        # weight tile row drow: gate row drow (< 64) or up row drow - 64 of the workgroup's 64 rows; the up rows are
        # m_rows further on in the binding (no branch: a branch around the loads drains vmcnt(0) at its join every phase)
        e("  %drow_g = index.rem %drow, %c64 : index")
        e("  %ff_up = index.cmp uge, %drow, %c64 : index")
        e("  %ff_upr = scf.select %ff_up, %m_rows, %c0 : index")
        e("  %drow_t = index.add %drow_g, %ff_upr : index")
        e("  %drow_gi = index.cast %drow_t : index to i32")
        e("  %grow_i = scalar.addi %wg_row_i, %drow_gi : i32")
    else:
        e("  %grow_i = scalar.addi %wg_row_i, %drow_i : i32")
    e("  %k_blocks_i = index.cast %k_blocks : index to i32")
    e("  %bpr_i = scalar.muli %k_blocks_i, %cbbi : i32")
    e("  %row_off_i = scalar.muli %grow_i, %bpr_i : i32")
    if MX:
        e(f"  %cbbu_i = scalar.constant {bbu} : i32")
        e("  %bpru_i = scalar.muli %k_blocks_i, %cbbu_i : i32")
        e("  %m_rows_i = index.cast %m_rows : index to i32")
        e("  %gate_tot_i = scalar.muli %m_rows_i, %bpr_i : i32")
        e("  %row_off_u0 = scalar.muli %grow_i, %bpru_i : i32")
        e("  %row_off_u = scalar.addi %gate_tot_i, %row_off_u0 : i32")
    e(f"  %cslots = index.constant {slots} : index")
    e("  %slot_c = index.min %slot, %cslots : index")
    e("  %decoder = index.cmp ult, %slot, %cslots : index")
    e("  %slot_i0 = index.cast %slot_c : index to i32")
    e("  %csplit = scalar.constant 1 : i32")
    e("  %slot_i = scalar.divui %slot_i0, %csplit : i32")
    e("  %sub_i = scalar.remui %slot_i0, %csplit : i32")
    e("  %cgpl = scalar.constant 1 : i32")
    e("  %gl_i = scalar.muli %slot_i, %cgpl : i32")
    e("  %kphases = index.mul %k_blocks, %cph : index")
    if legacy:
        # activation staging map: lane tid stages segments [aseg0, aseg0+aspl) of token row tid % BN of the tile
        e(f"  %atok = index.rem %tid, %c{BN} : index")
        e(f"  %apart = index.div %tid, %c{BN} : index")
        e(f"  %caspl8 = index.constant {8 * aspl} : index")
        e("  %aseg0 = index.mul %apart, %caspl8 : index")
        e("  %atok_g = index.add %wtb, %atok : index")
        e("  %arow_g = index.mul %atok_g, %apitch : index")
    else:
        # segment s = tid + i * LANES (clamped): token row s / aseg, f16 column 8 * (s % aseg)
        e(f"  %gs_last = index.constant {BN * aseg - 1} : index")
        e(f"  %gs_aseg = index.constant {aseg} : index")
        sep = "_" if nsl > 10 else ""   # %gs1 + "0" would collide with %gs10 past ten slices
        for i in range(nsl):
            e(f"  %gs{i}{sep}c = index.constant {i * LANES} : index")
            e(f"  %gs{i}{sep}0 = index.add %tid, %gs{i}{sep}c : index")
            e(f"  %gs{i} = index.min %gs{i}{sep}0, %gs_last : index")
            e(f"  %gr{i} = index.div %gs{i}, %gs_aseg : index")
            e(f"  %gq{i} = index.rem %gs{i}, %gs_aseg : index")
            e(f"  %gc{i} = index.mul %gq{i}, %c8 : index")
            e(f"  %grt{i} = index.add %wtb, %gr{i} : index")
            e(f"  %grp{i} = index.mul %grt{i}, %apitch : index")
            e(f"  %gro{i} = index.add %grp{i}, %gc{i} : index")
    L.extend(F["setup"]())
    if MX:
        L.extend(as_up(lambda: Fu["setup"]()))
    if dq:
        # the setup's LDS tables visible, then this workgroup's phase (wg_y) decoded into the LDS tile as the GEMM does it
        e("  %z8s = scalar.constant 0 : i8")
        e("  %z8v = vector.splat %z8s : vector<8xi8>")
        e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        if DQ_BFP:
            bks, bpasses = DQ_BFP
            bpanel = sum(bpasses * 4 * GE.slab_bytes(k) for k in bks)
            nfrag = (BM // 8) * (ksub // 8)
            assert BM % 64 == 0 and ksub % 8 == 0 and nfrag <= LANES
            e(f"  %dq_bpanel = index.constant {bpanel} : index")
            e("  %dq_bcols = index.div %m_rows, %c64 : index")
            e("  %dq_tot = index.mul %dq_bcols, %dq_bpanel : index")
            e("  %dq_out = buffer.view %output_na[%base] : buffer -> view<[%dq_tot]xi8>")
        else:
            e("  %dq_tot = index.mul %m_rows, %ktot : index")
            e("  %dq_out = buffer.view %output_na[%base] : buffer -> view<[%dq_tot]xf16>")
        e("  %dq_g = index.add %wg_y, %c0 : index")
        e(f"  %dq_np = index.constant {DQ_BLOCKS * G.PH} : index")
        e("  %dq_p0 = index.mul %dq_g, %dq_np : index")
        # phase p's bytes are carried in registers (loaded during phase p - 1); the tile is double-buffered, so one
        # barrier per phase separates decode into buffer p % 2 from its copy-out (buffer p % 2 is next written at p + 2)
        def dq_loads(pfx, kp):
            e(f"  %{pfx}kb = index.div {kp}, %cph : index")
            e(f"  %{pfx}ph = index.rem {kp}, %cph : index")
            e(f"  %{pfx}kbi = index.cast %{pfx}kb : index to i32")
            e(f"  %{pfx}phi = index.cast %{pfx}ph : index to i32")
            e(f"  %{pfx}bo = scalar.muli %{pfx}kbi, %cbbi : i32")
            e(f"  %{pfx}blk = scalar.addi %row_off_i, %{pfx}bo : i32")
            e(f"  %{pfx}gg = scalar.muli %{pfx}phi, %cgppi : i32")
            e(f"  %{pfx}gb = scalar.addi %{pfx}gg, %gl_i : i32")
            Ld, wd = loads(pfx + "w_", f"%{pfx}blk", f"%{pfx}gb")
            L.extend(Ld)
            return wd
        e("  %dq_plast0 = index.add %dq_p0, %dq_np : index")
        e("  %dq_plast = index.sub %dq_plast0, %c1 : index")
        w0 = dq_loads("dq0_", "%dq_p0")
        orig = w0
        w0p = G.pack_vals(e, w0, "dq0")
        ctys = ", ".join(ty for _, ty in w0p)
        e("  " + ", ".join(f"%dqo{x}" for x in range(len(w0p))) + " = scf.for %dq_it = [%c0 to %dq_np step %c1]("
          + ", ".join(f"%dqv{x} = {nm} : {ty}" for x, (nm, ty) in enumerate(w0p)) + f") -> ({ctys}) {{")
        e("  %dq_kp = index.add %dq_p0, %dq_it : index")
        e("  %dq_kb = index.div %dq_kp, %cph : index")
        e("  %dq_ph = index.rem %dq_kp, %cph : index")
        e("  %dq_phi = index.cast %dq_ph : index to i32")
        e("  %dq_gg = scalar.muli %dq_phi, %cgppi : i32")
        e("  %dq_gb = scalar.addi %dq_gg, %gl_i : i32")
        e("  %dq_buf = index.rem %dq_it, %c2 : index")
        e("  %dq_bo0 = index.mul %dq_buf, %wl_tb : index")
        e("  %dq_bof = index.cast %dq_bo0 : index to offset")
        e(f"  %wl_dq = buffer.view %wl[%dq_bof] : buffer -> view<{BM}x{G.ROWP}xf16>")
        e("  scf.if %decoder {")
        names = G.unpack_vals(e, [(f"%dqv{x}", ty) for x, (_, ty) in enumerate(w0p)], orig)
        L.extend(l.replace("%wl_view[", "%wl_dq[") for l in compute(names, "%dq_gb"))
        e("  }")
        # the next phase's bytes, in flight while this phase is copied out
        e("  scf.schedule.fence")
        e("  %dq_kn0 = index.add %dq_kp, %c1 : index")
        e("  %dq_kn = index.min %dq_kn0, %dq_plast : index")
        wn = G.pack_vals(e, dq_loads("dqn_", "%dq_kn"), "dqn")
        e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        if DQ_BFP:
            # encode the BM x KSUB tile's 8 x 8 fragments into the NPU weight stream, one lane per fragment
            e(f"  %bf_nfrag = index.constant {nfrag} : index")
            e("  %bf_live = index.cmp ult, %tid, %bf_nfrag : index")
            e("  scf.if %bf_live {")
            for nm, v in (("c2", 2), ("c144", 144), ("c72", 72), ("csub", 4), ("cpk", sum(bks)), ("cpass", bpasses),
                          ("c8", 8), ("ckq", ksub // 8)):
                e(f"    %bf_{nm} = index.constant {v} : index")
            e("    %bf_rr = index.div %tid, %bf_ckq : index")
            e("    %bf_kk = index.rem %tid, %bf_ckq : index")
            e("    %bf_r0 = index.mul %bf_rr, %bf_c8 : index")
            e("    %bf_gr = index.add %wg_row, %bf_r0 : index")
            e("    %bf_g8 = index.div %bf_gr, %bf_c8 : index")
            e("    %bf_kb0 = index.mul %dq_kp, %bf_ckq : index")
            e("    %bf_kbi = index.add %bf_kb0, %bf_kk : index")
            frag = GE.emit_offset(e, "wgt", 0, bks, bpasses, 64, True, "%bf_g8", "%bf_kbi", p="bf_")
            e("    %bf_kc = index.mul %bf_kk, %bf_c8 : index")

            def bf_load(r):
                e(f"    %bf_ro{r} = index.constant {r} : index")
                e(f"    %bf_lr{r} = index.add %bf_r0, %bf_ro{r} : index")
                e(f"    %bf_xh{r} = vector.load %wl_dq[%bf_lr{r}, %bf_kc] : view<{BM}x{G.ROWP}xf16> -> vector<8xf16>")
                return f"%bf_xh{r}"

            GE.emit_encode(e, bf_load, frag, "%dq_out", "%dq_tot", p="bf_")
            e("  }")
            e("  scf.yield " + ", ".join(nm for nm, _ in wn) + f" : {ctys}")
            e("  }")
            e("  }")
            e("  kernel.return")
            e("}")
            return "\n".join(L) + "\n"
        # copy the BM x KSUB tile to out[row][K] (f16), 8 halves per store
        e("  %dq_k0 = index.mul %dq_kp, %cksub : index")
        cpr = ksub // 8
        nch = BM * cpr
        e(f"  %dq_cpr = index.constant {cpr} : index")
        e(f"  %dq_qm = index.constant {BM * cpr - 1} : index")
        for i in range(-(-nch // LANES)):
            e(f"  %dq_q{i}c = index.constant {i * LANES} : index")
            e(f"  %dq_q{i}0 = index.add %tid, %dq_q{i}c : index")
            e(f"  %dq_q{i} = index.min %dq_q{i}0, %dq_qm : index")
            e(f"  %dq_r{i} = index.div %dq_q{i}, %dq_cpr : index")
            e(f"  %dq_cq{i} = index.rem %dq_q{i}, %dq_cpr : index")
            e(f"  %dq_c{i} = index.mul %dq_cq{i}, %c8 : index")
            e(f"  %dq_v{i} = vector.load %wl_dq[%dq_r{i}, %dq_c{i}] : view<{BM}x{G.ROWP}xf16> -> vector<8xf16>")
            e(f"  %dq_gr{i} = index.add %wg_row, %dq_r{i} : index")
            e(f"  %dq_go{i} = index.mul %dq_gr{i}, %ktot : index")
            e(f"  %dq_gk{i} = index.add %dq_k0, %dq_c{i} : index")
            e(f"  %dq_ga{i} = index.add %dq_go{i}, %dq_gk{i} : index")
            e(f"  vector.store %dq_v{i}, %dq_out[%dq_ga{i}] : vector<8xf16>, view<[%dq_tot]xf16>")
        e("  scf.yield " + ", ".join(nm for nm, _ in wn) + f" : {ctys}")
        e("  }")
        e("  }")
        e("  kernel.return")
        e("}")
        return "\n".join(L) + "\n"
    e("  %z8s = scalar.constant 0 : i8")
    e("  %z8v = vector.splat %z8s : vector<8xi8>")
    e(f"  %zeros = vector.constant 0.0 : {V8}")
    e(f"  %init = vector.fragment<init> %zeros shape [%m, %n] : {V8}")
    NA = FM * FN
    types = ", ".join([V8] * NA)

    def a_loads(p, kbase):
        """Load this lane's part of its token row of a phase's activation tile as 16-byte vectors.
        Clamped, so the extra iteration's loads stay in bounds."""
        vals = []
        if not legacy:
            for i in range(nsl):
                e(f"    %{p}gq{i} = index.add %gro{i}, {kbase} : index")
                e(f"    %{p}gqc{i} = index.min %{p}gq{i}, %a_last8 : index")
                e(f"    %{p}av{i} = vector.load %a_flat[%{p}gqc{i}] : view<[%a_total]xf16> -> vector<8xf16>")
                vals.append((f"%{p}av{i}", "vector<8xf16>"))
            return vals
        e(f"    %{p}ab = index.add %arow_g, {kbase} : index")
        e(f"    %{p}ao = index.add %{p}ab, %aseg0 : index")
        for sg in range(aspl):
            e(f"    %{p}ac{sg}0 = index.constant {8 * sg} : index")
            e(f"    %{p}aq{sg} = index.add %{p}ao, %{p}ac{sg}0 : index")
            e(f"    %{p}aqc{sg} = index.min %{p}aq{sg}, %a_last8 : index")
            e(f"    %{p}av{sg} = vector.load %a_flat[%{p}aqc{sg}] : view<[%a_total]xf16> -> vector<8xf16>")
            vals.append((f"%{p}av{sg}", "vector<8xf16>"))
        return vals

    def stage_acts(names, view):
        """Store this lane's staged activation vectors into the activation tile view (rows x arow)."""
        for sg, nm in enumerate(names):
            if not legacy:
                e(f"    vector.store {nm}, {view}[%gr{sg}, %gc{sg}] : vector<8xf16>, view<{BN}x{arow}xf16>")
                continue
            e(f"    %as{sg}c = index.constant {8 * sg} : index")
            e(f"    %as{sg} = index.add %aseg0, %as{sg}c : index")
            e(f"    vector.store {nm}, {view}[%atok, %as{sg}] : vector<8xf16>, view<{BN}x{arow}xf16>")

    if DECAHEAD:
        # phase 0 decoded into weight tile 0 now; phase 1's bytes carried
        L0, w00 = loads("pf_", "%row_off_i", "%gl_i")
        L.extend(L0)
        e("  scf.if %decoder {")
        L.extend(compute([nm for nm, _ in w00], "%gl_i"))
        e("  }")
        e("  %kp1_l = index.sub %kphases, %c1 : index")
        e("  %kp1 = index.min %c1, %kp1_l : index")
        e("  %kb1 = index.div %kp1, %cph : index")
        e("  %ph1 = index.rem %kp1, %cph : index")
        e("  %kb1_i = index.cast %kb1 : index to i32")
        e("  %ph1_i = index.cast %ph1 : index to i32")
        e("  %blk1o = scalar.muli %kb1_i, %cbbi : i32")
        e("  %blk1 = scalar.addi %row_off_i, %blk1o : i32")
        e("  %gb1g = scalar.muli %ph1_i, %cgppi : i32")
        e("  %gb1 = scalar.addi %gb1g, %gl_i : i32")
        L1, wv0 = loads("p1_", "%blk1", "%gb1")
        L.extend(L1)
    elif DBUF:
        # phase 0 decoded and staged into buffer 0 now; phase 1's bytes and activations carried
        L0, w00 = loads("pf_", "%row_off_i", "%gl_i")
        L.extend(L0)
        a00 = [] if AFRAG else a_loads("pa0_", "%c0")
        # the setup's LDS tables (IQ grids) must be visible before the first decode; the loop's top barrier did this before
        e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        e("  scf.if %decoder {")
        L.extend(compute([nm for nm, _ in w00], "%gl_i"))
        e("  }")
        stage_acts([nm for nm, _ in a00], "%al_rows")
        e("  %kp1_l = index.sub %kphases, %c1 : index")
        e("  %kp1 = index.min %c1, %kp1_l : index")
        e("  %kb1 = index.div %kp1, %cph : index")
        e("  %ph1 = index.rem %kp1, %cph : index")
        e("  %kb1_i = index.cast %kb1 : index to i32")
        e("  %ph1_i = index.cast %ph1 : index to i32")
        e("  %blk1o = scalar.muli %kb1_i, %cbbi : i32")
        e("  %blk1 = scalar.addi %row_off_i, %blk1o : i32")
        e("  %gb1g = scalar.muli %ph1_i, %cgppi : i32")
        e("  %gb1 = scalar.addi %gb1g, %gl_i : i32")
        L1, wv0 = loads("p1_", "%blk1", "%gb1")
        L.extend(L1)
        e("  %kk1 = index.mul %kp1, %cksub : index")
        e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    else:
        # prefetch phase 0: weight bytes and activation row
        L0, wv0 = loads("pf_", "%row_off_i", "%gl_i")
        L.extend(L0)
    wv0u = []
    if MX and t.ffn_inload:
        del L[len(L) - len(L0):]   # no prefetch: the decode loads (in place: e appends to L)
        wv0 = []
    elif MX:
        L0u, wv0u = as_up(lambda: loads_u("pfu_", "%row_off_u", "%gl_i"))
        L.extend(L0u)
    orig0, orig0u = wv0, wv0u
    wv0 = G.pack_vals(e, wv0, "0")
    wv0u = G.pack_vals(e, wv0u, "0u")
    av0 = [] if AFRAG else a_loads("pa_", "%kk1" if DBUF else "%c0")
    carried = wv0 + wv0u + av0
    ca = ", ".join(f"%a{i} = %init : {V8}" for i in range(NA))
    ca += "".join(f", %cv{x} = {nm} : {ty}" for x, (nm, ty) in enumerate(carried))
    carried_t = types + "".join(f", {ty}" for _, ty in carried)
    res = ", ".join(f"%acc{i}" for i in range(NA)) + "".join(f", %cvo{x}" for x in range(len(carried)))
    e("  " + res + f" = scf.for %kp = [%c0 to %kphases step %c1]({ca}) -> ({carried_t})  {{")
    if not DBUF:
        e("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    e("    %kb = index.div %kp, %cph : index")
    e("    %ph = index.rem %kp, %cph : index")
    e("    %ph_i = index.cast %ph : index to i32")
    e("    %phg_i = scalar.muli %ph_i, %cgppi : i32")
    e("    %gb_i = scalar.addi %phg_i, %gl_i : i32")
    e("    %kb_k = index.mul %kp, %cksub : index")
    cur_w = [(f"%cv{x}", ty) for x, (_, ty) in enumerate(wv0)]
    cur_wu = [(f"%cv{len(wv0) + x}", ty) for x, (_, ty) in enumerate(wv0u)]
    cur_a = [f"%cv{len(wv0) + len(wv0u) + x}" for x in range(len(av0))]

    def afrag_rhs(st, j):
        """afrag: k step st's B fragment j straight from the input (defines %srhs{st}_{j})"""
        e(f"    %stc{st}_{j} = index.add %wt_off, %c{16 * j} : index")
        e(f"    %sgk{st}_{j} = index.add %kb_k, %sks{st} : index")
        e(f"    %sgt{st}_{j} = index.add %token_base, %c{16 * j} : index")
        if t.atiled:
            # tile (token / 16, k / 16) starts at ((token / 16) * (ktot / 16) + k / 16) * 256 halves
            e(f"    %sgtt{st}_{j} = index.div %sgt{st}_{j}, %c16 : index")
            e(f"    %sgkt{st}_{j} = index.div %sgk{st}_{j}, %c16 : index")
            e(f"    %sgr{st}_{j} = index.mul %sgtt{st}_{j}, %a_ktiles : index")
            e(f"    %sgi{st}_{j} = index.add %sgr{st}_{j}, %sgkt{st}_{j} : index")
            e(f"    %sgo{st}_{j} = index.mul %sgi{st}_{j}, %c512 : index")
            e(f"    %sgob{st}_{j} = index.cast %sgo{st}_{j} : index to offset")
            e(f"    %sgv{st}_{j} = buffer.view %input_na[%sgob{st}_{j}] : buffer -> view<16x16xf16, %a_tlay>")
            e(f"    %srhs{st}_{j} = vector.fragment.load<rhs> %sgv{st}_{j}[%c0, %c0] shape [%k, %n] : view<16x16xf16, %a_tlay> -> {VF}")
        else:
            # straight from the [tokens][K] input: each lane's 16 halves of its token row are contiguous
            e(f"    %srhs{st}_{j} = vector.fragment.load<rhs> %a_rhs[%sgk{st}_{j}, %sgt{st}_{j}] shape [%k, %n] : view<[%ktot]x[%tokens]xf16, %a_layout> -> {VF}")
    if DBUF:
        # the carried registers hold phase kp+1: decode and stage them into buffer (kp+1)%2 while buffer kp%2 is multiplied
        e("    %kpl = index.sub %kphases, %c1 : index")
        e("    %kp_d0 = index.add %kp, %c1 : index")
        e("    %kp_d = index.min %kp_d0, %kpl : index")
        e("    %ph_d = index.rem %kp_d, %cph : index")
        e("    %ph_di = index.cast %ph_d : index to i32")
        e("    %phg_d = scalar.muli %ph_di, %cgppi : i32")
        e("    %gb_d = scalar.addi %phg_d, %gl_i : i32")
        e("    %buf_d = index.rem %kp_d0, %c2 : index")
        e("    %buf_m = index.rem %kp, %c2 : index")
        e("    %wo_d0 = index.mul %buf_d, %wl_tb : index")
        e("    %wo_d = index.cast %wo_d0 : index to offset")
        e(f"    %wl_dec = buffer.view %wl[%wo_d] : buffer -> view<{BM}x{G.ROWP}xf16>")
        e("    %wo_m0 = index.mul %buf_m, %wl_tb : index")
        e("    %wo_m = index.cast %wo_m0 : index to offset")
        e(f"    %wl_mma = buffer.view %wl[%wo_m] : buffer -> view<{BM}x{G.ROWP}xf16>")
        e("    %ao_d0 = index.mul %buf_d, %al_tb : index")
        e("    %ao_d = index.cast %ao_d0 : index to offset")
        e(f"    %al_dec = buffer.view %al[%ao_d] : buffer -> view<{BN}x{arow}xf16>")
        e("    %ao_m0 = index.mul %buf_m, %al_tb : index")
        e("    %ao_m = index.cast %ao_m0 : index to offset")
        e(f"    %al_mma = buffer.view %al[%ao_m] : buffer -> view<{ksub}x{BN}xf16, %al_layout>")
        e("    scf.if %decoder {")
        names = G.unpack_vals(e, cur_w, orig0)
        L.extend(l.replace("%wl_view[", "%wl_dec[") for l in compute(names, "%gb_d"))
        e("    }")
        stage_acts(cur_a, "%al_dec")
    if t.b0early:
        # k step 0's B fragments now: they do not depend on the barrier, and the decode below hides their latency
        e("    %sks0 = index.constant 0 : index")
        for j in range(t.tn // 16):
            afrag_rhs(0, j)
        if t.b0early > 1:
            # step 1's prefetched B fragments too: a copy of one would otherwise drain the queue right after its load
            e("    %sks1 = index.constant 16 : index")
            for j in range(min(t.bpre, t.tn // 16)):
                afrag_rhs(1, j)
    # decode (only the decoding slots) into the weight tile
    if not DECAHEAD and not DBUF:
        e("    scf.if %decoder {")
        if MX and t.ffn_inload:
            e("    %kb_i0 = index.cast %kb : index to i32")
            e("    %blk_cg0 = scalar.muli %kb_i0, %cbbi : i32")
            e("    %blk_cg = scalar.addi %row_off_i, %blk_cg0 : i32")
            e("    %blk_cu0 = scalar.muli %kb_i0, %cbbu_i : i32")
            e("    %blk_cu = scalar.addi %row_off_u, %blk_cu0 : i32")
            e("    scf.if %ff_up {")
            Lcu, vcu = as_up(lambda: loads_u("cu_", "%blk_cu", "%gb_i"))
            L.extend(Lcu)
            L.extend(as_up(lambda: compute_u([n for n, _ in vcu], "%gb_i")))
            e("    } else {")
            Lcg, vcg = loads("cg_", "%blk_cg", "%gb_i")
            L.extend(Lcg)
            names = [n for n, _ in vcg]
        elif MX:
            e("    scf.if %ff_up {")
            names_u = G.unpack_vals(e, cur_wu, orig0u)
            L.extend(as_up(lambda: compute_u(names_u, "%gb_i")))
            e("    } else {")
        if not (MX and t.ffn_inload):
            names = G.unpack_vals(e, cur_w, orig0)
        L.extend(compute(names, "%gb_i"))
        if MX:
            e("    }")
        e("    }")
    # stage the activation row into the LDS activation tile
    for sg, nm in ([] if DBUF else enumerate(cur_a)):
        if not legacy:
            e(f"    vector.store {nm}, %al_rows[%gr{sg}, %gc{sg}] : vector<8xf16>, view<{BN}x{arow}xf16>")
            continue
        e(f"    %as{sg}c = index.constant {8 * sg} : index")
        e(f"    %as{sg} = index.add %aseg0, %as{sg}c : index")
        e(f"    vector.store {nm}, %al_rows[%atok, %as{sg}] : vector<8xf16>, view<{BN}x{arow}xf16>")
    def next_loads():
        # Next phase's loads. The fence keeps them below this phase's LDS stores.
        # Else the scheduler hoists them above the stores and then waits vmcnt(0) for them before the first store.
        if not (t.wlate and KSL):
            e("    scf.schedule.fence")
        e(f"    %kp_n0 = index.add %kp, %c{2 if DECAHEAD or DBUF else 1} : index")
        e("    %kp_last = index.sub %kphases, %c1 : index")
        e("    %kp_n = index.min %kp_n0, %kp_last : index")
        e("    %kb_n = index.div %kp_n, %cph : index")
        e("    %ph_n = index.rem %kp_n, %cph : index")
        e("    %kb_ni = index.cast %kb_n : index to i32")
        e("    %ph_ni = index.cast %ph_n : index to i32")
        e("    %blk_off0n = scalar.muli %kb_ni, %cbbi : i32")
        e("    %blk_n = scalar.addi %row_off_i, %blk_off0n : i32")
        e("    %phg_n = scalar.muli %ph_ni, %cgppi : i32")
        e("    %gb_n = scalar.addi %phg_n, %gl_i : i32")
        Ln, nxt = loads("nx_", "%blk_n", "%gb_n")
        if DECAHEAD and DECLOAD:
            # wave-uniform: the decoding lanes are whole waves
            wt = ", ".join(ty for _, ty in cur_w)
            e("    %sg_ld = kernel.subgroup.id : index")
            e(f"    %cdl = index.constant {slots * BM // WS} : index")
            e("    %ld_wave = index.cmp ult, %sg_ld, %cdl : index")
            e("    " + ", ".join(f"%nxw{x}" for x in range(len(cur_w))) + f" = scf.if %ld_wave -> ({wt}) {{")
            L.extend(Ln)
            packed = G.pack_vals(e, nxt, "n")
            e("      scf.yield " + ", ".join(nm for nm, _ in packed) + f" : {wt}")
            e("    } else {")
            e("      scf.yield " + ", ".join(nm for nm, _ in cur_w) + f" : {wt}")
            e("    }")
            nxt = [(f"%nxw{x}", ty) for x, (_, ty) in enumerate(cur_w)]
        else:
            L.extend(Ln if not (MX and t.ffn_inload) else [])
            nxt = G.pack_vals(e, nxt, "n") if not (MX and t.ffn_inload) else []
            if MX and not t.ffn_inload:
                e("    %blk_off0n_u = scalar.muli %kb_ni, %cbbu_i : i32")
                e("    %blk_n_u = scalar.addi %row_off_u, %blk_off0n_u : i32")
                Lnu, nxtu = as_up(lambda: loads_u("nxu_", "%blk_n_u", "%gb_n"))
                L.extend(Lnu)
                nxt = nxt + G.pack_vals(e, nxtu, "nu")
        if DECAHEAD:
            e("    %kp_a0 = index.add %kp, %c1 : index")
            e("    %kp_a = index.min %kp_a0, %kp_last : index")
            e("    %kk_n = index.mul %kp_a, %cksub : index")
        else:
            e("    %kk_n = index.mul %kp_n, %cksub : index")
        anx = [] if AFRAG else a_loads("na_", "%kk_n")
        return nxt, anx
    # wlate: the next phase's weight loads go in the last k step, after its B loads (vmcnt is in order: B waits
    # behind older weight loads would wait for DRAM); they stop holding registers through the MMA section
    late = t.wlate and KSL
    if not late:
        nxt, anx = next_loads()
    if not DBUF:
        e("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    if DECAHEAD:
        # phase kp+1 into tile (kp+1)%2 while every wave multiplies tile kp%2
        e("    %kp_d0 = index.add %kp, %c1 : index")
        e("    %kp_d = index.min %kp_d0, %kp_last : index")
        e("    %ph_d = index.rem %kp_d, %cph : index")
        e("    %ph_di = index.cast %ph_d : index to i32")
        e("    %phg_d = scalar.muli %ph_di, %cgppi : i32")
        e("    %gb_d = scalar.addi %phg_d, %gl_i : i32")
        e("    %buf_d = index.rem %kp_d0, %c2 : index")
        e("    %off_d0 = index.mul %buf_d, %wl_tb : index")
        e("    %off_d = index.cast %off_d0 : index to offset")
        e(f"    %wl_dec = buffer.view %wl[%off_d] : buffer -> view<{BM}x{G.ROWP}xf16>")
        e("    %buf_m = index.rem %kp, %c2 : index")
        e("    %off_m0 = index.mul %buf_m, %wl_tb : index")
        e("    %off_m = index.cast %off_m0 : index to offset")
        e(f"    %wl_mma = buffer.view %wl[%off_m] : buffer -> view<{BM}x{G.ROWP}xf16>")
        # decoding lanes are whole waves (tid < slots*BM): branch on the wave id so the branch is uniform.
        # A divergent branch right before the MMA loop is rejected (divergent_loop_single_entry).
        assert (slots * BM) % WS == 0
        e("    %sg_id = kernel.subgroup.id : index")
        e(f"    %cdecw = index.constant {slots * BM // WS} : index")
        e("    %dec_wave = index.cmp ult, %sg_id, %cdecw : index")
        e("    scf.if %dec_wave {")
        names = G.unpack_vals(e, cur_w, orig0)
        L.extend(l.replace("%wl_view[", "%wl_dec[") for l in compute(names, "%gb_d"))
        e("    }")
    wlv = "%wl_mma" if DECAHEAD or DBUF else "%wl_view"
    alv = "%al_mma" if DBUF else "%al_t"
    cb = ", ".join(f"%b{i} = %a{i} : {V8}" for i in range(NA))
    if KSL:
        # the k steps straight-line in one block: low CSE shares the fragment address math and the step offset becomes an immediate
        nst = ksub // 16
        acc = [f"%a{i}" for i in range(NA)]
        for st in range(nst):
            if st:
                e("    scf.schedule.fence")
            if not (t.bpre and st) and not (t.b0early and st == 0):
                e(f"    %sks{st} = index.constant {16 * st} : index")
            if t.bpre and st + 1 < nst and not (t.b0early > 1 and st == 0):
                e(f"    %sks{st + 1} = index.constant {16 * (st + 1)} : index")

            def slhs(i):
                e(f"    %slr{st}_{i} = index.add %wr_off, %c{16 * i} : index")
                e(f"    %slhs{st}_{i} = vector.fragment.load<lhs> {wlv}[%slr{st}_{i}, %sks{st}] shape [%m, %k] : view<{BM}x{G.ROWP}xf16> -> {VF}")
            if not t.lhs_stream:
                for i in range(FM):
                    slhs(i)

            def srhs(j, st=st):
                if AFRAG:
                    if not (t.b0early and st == 0) and not (t.b0early > 1 and st == 1 and j < t.bpre):
                        afrag_rhs(st, j)
                else:
                    e(f"    %stc{st}_{j} = index.add %wt_off, %c{16 * j} : index")
                    e(f"    %srhs{st}_{j} = vector.fragment.load<rhs> {alv}[%sks{st}, %stc{st}_{j}] shape [%k, %n] : view<{ksub}x{BN}xf16, %al_layout> -> {VF}")

            def smma(i, j):
                n = i * FN + j
                name = f"%r{n}" if st == nst - 1 else f"%sn{st}_{n}"
                e(f"    {name} = vector.mma %slhs{st}_{i}, %srhs{st}_{j}, {acc[n]} : {VF}, {VF}, {V8}")
            if t.bpre:
                # B fragments j < bpre of this step were issued by the previous step (step 0: now); issue the next step's
                for j in range(FN):
                    if st == 0 or j >= t.bpre:
                        srhs(j)
                if st + 1 < nst:
                    for j in range(min(t.bpre, FN)):
                        srhs(j, st + 1)
                if late and st == nst - 1:
                    nxt, anx = next_loads()
            if t.lhs_stream:
                if not t.bpre:
                    for j in range(FN):
                        srhs(j)
                    if late and st == nst - 1:
                        nxt, anx = next_loads()
                for i in range(FM):
                    slhs(i)
                    for j in range(FN):
                        smma(i, j)
                    if (i + 1) % t.lhs_stream == 0 and i + 1 < FM:
                        e("    scf.schedule.fence")
            elif RHS_OUTER:
                # rhs-outer: each activation fragment dies after its FM MMAs, so a step holds FM + ~RHS_FENCE fragments, not FM + FN
                for j in range(FN):
                    srhs(j)
                    for i in range(FM):
                        smma(i, j)
                    if RHS_FENCE and j + 1 < FN and (j + 1) % RHS_FENCE == 0:
                        e("    scf.schedule.fence")
            else:
                if not t.bpre:
                    for j in range(FN):
                        srhs(j)
                    if late and st == nst - 1:
                        nxt, anx = next_loads()
                for i in range(FM):
                    for j in range(FN):
                        smma(i, j)
            acc = [f"%sn{st}_{n}" for n in range(NA)]
        if DBUF:
            e("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        e("    scf.yield " + ", ".join(f"%r{i}" for i in range(NA))
          + "".join(f", {nm}" for nm, _ in nxt + anx) + f" : {carried_t}")
        e("  }")
    else:
        e("    " + ", ".join(f"%r{i}" for i in range(NA)) + f" = scf.for %ks = [%c0 to %cksub step %c16]({cb}) -> ({types})  {{")
        for i in range(FM):
            e(f"      %lr{i} = index.add %wr_off, %c{16 * i} : index")
            e(f"      %lhs{i} = vector.fragment.load<lhs> {wlv}[%lr{i}, %ks] shape [%m, %k] : view<{BM}x{G.ROWP}xf16> -> {VF}")

        def rhs_load(j):
            e(f"      %tc{j} = index.add %wt_off, %c{16 * j} : index")
            e(f"      %rhs{j} = vector.fragment.load<rhs> {alv}[%ks, %tc{j}] shape [%k, %n] : view<{ksub}x{BN}xf16, %al_layout> -> {VF}")
        if RHS_OUTER:
            # rhs-outer: each rhs fragment dies after its FM MMAs, so only the lhs fragments and one or two rhs are live.
            # Every accumulator still takes one MMA per k step: same values.
            for j in range(FN):
                rhs_load(j)
                for i in range(FM):
                    n = i * FN + j
                    e(f"      %n{n} = vector.mma %lhs{i}, %rhs{j}, %b{n} : {VF}, {VF}, {V8}")
                if RHS_FENCE and j + 1 < FN and (j + 1) % RHS_FENCE == 0:
                    e("      scf.schedule.fence")
        else:
            for j in range(FN):
                rhs_load(j)
            for i in range(FM):
                for j in range(FN):
                    n = i * FN + j
                    e(f"      %n{n} = vector.mma %lhs{i}, %rhs{j}, %b{n} : {VF}, {VF}, {V8}")
        e("      scf.yield " + ", ".join(f"%n{i}" for i in range(NA)) + f" : {types}")
        e("    }")
        if DBUF:
            e("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        e("    scf.yield " + ", ".join(f"%r{i}" for i in range(NA))
          + "".join(f", {nm}" for nm, _ in nxt + anx) + f" : {carried_t}")
        e("  }")
    if t.swepi:
        lds_epilogue(e, t, kr, V8, sw=True, masked=masked)
        e("  kernel.return")
        e("}")
        return "\n".join(L) + "\n"
    if sw:
        swiglu_epilogue(e, t, arow, masked)
        e("  kernel.return")
        e("}")
        return "\n".join(L) + "\n"
    if TM == 32:
        lds_epilogue(e, t, kr, V8, qg=qg, masked=masked)
        e("  kernel.return")
        e("}")
        return "\n".join(L) + "\n"
    if ff:
        _lds_epilogue_ffn(e, t, V8)
        e("  kernel.return")
        e("}")
        return "\n".join(L) + "\n"
    if t.tallepi:
        _lds_epilogue_tall(e, t, kr, V8, masked=masked, qg=qg, sr=esr)
        e("  kernel.return")
        e("}")
        return "\n".join(L) + "\n"
    e(f"  %out_layout = encoding.layout.strided [%c1, {orw()}] : encoding<layout>")
    e("  %out_t_view = buffer.view %output_na[%base] : buffer -> view<[%m_rows]x[%tokens]xf32, %out_layout>")
    if kr:
        e("  %res_t_view = buffer.view %resid_na[%base] : buffer -> view<[%m_rows]x[%tokens]xf32, %out_layout>")
    for i in range(FM):
        e(f"  %or{i} = index.add %m_origin, %c{16 * i} : index")
    for j in range(FN):
        e(f"  %ot{j} = index.add %token_base, %c{16 * j} : index")
    for i in range(FM):
        for j in range(FN):
            a = i * FN + j
            val = f"%acc{a}"
            if kr:
                e(f"  %rf{a} = vector.fragment.load<result> %res_t_view[%or{i}, %ot{j}] shape [%m, %n] : view<[%m_rows]x[%tokens]xf32, %out_layout> -> {V8}")
                e(f"  %rs{a} = vector.addf %rf{a}, %acc{a} : {V8}")
                val = f"%rs{a}"
            e(f"  vector.fragment.store<result> {val}, %out_t_view[%or{i}, %ot{j}] shape [%m, %n] : {V8}, view<[%m_rows]x[%tokens]xf32, %out_layout>")
    e("  kernel.return")
    e("}")
    return "\n".join(L) + "\n"


# swiglu through lds_epilogue: the gate loads of a 16-token column are issued this many columns before it is processed,
# so their memory latency overlaps the column in between (0: each column loads its gate just before use). Measured
# 2026-10-03: 1 is -1.2% IQ3_S / -0.7% IQ3_XXS swiglu cycles; 2 and 4 are slower (the epilogue is mostly a gate-stream
# bandwidth burst at the end of each workgroup, not load latency).
SW_GATE_AHEAD = 1


# silu's 1 / (1 + exp(-g)): v_rcp_f32 and one Newton step instead of the exact division (~10 VALU -> 4). Bit-exact for
# every f32 dn in [1, 2^126] (exhaustive on the GPU, yah-scratch/rcp/rcp_dump.loom: 0 of 1.06e9 differ) and dn = inf (the
# select). In (2^126, inf) 1/dn is denormal and can differ, but there g is in [-88.7, -87.3], so g * iv * up is below
# 1e-35 for any finite up and f16 rounds both to the same signed zero.
FAST_RCP = True


def rcp_consts(e, ind="  "):
    if FAST_RCP:
        e(f"{ind}%rcp_infb = scalar.constant 2139095040 : i32")
        e(f"{ind}%rcp_inf = scalar.bitcast %rcp_infb : i32 to f32")
        e(f"{ind}%rcp_zero = scalar.constant 0.0 : f32")


def recip1(e, out, dn, ind="  "):
    """out = 1 / dn for dn >= 1 (a sigmoid's denominator); needs %one, %negone and rcp_consts."""
    if not FAST_RCP:
        e(f"{ind}{out} = scalar.divf %one, {dn} : f32")
        return
    e(f"{ind}{out}_r0 = scalar.divf<nnan|ninf|nsz|arcp> %one, {dn} : f32")
    e(f"{ind}{out}_nd = scalar.mulf {dn}, %negone : f32")
    e(f"{ind}{out}_er = scalar.fmaf {out}_nd, {out}_r0, %one : f32")
    e(f"{ind}{out}_r1 = scalar.fmaf {out}_er, {out}_r0, {out}_r0 : f32")
    e(f"{ind}{out}_if = scalar.cmpf oeq, {dn}, %rcp_inf : f32")
    e(f"{ind}{out} = scf.select {out}_if, %rcp_zero, {out}_r1 : f32")


def lds_epilogue(e, t, kr, V8, sw=False, qg=False, masked=False):
    """Store out[t*m + r] (+ resid) for the wave's TM x TN tile, one 16-token column of fragments at a time.
    Fragments go to an LDS slab; each lane reads 16 contiguous rows of one token (two lanes per token) and writes 4 b128 stores.
    A direct fragment store writes each lane's values at an 8-byte row stride instead. Same values: bit-identical."""
    if sw and SW_GATE_AHEAD:
        return _lds_epilogue_ahead(e, t, kr, V8, sw, qg, masked)
    TM, FM, FN = t.tm, t.tm // 16, t.tn // 16
    e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    e(f"  %es_ctm = index.constant {TM + EPAD} : index")
    e("  %es_lay = encoding.layout.strided [%c1, %es_ctm] : encoding<layout>")
    e(f"  %es_wb = index.constant {(TM + EPAD) * 16 * 4} : index")
    e("  %es_off_i = index.mul %wave, %es_wb : index")
    e("  %es_off = index.cast %es_off_i : index to offset")
    e(f"  %es_view = buffer.view %al[%es_off] : buffer -> view<{TM}x16xf32, %es_lay>")
    e(f"  %es_flat = buffer.view %al[%es_off] : buffer -> view<{(TM + EPAD) * 16}xf32>")
    if sw:
        # swiglu: out f16 = f16(silu(gate) * acc), swiglu_epilogue's scalar ops per element (bit-identical), 4 rows per load/store
        e("  %gate_view = buffer.view %gate_na[%base] : buffer -> view<[%out_total]xf32>")
        e("  %out_h = buffer.view %output_na[%base] : buffer -> view<[%out_total]xf16>")
        e("  %negone = scalar.constant -1.0 : f32")
        rcp_consts(e)
        e("  %one = scalar.constant 1.0 : f32")
    elif qg:
        # row r = head*512 + half*256 + d goes to (q|gate)[t][head*256 + d]; a wave's TM=32 rows sit inside one half
        e("  %qg_tot = index.div %out_total, %c2 : index")
        e("  %qg_rows = index.div %m_rows, %c2 : index")
        e("  %q_flat = buffer.view %output_na[%base] : buffer -> view<[%qg_tot]xf32>")
        e("  %g_flat = buffer.view %gate_out_na[%base] : buffer -> view<[%qg_tot]xf32>")
        e("  %qg_last4 = index.sub %qg_tot, %c4 : index")
        e("  %qg_c512 = index.constant 512 : index")
        e("  %qg_c256 = index.constant 256 : index")
    else:
        e(f"  %out_flat = buffer.view %output_na[%base] : buffer -> view<[%out_total]x{'f16' if OUT16 and not kr else 'f32'}>")
    if kr:
        e("  %res_flat = buffer.view %resid_na[%base] : buffer -> view<[%out_total]xf32>")
    if masked:
        e("  %es_tok_last = index.sub %tokens, %c1 : index")
    e("  %es_lane = index.rem %tid, %c32 : index")
    e("  %es_t = index.div %es_lane, %c2 : index")
    e("  %es_h0 = index.rem %es_lane, %c2 : index")
    e("  %es_h = index.mul %es_h0, %c16 : index")
    e(f"  %es_tt = index.mul %es_t, %es_ctm : index")
    e("  %es_rd = index.add %es_tt, %es_h : index")
    e("  %es_row = index.add %m_origin, %es_h : index")
    if qg:
        e("  %qg_head = index.div %es_row, %qg_c512 : index")
        e("  %qg_w = index.rem %es_row, %qg_c512 : index")
        e("  %qg_half = index.div %qg_w, %qg_c256 : index")
        e("  %qg_d = index.rem %qg_w, %qg_c256 : index")
        e("  %qg_hb = index.mul %qg_head, %qg_c256 : index")
        e("  %qg_col = index.add %qg_hb, %qg_d : index")
        e("  %qg_isq = index.cmp eq, %qg_half, %c0 : index")
    for j in range(FN):
        for i in range(FM):
            e(f"  %es_r{i}_{j} = index.constant {16 * i} : index")
            e(f"  vector.fragment.store<result> %acc{i * FN + j}, %es_view[%es_r{i}_{j}, %c0] shape [%m, %n] : {V8}, view<{TM}x16xf32, %es_lay>")
        e(f"  %es_tc{j} = index.constant {16 * j} : index")
        e(f"  %es_tk{j}0 = index.add %token_base, %es_tc{j} : index")
        e(f"  %es_tk{j} = index.add %es_tk{j}0, %es_t : index")
        es_tka = f"%es_tk{j}"
        if masked:
            # the guard below skips tokens past the last; the clamp lets the compiler prove the addresses in bounds
            e(f"  %es_tkc{j} = index.min %es_tk{j}, %es_tok_last : index")
            es_tka = f"%es_tkc{j}"
        e(f"  %es_tm{j} = index.mul {es_tka}, {orw()} : index")
        e(f"  %es_ob{j} = index.add %es_tm{j}, %es_row : index")
        if qg:
            e(f"  %qg_tm{j} = index.mul {es_tka}, %qg_rows : index")
            e(f"  %qg_ob{j} = index.add %qg_tm{j}, %qg_col : index")
        if masked:
            # this lane's token past the last valid one: no residual / gate load, no store
            e(f"  %es_ok{j} = index.cmp ult, %es_tk{j}, %tokens : index")
            e(f"  scf.if %es_ok{j} {{")
        for q in range(4):
            e(f"  %es_q{j}_{q}c = index.constant {4 * q} : index")
            e(f"  %es_ri{j}_{q} = index.add %es_rd, %es_q{j}_{q}c : index")
            e(f"  %es_v{j}_{q} = vector.load %es_flat[%es_ri{j}_{q}] : view<{(TM + EPAD) * 16}xf32> -> vector<4xf32>")
            e(f"  %es_oi{j}_{q} = index.add %es_ob{j}, %es_q{j}_{q}c : index")
            val = f"%es_v{j}_{q}"
            if kr:
                e(f"  %es_rf{j}_{q} = vector.load %res_flat[%es_oi{j}_{q}] : view<[%out_total]xf32> -> vector<4xf32>")
                e(f"  %es_rs{j}_{q} = vector.addf %es_rf{j}_{q}, %es_v{j}_{q} : vector<4xf32>")
                val = f"%es_rs{j}_{q}"
            if sw:
                e(f"  %es_g{j}_{q} = vector.load %gate_view[%es_oi{j}_{q}] : view<[%out_total]xf32> -> vector<4xf32>")
                hs = []
                for x in range(4):
                    y = f"{j}_{q}_{x}"
                    e(f"  %g_{y} = vector.extract %es_g{j}_{q}[{x}] : vector<4xf32> -> f32")
                    e(f"  %v_{y} = vector.extract %es_v{j}_{q}[{x}] : vector<4xf32> -> f32")
                    e(f"  %ng_{y} = scalar.mulf %g_{y}, %negone : f32")
                    e(f"  %ex_{y} = scalar.expf<afn> %ng_{y} : f32")
                    e(f"  %dn_{y} = scalar.addf %one, %ex_{y} : f32")
                    recip1(e, f"%iv_{y}", f"%dn_{y}")
                    e(f"  %sg_{y} = scalar.mulf %g_{y}, %iv_{y} : f32")
                    e(f"  %ac_{y} = scalar.mulf %sg_{y}, %v_{y} : f32")
                    e(f"  %h_{y} = scalar.fptrunc %ac_{y} : f32 to f16")
                    hs.append(f"%h_{y}")
                e(f"  %hv{j}_{q} = vector.from_elements {', '.join(hs)} : vector<4xf16>")
                e(f"  vector.store %hv{j}_{q}, %out_h[%es_oi{j}_{q}] : vector<4xf16>, view<[%out_total]xf16>")
            elif qg:
                # always in range; the clamp states it for the bound proof
                e(f"  %qg_oir{j}_{q} = index.add %qg_ob{j}, %es_q{j}_{q}c : index")
                e(f"  %qg_oi{j}_{q} = index.min %qg_oir{j}_{q}, %qg_last4 : index")
                e(f"  scf.if %qg_isq {{")
                e(f"    vector.store {val}, %q_flat[%qg_oi{j}_{q}] : vector<4xf32>, view<[%qg_tot]xf32>")
                e("  } else {")
                e(f"    vector.store {val}, %g_flat[%qg_oi{j}_{q}] : vector<4xf32>, view<[%qg_tot]xf32>")
                e("  }")
            else:
                if OUT16 and not kr:   # f16 output (emit_prefill_pp.O16_MT: consumers read f16)
                    e(f"  %es_h{j}_{q} = vector.fptrunc {val} : vector<4xf32> to vector<4xf16>")
                    e(f"  vector.store %es_h{j}_{q}, %out_flat[%es_oi{j}_{q}] : vector<4xf16>, view<[%out_total]xf16>")
                else:
                    e(f"  vector.store {val}, %out_flat[%es_oi{j}_{q}] : vector<4xf32>, view<[%out_total]xf32>")
        if masked:
            e("  }")




def _lds_epilogue_tall(e, t, kr, V8, masked=False, sr=16, qg=False):
    """lds_epilogue for waves taller than 32 rows (afrag: 128): sr-row slabs, so 16 waves' slabs fit in the weight tile,
    which is free after the K loop. Per 16-token column and slab, each lane reads sr / 2 contiguous rows of one token (two
    lanes per token) and writes them with b128 stores. kstore / kres / kqg (a slab sits inside one 256-row q or gate
    half: row r = head * 512 + half * 256 + d goes to (q | gate)[t][head * 256 + d]). Same values as lds_epilogue."""
    TM, FM, FN = t.tm, t.tm // 16, t.tn // 16
    assert TM % sr == 0 and sr % 16 == 0
    e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    e(f"  %et_cs = index.constant {sr + EPAD} : index")
    e("  %et_lay = encoding.layout.strided [%c1, %et_cs] : encoding<layout>")
    e(f"  %et_wb = index.constant {(sr + EPAD) * 16 * 4} : index")
    e("  %et_off_i = index.mul %wave, %et_wb : index")
    e("  %et_off = index.cast %et_off_i : index to offset")
    e(f"  %et_view = buffer.view %wl[%et_off] : buffer -> view<{sr}x16xf32, %et_lay>")
    e(f"  %et_flat = buffer.view %wl[%et_off] : buffer -> view<{(sr + EPAD) * 16}xf32>")
    if qg:
        e("  %qg_tot = index.div %out_total, %c2 : index")
        e("  %qg_rows = index.div %m_rows, %c2 : index")
        e("  %q_flat = buffer.view %output_na[%base] : buffer -> view<[%qg_tot]xf32>")
        e("  %g_flat = buffer.view %gate_out_na[%base] : buffer -> view<[%qg_tot]xf32>")
        e("  %qg_last4 = index.sub %qg_tot, %c4 : index")
        e("  %qg_c512 = index.constant 512 : index")
        e("  %qg_c256 = index.constant 256 : index")
    else:
        e(f"  %out_flat = buffer.view %output_na[%base] : buffer -> view<[%out_total]x{'f16' if OUT16 and not kr else 'f32'}>")
    if kr:
        e("  %res_flat = buffer.view %resid_na[%base] : buffer -> view<[%out_total]xf32>")
    if masked:
        e("  %et_tok_last = index.sub %tokens, %c1 : index")
    e("  %et_lane = index.rem %tid, %c32 : index")
    co = t.ecoal and not qg and not masked
    assert co or sr == 16
    lpt = sr // 4            # lanes per token (16 rows: 4, 64 B; 32 rows: 8, 128 B)
    tpa = 32 // lpt          # tokens per access
    if co:
        # lane lpt * t8 + r: token t8 + tpa q, rows 4 r .. 4 r + 3 of the slab (one contiguous row segment per token per access)
        e(f"  %et_t = index.div %et_lane, %c{lpt} : index")
        e(f"  %et_h0 = index.rem %et_lane, %c{lpt} : index")
        e("  %et_h = index.mul %et_h0, %c4 : index")
    else:
        e("  %et_t = index.div %et_lane, %c2 : index")
        e("  %et_h0 = index.rem %et_lane, %c2 : index")
        e(f"  %et_h = index.mul %et_h0, %c{sr // 2} : index")

    def qoff(y, q):
        # per-access offsets: LDS (slab) and global (output / residual); old mapping: rows + 4 q in both
        if co:
            e(f"  %et_ql{y} = index.constant {tpa * q * (sr + EPAD)} : index")
            e(f"  %et_qk{y} = index.constant {tpa * q} : index")
            e(f"  %et_qg{y} = index.mul %et_qk{y}, {orw()} : index")
            return f"%et_ql{y}", f"%et_qg{y}"
        e(f"  %et_q{y}c = index.constant {4 * q} : index")
        return f"%et_q{y}c", f"%et_q{y}c"
    qo = {}
    e("  %et_tt = index.mul %et_t, %et_cs : index")
    e("  %et_rd = index.add %et_tt, %et_h : index")
    e("  %et_rowb = index.add %m_origin, %et_h : index")
    pre = t.respre if kr and not masked and not qg else 0
    groups = [(g, j) for g in range(TM // sr) for j in range(FN)]

    def res_addr(g, j):
        q0 = f"{g}_{j}"
        e(f"  %et_tc{q0} = index.constant {16 * j} : index")
        e(f"  %et_tk{q0}0 = index.add %token_base, %et_tc{q0} : index")
        e(f"  %et_tk{q0} = index.add %et_tk{q0}0, %et_t : index")
        e(f"  %et_tm{q0} = index.mul %et_tk{q0}, {orw()} : index")
        e(f"  %et_ob{q0} = index.add %et_tm{q0}, %et_row{g} : index")
        for q in range(sr // 8):
            y = f"{q0}_{q}"
            qo[y] = qoff(y, q)
            e(f"  %et_oi{y} = index.add %et_ob{q0}, {qo[y][1]} : index")
            e(f"  %et_rf{y} = vector.load %res_flat[%et_oi{y}] : view<[%out_total]xf32> -> vector<4xf32>")
    if pre:
        # residual loads run `pre` slab groups ahead of their adds: one DRAM round trip per wave, not one per group
        for g in range(TM // sr):
            e(f"  %et_gr{g} = index.constant {g * sr} : index")
            e(f"  %et_row{g} = index.add %et_rowb, %et_gr{g} : index")
        for n in range(min(pre, len(groups))):
            res_addr(*groups[n])
        for n, (g, j) in enumerate(groups):
            q0 = f"{g}_{j}"
            e("  scf.schedule.fence")
            if n + pre < len(groups):
                res_addr(*groups[n + pre])
            for i in range(g * sr // 16, (g + 1) * sr // 16):
                e(f"  %et_r{i}_{j} = index.constant {16 * i - g * sr} : index")
                e(f"  vector.fragment.store<result> %acc{i * FN + j}, %et_view[%et_r{i}_{j}, %c0] shape [%m, %n] : {V8}, view<{sr}x16xf32, %et_lay>")
            for q in range(sr // 8):
                y = f"{q0}_{q}"
                e(f"  %et_ri{y} = index.add %et_rd, {qo[y][0]} : index")
                e(f"  %et_v{y} = vector.load %et_flat[%et_ri{y}] : view<{(sr + EPAD) * 16}xf32> -> vector<4xf32>")
                e(f"  %et_rs{y} = vector.addf %et_rf{y}, %et_v{y} : vector<4xf32>")
                e(f"  vector.store %et_rs{y}, %out_flat[%et_oi{y}] : vector<4xf32>, view<[%out_total]xf32>")
    for g in range(TM // sr) if not pre else []:
        e(f"  %et_gr{g} = index.constant {g * sr} : index")
        e(f"  %et_row{g} = index.add %et_rowb, %et_gr{g} : index")
        if qg:
            e(f"  %qg_head{g} = index.div %et_row{g}, %qg_c512 : index")
            e(f"  %qg_w{g} = index.rem %et_row{g}, %qg_c512 : index")
            e(f"  %qg_half{g} = index.div %qg_w{g}, %qg_c256 : index")
            e(f"  %qg_d{g} = index.rem %qg_w{g}, %qg_c256 : index")
            e(f"  %qg_hb{g} = index.mul %qg_head{g}, %qg_c256 : index")
            e(f"  %qg_col{g} = index.add %qg_hb{g}, %qg_d{g} : index")
            e(f"  %qg_isq{g} = index.cmp eq, %qg_half{g}, %c0 : index")
        for j in range(FN):
            q0 = f"{g}_{j}"
            if kr:
                # one slab at a time: else the residual loads are hoisted over every slab (256 VGPRs beside the accumulators)
                e("  scf.schedule.fence")
            for i in range(g * sr // 16, (g + 1) * sr // 16):
                e(f"  %et_r{i}_{j} = index.constant {16 * i - g * sr} : index")
                e(f"  vector.fragment.store<result> %acc{i * FN + j}, %et_view[%et_r{i}_{j}, %c0] shape [%m, %n] : {V8}, view<{sr}x16xf32, %et_lay>")
            e(f"  %et_tc{q0} = index.constant {16 * j} : index")
            e(f"  %et_tk{q0}0 = index.add %token_base, %et_tc{q0} : index")
            e(f"  %et_tk{q0} = index.add %et_tk{q0}0, %et_t : index")
            tka = f"%et_tk{q0}"
            if masked:
                e(f"  %et_tkc{q0} = index.min %et_tk{q0}, %et_tok_last : index")
                tka = f"%et_tkc{q0}"
            if qg:
                e(f"  %et_tm{q0} = index.mul {tka}, %qg_rows : index")
                e(f"  %et_ob{q0} = index.add %et_tm{q0}, %qg_col{g} : index")
            else:
                e(f"  %et_tm{q0} = index.mul {tka}, {orw()} : index")
                e(f"  %et_ob{q0} = index.add %et_tm{q0}, %et_row{g} : index")
            if masked:
                e(f"  %et_ok{q0} = index.cmp ult, %et_tk{q0}, %tokens : index")
                e(f"  scf.if %et_ok{q0} {{")
            for q in range(sr // 8):
                y = f"{q0}_{q}"
                ql, qg_ = qoff(y, q)
                e(f"  %et_ri{y} = index.add %et_rd, {ql} : index")
                e(f"  %et_v{y} = vector.load %et_flat[%et_ri{y}] : view<{(sr + EPAD) * 16}xf32> -> vector<4xf32>")
                e(f"  %et_oi{y} = index.add %et_ob{q0}, {qg_} : index")
                val = f"%et_v{y}"
                if kr:
                    e(f"  %et_rf{y} = vector.load %res_flat[%et_oi{y}] : view<[%out_total]xf32> -> vector<4xf32>")
                    e(f"  %et_rs{y} = vector.addf %et_rf{y}, %et_v{y} : vector<4xf32>")
                    val = f"%et_rs{y}"
                if qg:
                    # always in range; the clamp states it for the bound proof
                    e(f"  %qg_oi{y} = index.min %et_oi{y}, %qg_last4 : index")
                    e(f"  scf.if %qg_isq{g} {{")
                    e(f"    vector.store {val}, %q_flat[%qg_oi{y}] : vector<4xf32>, view<[%qg_tot]xf32>")
                    e("  } else {")
                    e(f"    vector.store {val}, %g_flat[%qg_oi{y}] : vector<4xf32>, view<[%qg_tot]xf32>")
                    e("  }")
                else:
                    if OUT16 and not kr:   # f16 output (emit_prefill_pp.O16_MT: consumers read f16)
                        e(f"  %et_h16{y} = vector.fptrunc {val} : vector<4xf32> to vector<4xf16>")
                        e(f"  vector.store %et_h16{y}, %out_flat[%et_oi{y}] : vector<4xf16>, view<[%out_total]xf16>")
                    else:
                        e(f"  vector.store {val}, %out_flat[%et_oi{y}] : vector<4xf32>, view<[%out_total]xf32>")
            if masked:
                e("  }")


def _lds_epilogue_ffn(e, t, V8, sr=16):
    """ffn: out[t][r] = f16(silu(gate) * up) for the workgroup's 64 rows; gate in accumulator rows 0-63 (fragments 0-3),
    up in 64-127 (4-7), the same lane / element in both. The swiglu GEMM's scalar ops per element (bit-identical: the
    gate is the f32 value the gate GEMM stored), then through sr-row slabs in the weight tile as _lds_epilogue_tall,
    4-row f16 stores, row-major or fragment-major (t.tout). (Staging gate and up through LDS slabs first and doing the
    math per slab was slower: IQ4_XS 41.31 vs 40.70 M.)"""
    FN = t.tn // 16
    FG = t.tm // 32                  # gate fragments
    assert sr == 16
    e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    e("  %negone = scalar.constant -1.0 : f32")
    rcp_consts(e)
    e("  %one = scalar.constant 1.0 : f32")
    for i in range(FG):
        for j in range(FN):
            g, u = i * FN + j, (i + FG) * FN + j
            xs = []
            for x in range(8):
                y = f"{i}_{j}_{x}"
                e(f"  %fg_{y} = vector.extract %acc{g}[{x}] : {V8} -> f32")
                e(f"  %fu_{y} = vector.extract %acc{u}[{x}] : {V8} -> f32")
                e(f"  %fng_{y} = scalar.mulf %fg_{y}, %negone : f32")
                e(f"  %fex_{y} = scalar.expf<afn> %fng_{y} : f32")
                e(f"  %fdn_{y} = scalar.addf %one, %fex_{y} : f32")
                recip1(e, f"%fiv_{y}", f"%fdn_{y}")
                e(f"  %fsg_{y} = scalar.mulf %fg_{y}, %fiv_{y} : f32")
                e(f"  %fac_{y} = scalar.mulf %fsg_{y}, %fu_{y} : f32")
                xs.append(f"%fac_{y}")
            e(f"  %fo{i}_{j} = vector.from_elements {', '.join(xs)} : {V8}")
    e(f"  %et_cs = index.constant {sr + EPAD} : index")
    e("  %et_lay = encoding.layout.strided [%c1, %et_cs] : encoding<layout>")
    e(f"  %et_wb = index.constant {(sr + EPAD) * 16 * 4} : index")
    e("  %et_off_i = index.mul %wave, %et_wb : index")
    e("  %et_off = index.cast %et_off_i : index to offset")
    e(f"  %et_view = buffer.view %wl[%et_off] : buffer -> view<{sr}x16xf32, %et_lay>")
    e(f"  %et_flat = buffer.view %wl[%et_off] : buffer -> view<{(sr + EPAD) * 16}xf32>")
    e("  %out_h = buffer.view %output_na[%base] : buffer -> view<[%out_total]xf16>")
    e("  %et_last4 = index.sub %out_total, %c4 : index")
    e("  %et_lane = index.rem %tid, %c32 : index")
    e("  %et_t = index.div %et_lane, %c2 : index")
    e("  %et_h0 = index.rem %et_lane, %c2 : index")
    e(f"  %et_h = index.mul %et_h0, %c{sr // 2} : index")
    e("  %et_tt = index.mul %et_t, %et_cs : index")
    e("  %et_rd = index.add %et_tt, %et_h : index")
    e("  %et_rowb = index.add %m_origin, %et_h : index")
    if t.tout:
        e("  %et_mt = index.div %m_rows, %c16 : index")
    for g in range(FG):
        e(f"  %et_gr{g} = index.constant {g * sr} : index")
        e(f"  %et_row{g} = index.add %et_rowb, %et_gr{g} : index")
        for j in range(FN):
            q0 = f"{g}_{j}"
            e("  scf.schedule.fence")
            e(f"  vector.fragment.store<result> %fo{g}_{j}, %et_view[%c0, %c0] shape [%m, %n] : {V8}, view<{sr}x16xf32, %et_lay>")
            e(f"  %et_tc{q0} = index.constant {16 * j} : index")
            e(f"  %et_tk{q0}0 = index.add %token_base, %et_tc{q0} : index")
            e(f"  %et_tk{q0} = index.add %et_tk{q0}0, %et_t : index")
            for q in range(sr // 8):
                y = f"{q0}_{q}"
                e(f"  %et_q{y}c = index.constant {4 * q} : index")
                e(f"  %et_ri{y} = index.add %et_rd, %et_q{y}c : index")
                e(f"  %et_v{y} = vector.load %et_flat[%et_ri{y}] : view<{(sr + EPAD) * 16}xf32> -> vector<4xf32>")
                e(f"  %et_h{y} = vector.fptrunc %et_v{y} : vector<4xf32> to vector<4xf16>")
                e(f"  %et_r{y} = index.add %et_row{g}, %et_q{y}c : index")
                if t.tout:
                    e(f"  %et_a{y} = index.div %et_tk{q0}, %c16 : index")
                    e(f"  %et_b{y} = index.rem %et_tk{q0}, %c16 : index")
                    e(f"  %et_c{y} = index.div %et_r{y}, %c16 : index")
                    e(f"  %et_d{y} = index.rem %et_r{y}, %c16 : index")
                    e(f"  %et_e{y} = index.mul %et_a{y}, %et_mt : index")
                    e(f"  %et_f{y} = index.add %et_e{y}, %et_c{y} : index")
                    e(f"  %et_g{y} = index.mul %et_f{y}, %c256 : index")
                    e(f"  %et_i{y} = index.mul %et_b{y}, %c16 : index")
                    e(f"  %et_j{y} = index.add %et_g{y}, %et_i{y} : index")
                    e(f"  %et_o{y}0 = index.add %et_j{y}, %et_d{y} : index")
                else:
                    e(f"  %et_m{y} = index.mul %et_tk{q0}, {orw()} : index")
                    e(f"  %et_o{y}0 = index.add %et_m{y}, %et_r{y} : index")
                e(f"  %et_o{y} = index.min %et_o{y}0, %et_last4 : index")   # always in range; for the bound proof
                e(f"  vector.store %et_h{y}, %out_h[%et_o{y}] : vector<4xf16>, view<[%out_total]xf16>")


def _lds_epilogue_ahead(e, t, kr, V8, sw=False, qg=False, masked=False):
    """lds_epilogue with the swiglu gate loads issued SW_GATE_AHEAD columns before use (same values).
    Store out[t*m + r] (+ resid) for the wave's TM x TN tile, one 16-token column of fragments at a time.
    Fragments go to an LDS slab; each lane reads 16 contiguous rows of one token (two lanes per token) and writes 4 b128 stores.
    A direct fragment store writes each lane's values at an 8-byte row stride instead. Same values: bit-identical."""
    TM, FM, FN = t.tm, t.tm // 16, t.tn // 16
    e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    e(f"  %es_ctm = index.constant {TM + EPAD} : index")
    e("  %es_lay = encoding.layout.strided [%c1, %es_ctm] : encoding<layout>")
    e(f"  %es_wb = index.constant {(TM + EPAD) * 16 * 4} : index")
    e("  %es_off_i = index.mul %wave, %es_wb : index")
    e("  %es_off = index.cast %es_off_i : index to offset")
    e(f"  %es_view = buffer.view %al[%es_off] : buffer -> view<{TM}x16xf32, %es_lay>")
    e(f"  %es_flat = buffer.view %al[%es_off] : buffer -> view<{(TM + EPAD) * 16}xf32>")
    if sw:
        # swiglu: out f16 = f16(silu(gate) * acc), swiglu_epilogue's scalar ops per element (bit-identical), 4 rows per load/store
        e("  %gate_view = buffer.view %gate_na[%base] : buffer -> view<[%out_total]xf32>")
        e("  %out_h = buffer.view %output_na[%base] : buffer -> view<[%out_total]xf16>")
        e("  %negone = scalar.constant -1.0 : f32")
        rcp_consts(e)
        e("  %one = scalar.constant 1.0 : f32")
    elif qg:
        # row r = head*512 + half*256 + d goes to (q|gate)[t][head*256 + d]; a wave's TM=32 rows sit inside one half
        e("  %qg_tot = index.div %out_total, %c2 : index")
        e("  %qg_rows = index.div %m_rows, %c2 : index")
        e("  %q_flat = buffer.view %output_na[%base] : buffer -> view<[%qg_tot]xf32>")
        e("  %g_flat = buffer.view %gate_out_na[%base] : buffer -> view<[%qg_tot]xf32>")
        e("  %qg_last4 = index.sub %qg_tot, %c4 : index")
        e("  %qg_c512 = index.constant 512 : index")
        e("  %qg_c256 = index.constant 256 : index")
    else:
        e(f"  %out_flat = buffer.view %output_na[%base] : buffer -> view<[%out_total]x{'f16' if OUT16 and not kr else 'f32'}>")
    if kr:
        e("  %res_flat = buffer.view %resid_na[%base] : buffer -> view<[%out_total]xf32>")
    if masked:
        e("  %es_tok_last = index.sub %tokens, %c1 : index")
    e("  %es_lane = index.rem %tid, %c32 : index")
    e("  %es_t = index.div %es_lane, %c2 : index")
    e("  %es_h0 = index.rem %es_lane, %c2 : index")
    e("  %es_h = index.mul %es_h0, %c16 : index")
    e(f"  %es_tt = index.mul %es_t, %es_ctm : index")
    e("  %es_rd = index.add %es_tt, %es_h : index")
    e("  %es_row = index.add %m_origin, %es_h : index")
    if qg:
        e("  %qg_head = index.div %es_row, %qg_c512 : index")
        e("  %qg_w = index.rem %es_row, %qg_c512 : index")
        e("  %qg_half = index.div %qg_w, %qg_c256 : index")
        e("  %qg_d = index.rem %qg_w, %qg_c256 : index")
        e("  %qg_hb = index.mul %qg_head, %qg_c256 : index")
        e("  %qg_col = index.add %qg_hb, %qg_d : index")
        e("  %qg_isq = index.cmp eq, %qg_half, %c0 : index")
    tka = {}

    def col_addr(j):
        """Column j's token and output offsets (es_tk{j}, es_ob{j}, es_oi{j}_q)."""
        e(f"  %es_tc{j} = index.constant {16 * j} : index")
        e(f"  %es_tk{j}0 = index.add %token_base, %es_tc{j} : index")
        e(f"  %es_tk{j} = index.add %es_tk{j}0, %es_t : index")
        es_tka = f"%es_tk{j}"
        if masked:
            # the guard skips tokens past the last; the clamp lets the compiler prove the addresses in bounds
            e(f"  %es_tkc{j} = index.min %es_tk{j}, %es_tok_last : index")
            es_tka = f"%es_tkc{j}"
        tka[j] = es_tka
        e(f"  %es_tm{j} = index.mul {es_tka}, {orw()} : index")
        e(f"  %es_ob{j} = index.add %es_tm{j}, %es_row : index")
        for q in range(4):
            e(f"  %es_q{j}_{q}c = index.constant {4 * q} : index")
            e(f"  %es_oi{j}_{q} = index.add %es_ob{j}, %es_q{j}_{q}c : index")
        if sw:
            # gate loads at the (clamped, in-bounds) address; issued SW_GATE_AHEAD columns before their use
            for q in range(4):
                e(f"  %es_g{j}_{q} = vector.load %gate_view[%es_oi{j}_{q}] : view<[%out_total]xf32> -> vector<4xf32>")
    ahead = SW_GATE_AHEAD if sw else 0
    for j in range(min(ahead, FN)):
        col_addr(j)
    for j in range(FN):
        if ahead and j + ahead < FN:
            col_addr(j + ahead)
        for i in range(FM):
            e(f"  %es_r{i}_{j} = index.constant {16 * i} : index")
            e(f"  vector.fragment.store<result> %acc{i * FN + j}, %es_view[%es_r{i}_{j}, %c0] shape [%m, %n] : {V8}, view<{TM}x16xf32, %es_lay>")
        if not ahead:
            col_addr(j)
        if qg:
            e(f"  %qg_tm{j} = index.mul {tka[j]}, %qg_rows : index")
            e(f"  %qg_ob{j} = index.add %qg_tm{j}, %qg_col : index")
        if masked:
            # this lane's token past the last valid one: no residual / gate load, no store
            e(f"  %es_ok{j} = index.cmp ult, %es_tk{j}, %tokens : index")
            e(f"  scf.if %es_ok{j} {{")
        for q in range(4):
            e(f"  %es_ri{j}_{q} = index.add %es_rd, %es_q{j}_{q}c : index")
            e(f"  %es_v{j}_{q} = vector.load %es_flat[%es_ri{j}_{q}] : view<{(TM + EPAD) * 16}xf32> -> vector<4xf32>")
            val = f"%es_v{j}_{q}"
            if kr:
                e(f"  %es_rf{j}_{q} = vector.load %res_flat[%es_oi{j}_{q}] : view<[%out_total]xf32> -> vector<4xf32>")
                e(f"  %es_rs{j}_{q} = vector.addf %es_rf{j}_{q}, %es_v{j}_{q} : vector<4xf32>")
                val = f"%es_rs{j}_{q}"
            if sw:
                hs = []
                for x in range(4):
                    y = f"{j}_{q}_{x}"
                    e(f"  %g_{y} = vector.extract %es_g{j}_{q}[{x}] : vector<4xf32> -> f32")
                    e(f"  %v_{y} = vector.extract %es_v{j}_{q}[{x}] : vector<4xf32> -> f32")
                    e(f"  %ng_{y} = scalar.mulf %g_{y}, %negone : f32")
                    e(f"  %ex_{y} = scalar.expf<afn> %ng_{y} : f32")
                    e(f"  %dn_{y} = scalar.addf %one, %ex_{y} : f32")
                    recip1(e, f"%iv_{y}", f"%dn_{y}")
                    e(f"  %sg_{y} = scalar.mulf %g_{y}, %iv_{y} : f32")
                    e(f"  %ac_{y} = scalar.mulf %sg_{y}, %v_{y} : f32")
                    e(f"  %h_{y} = scalar.fptrunc %ac_{y} : f32 to f16")
                    hs.append(f"%h_{y}")
                e(f"  %hv{j}_{q} = vector.from_elements {', '.join(hs)} : vector<4xf16>")
                e(f"  vector.store %hv{j}_{q}, %out_h[%es_oi{j}_{q}] : vector<4xf16>, view<[%out_total]xf16>")
            elif qg:
                # always in range; the clamp states it for the bound proof
                e(f"  %qg_oir{j}_{q} = index.add %qg_ob{j}, %es_q{j}_{q}c : index")
                e(f"  %qg_oi{j}_{q} = index.min %qg_oir{j}_{q}, %qg_last4 : index")
                e(f"  scf.if %qg_isq {{")
                e(f"    vector.store {val}, %q_flat[%qg_oi{j}_{q}] : vector<4xf32>, view<[%qg_tot]xf32>")
                e("  } else {")
                e(f"    vector.store {val}, %g_flat[%qg_oi{j}_{q}] : vector<4xf32>, view<[%qg_tot]xf32>")
                e("  }")
            else:
                if OUT16 and not kr:   # f16 output (emit_prefill_pp.O16_MT: consumers read f16)
                    e(f"  %es_h{j}_{q} = vector.fptrunc {val} : vector<4xf32> to vector<4xf16>")
                    e(f"  vector.store %es_h{j}_{q}, %out_flat[%es_oi{j}_{q}] : vector<4xf16>, view<[%out_total]xf16>")
                else:
                    e(f"  vector.store {val}, %out_flat[%es_oi{j}_{q}] : vector<4xf32>, view<[%out_total]xf32>")
        if masked:
            e("  }")


def swiglu_slab(t, arow):
    """Tokens per swiglu_epilogue slab: the larger of 32, 16 whose slabs fit in the activation tile, else 16."""
    return next((x for x in (32, 16) if x <= t.tn and t.nwave * 16 * x * 4 <= t.bn * arow * 2), 16)


def swiglu_epilogue(e, t, arow, masked=False):
    """out[t*m + r] = f16(silu(gate[t*m + r]) * acc[r][t]), the .loom kernel's scalar ops in order (bit-identical).
    As in gen_gemm_decode, each 16-row x ES-token slab goes through a per-wave f32 LDS tile; a loop walks it lane-contiguous.
    ES is the larger of 32, 16 for which all waves' slabs fit in the activation tile's LDS."""
    BN, TN, NWAVE, FM, FN = t.bn, t.tn, t.nwave, t.tm // 16, t.tn // 16
    ES = swiglu_slab(t, arow)
    assert TN % ES == 0
    V8 = "vector<8xf32>"
    e("  %ep_lay = encoding.layout.strided [%c1, %c16] : encoding<layout>")
    e(f"  %ep_wbytes = index.constant {16 * ES * 4} : index")
    e("  %ep_off_i = index.mul %wave, %ep_wbytes : index")
    e("  %ep_off = index.cast %ep_off_i : index to offset")
    e(f"  %ep_view = buffer.view %al[%ep_off] : buffer -> view<16x{ES}xf32, %ep_lay>")
    e(f"  %ep_flat = buffer.view %al[%ep_off] : buffer -> view<{16 * ES}xf32>")
    e("  %gate_view = buffer.view %gate_na[%base] : buffer -> view<[%out_total]xf32>")
    e("  %out_h = buffer.view %output_na[%base] : buffer -> view<[%out_total]xf16>")
    e("  %negone = scalar.constant -1.0 : f32")
    rcp_consts(e)
    e("  %one = scalar.constant 1.0 : f32")
    e("  %out_last = index.sub %out_total, %c1 : index")
    e("  %lane = index.rem %tid, %c32 : index")
    e(f"  %ep_n = index.constant {16 * ES // 32} : index")
    e(f"  %ep_last = index.constant {16 * ES - 1} : index")
    if t.tout:
        e("  %ep_mt = index.div %m_rows, %c16 : index")
    for i in range(FM):
        e(f"  %sr{i} = index.add %m_origin, %c{16 * i} : index")
        for h in range(TN // ES):
            q = f"{i}_{h}"
            e(f"  %st{q}c = index.constant {h * ES} : index")
            e(f"  %st{q} = index.add %token_base, %st{q}c : index")
            e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
            for jj in range(ES // 16):
                j = h * ES // 16 + jj
                e(f"  %sc{q}_{jj} = index.constant {16 * jj} : index")
                e(f"  vector.fragment.store<result> %acc{i * FN + j}, %ep_view[%c0, %sc{q}_{jj}] shape [%m, %n] : {V8}, view<16x{ES}xf32, %ep_lay>")
            e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
            e(f"  %eps{q} = scf.for %ee{q} = [%c0 to %ep_n step %c1](%em{q} = %c0 : index) -> (index) {{")
            e(f"    %e32_{q} = index.mul %ee{q}, %c32 : index")
            e(f"    %ef0_{q} = index.add %e32_{q}, %lane : index")
            e(f"    %ef_{q} = index.min %ef0_{q}, %ep_last : index")
            e(f"    %er_{q} = index.rem %ef_{q}, %c16 : index")
            e(f"    %et_{q} = index.div %ef_{q}, %c16 : index")
            e(f"    %v_{q} = view.load %ep_flat[%ef_{q}] : view<{16 * ES}xf32> -> f32")
            e(f"    %grow_{q} = index.add %sr{i}, %er_{q} : index")
            e(f"    %gtok_{q} = index.add %st{q}, %et_{q} : index")
            e(f"    %gto_{q} = index.mul %gtok_{q}, %m_rows : index")
            e(f"    %gix0_{q} = index.add %gto_{q}, %grow_{q} : index")
            if t.tout:
                # the output fragment-major (the gate stays row-major): tile (token / 16, row / 16), row fastest inside
                e(f"    %gtt_{q} = index.div %gtok_{q}, %c16 : index")
                e(f"    %gtr_{q} = index.rem %gtok_{q}, %c16 : index")
                e(f"    %grt_{q} = index.div %grow_{q}, %c16 : index")
                e(f"    %grr_{q} = index.rem %grow_{q}, %c16 : index")
                e(f"    %gtm_{q} = index.mul %gtt_{q}, %ep_mt : index")
                e(f"    %gti_{q} = index.add %gtm_{q}, %grt_{q} : index")
                e(f"    %gtb_{q} = index.mul %gti_{q}, %c256 : index")
                e(f"    %gtw_{q} = index.mul %gtr_{q}, %c16 : index")
                e(f"    %gtx_{q} = index.add %gtb_{q}, %gtw_{q} : index")
                e(f"    %tix0_{q} = index.add %gtx_{q}, %grr_{q} : index")
                e(f"    %tix_{q} = index.min %tix0_{q}, %out_last : index")
            e(f"    %gix_{q} = index.min %gix0_{q}, %out_last : index")
            if masked:
                e(f"    %gok_{q} = index.cmp ult, %gtok_{q}, %tokens : index")
                e(f"    scf.if %gok_{q} {{")
            e(f"    %g_{q} = view.load %gate_view[%gix_{q}] : view<[%out_total]xf32> -> f32")
            e(f"    %ng_{q} = scalar.mulf %g_{q}, %negone : f32")
            e(f"    %ex_{q} = scalar.expf<afn> %ng_{q} : f32")
            e(f"    %dn_{q} = scalar.addf %one, %ex_{q} : f32")
            recip1(e, f"%iv_{q}", f"%dn_{q}", "    ")
            e(f"    %sg_{q} = scalar.mulf %g_{q}, %iv_{q} : f32")
            e(f"    %ac_{q} = scalar.mulf %sg_{q}, %v_{q} : f32")
            e(f"    %h_{q} = scalar.fptrunc %ac_{q} : f32 to f16")
            e(f"    view.store %h_{q}, %out_h[%{'tix' if t.tout else 'gix'}_{q}] : f16, view<[%out_total]xf16>")
            if masked:
                e("    }")
            e(f"    scf.yield %em{q} : index")
            e("  }")
