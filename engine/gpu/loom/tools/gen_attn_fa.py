#!/usr/bin/env python3
"""Generate yah_attn_wmma, the FlashAttention-style causal prefill attention (softmax in registers), and yah_transpose_v16.

usage: gen_attn_fa.py [out.loom]
       gen_attn_fa.py vtrans [out.loom]

Not HIP's arithmetic order: checked by the T1 numerics gate, not by md5.

Bindings: query and gate (f32 [token][6144]), key_cache, value_cache, output (f16 [token][6144]), lse (not read),
then kscale (quantized K), vstat4 (quantized V) and ptab (paged caches) when those modes are on.
Grid: (query blocks of 32 tokens, head pairs); 256 threads = 8 waves.
One workgroup = 32 query tokens x 2 query heads of one GQA group (4 query blocks of 16).
Wave w owns query block w/2 and head-dim half w%2.
  S^T = K Q^T: K as A (rows = keys), Q^T as B (columns = queries), 8 WMMAs over the wave's 128 dims.
               The wave pair adds partials through a private LDS slot (s = own + partner: f32 add commutes, both waves agree).
  softmax:     lane (q = lane%16, h = lane/16) holds keys 8h..8h+7 of query q.
               K rows are permuted at staging (LDS row 2i+h holds key 8h+i), so the row max is an in-lane max + one xor-16 shuffle.
  P^T as B:    f16(p) of both halves, one xor-16 shuffle of 4 dwords, concat.
  O^T += V^T P^T: V^T as A (rows = dims), permuted at staging so element i of lane half h is dim 8h+i (contiguous output per lane), 8 WMMAs.
K is staged one tile ahead; V(i) is loaded at the top of phase A and staged at its end.
Two barriers per tile: A = QK, S store, stage V; B = stage next K, softmax, P.V.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_kvq  # noqa: E402

V4 = "vector<4xf32>"
V8 = "vector<8xf32>"
V8H = "vector<8xf16>"
V16H = "vector<16xf16>"
V4I = "vector<4xi32>"

# Largest prompt the token-count facts admit; emit_prefill_pp.py sets it to B when B > 2048.
MAX_TOKENS = 2048
# A schedule fence after every QKF QK MMAs caps the K fragments in flight.
QKF = 1    # 1: one K fragment in flight, which with QDIRECT gets the 2 x 32 shape to 192 VGPRs (4 workgroups per WGP)
# PIPE2 (FA3's 2-stage pipeline, same arithmetic per element): the QK of tile j+1 runs beside the softmax and P.V of tile j,
# so a wave's own MMAs fill its softmax's dependent stalls. Region 1: partner S(j), softmax(j), [QK(j+1) from the other
# K buffer, loads of V(j+1), K(j+2)], P.V(j); barrier; region 2: store S(j+1), stage V(j+1), stage K(j+2); barrier.
# K is double-buffered (K2_OFF, inside the prologue Q stage's footprint).
PIPE2 = False
# VSWZ: V^T staging thread t (of a group of 16) stages dim (t%2)*8 + t/2 into LDS row t, instead of dim t into row
# 2*(t%8) + t/8: the same LDS contents, but each 8-lane pass of the b128 stores writes 8 consecutive 48-B rows (8
# disjoint 4-bank groups) instead of rows 0, 2, .. 14 (two lanes per bank group).
VSWZ = True
VSWZ_T = True    # VSWZ also in TILED_OUT builds: there it costs the 4th workgroup (236 VGPRs) but measured 72.3 vs 77.9 M (pp2048)
VSWZR = True     # VSWZ: recompute the staged dim's global offset per tile instead of keeping it live through the loop
VSWZF = 0        # formulation of the swizzled dim (same values; the register allocator is sensitive to it)
# QDIRECT (Q16 only): each lane loads its Q^T fragments straight from the f16 query (lane = query row, 16 contiguous dims
# = one 32-B load per fragment), no LDS Q stage: no prologue stores (4-way bank conflicts), barriers or drain store, and
# the LDS high-water mark drops to the loop's (K, V, S). Same values (f16(q / 16), zero for rows past the chunk).
QDIRECT = None    # None: follow Q16
# PIPE2_EARLY: the QK of tile j+1 goes in before the softmax's rescale branch (its loads stay after it: a branch join
# drains outstanding global loads)
PIPE2_EARLY = False
# The O rescale is skipped when no row max of the wave grew (FA4 conditional rescale).
# The skip path yields alpha = 1.0, so the result has the same bits as always rescaling.
# Rounding follows HIP where it is cheap, so the output stays near the HIP-order golden (T1):
# p = exp2((s - m) * log2e) instead of one fma, the row sum grouped as HIP does per tile
# (4-key partials, (a + b) + (c + d) across the lane halves, sum = fma(sum, prior, part)), and o / sum by IEEE division.
# Workgroup order: head pair fastest, so the 3 pairs of a KV head (GQA 6) run side by side on the same K/V tiles (L2 hits).
# Longest query blocks go first (LPT).

# LDS (bytes). The Q stage (prologue only) aliases the rest.
# Pitches are conflict-free under the 8-lane passes of b128 accesses (128 B): K rows 528 B, V rows 48 B.
# 32-B V rows put lanes r and r+4 on the same banks.
KT_PITCH = 264                       # K: 16 keys x 256 dims (+8 pad)
VT_PITCH = 24                        # V^T: 256 dims x 16 keys (+8 pad)
# K(i+1) is loaded at the top of phase A and staged in phase B of the same tile.
# So nothing in flight crosses the loop back edge, where the compiler drains vmcnt(0).
K_OFF = 0                            # 16 x 264 x 2
V_OFF = 16 * 264 * 2
HPW, QT = 2, 32                      # query heads, query tokens per workgroup
NQB = HPW * QT // 16                 # query blocks (wave pairs)
NT = 64 * NQB                        # threads
KT = 16                              # keys per tile
# VQ8 (V half of kv8a16): V^T as 255-level bytes per channel per 16-key tile, f16 (S, C') = (256 s, c - 384 s) (yah_vq8).
# Staging builds f16 1 + u/256 (0x3c00 | u << 2) with masks, then one packed fma f * S + C'.
# Ranges are per tile, so they stream with chunked prefill (no prompt-wide statistics).
VQ8 = gen_kvq.kv_bits()[1] == 8
# VQ4 (V half of the kv4 configs): V^T as 15-level nibbles per channel per 16-key tile, f16 (S, C') = (16 s, c - 23 s) (yah_vq4).
# Staging builds f16 1 + u/16 (0x3c00 | u << 6) with masks, then one packed fma f * S + C'. P.V stays f16.
VQ4 = gen_kvq.kv_bits()[1] == 4
# Quantized K (engine/run/kvq/README.md): the K cache is centred by its per-channel prompt mean (yah_kmean).
# Staging decodes it to f16, and QK^T runs the f16 path with an unquantized f16 Q.
#   K4 (kv4a16): H256 (k - m) as asymmetric int4 per 32-dim group (yah_kq4).
#     Decoded as (1 + u/16) * 16 s + lo - 16 s, one packed fma per pair; Q is rotated by the same H256 at staging.
#   K8 (kv8a16): int8 per token half (yah_kq8), decoded as (1 + u/256) * 256 s - 384 s.
K4 = gen_kvq.kv_bits()[0] == 4
K8 = gen_kvq.kv_bits()[0] == 8
KDEC = K4 or K8                       # K decoded to f16 at staging
# PAGED: the K / V caches are paged in 256-token pages; ptab[logical page] = physical page.
# K rows (and K scales): row' = ptab[row / 256] * 256 + row % 256.
# V^T tiles (and V stats): tile' = ptab[tile / 16] * 16 + tile % 16. The layouts do not change.
# Every 16-key tile lies in one page: one uniform table load per K tile and per V tile.
# emit_prefill_pp.py clears it when the context is not a multiple of 256.
PAGED = True
# TILED_OUT: store the output fragment-major for an afrag o-projection (gen_gemm_tile Tile.atiled over K = 6144): 16 x 16
# tiles (token / 16, column / 16) of 256 halves, column fastest. A lane's 8 halves sit inside one tile row.
TILED_OUT = False
# GATE_EACH: the epilogue loads each fragment's gate right before its use instead of all 8 up front (64 fewer live
# registers at the epilogue peak; the paged fragment-major build otherwise allocates 236 VGPRs)
GATE_EACH = False
# GATE_FENCE: every gate address first, a schedule fence, then the loads: without it the allocator recycles each address
# register right after its load and Loom waits for that load (vmcnt) before the overwrite, serializing the 16 loads
GATE_FENCE = True
# DIAG_MASK: the causal / context mask (8 compares, ands and selects per lane) under a wave-uniform branch taken only when
# some lane of the wave has a key past its limit or a dead row (the diagonal and last tiles). Same values: an unmasked
# tile's selects pass S through. The branch sits after the K staging and the S partner loads (the mask needs them), so
# its join drains nothing.
DIAG_MASK = True
DIAG_SWAP = True    # the mask branch as 'all lanes unmasked' with the pass-through arm first (layout order)
KCLAMP_S = True     # K tile rows clamped by one scalar min of the tile start (cap - 16), not a per-lane min
KVMAJOR = False     # workgroup order KV head slowest (lost: +67% at 96K, see docs/results.md)
# PAIR: the wave pair of a query block splits by role instead of head_dim: wave A holds all of Q, runs the 16 QK MMAs
# (two 8-MMA chains, dims 0..127 and 128..255, added as the head_dim split adds its partials) and the softmax once, and
# writes P^T (f16) and the rescale factor to LDS; wave B holds all of O and runs the 16 P.V MMAs one tile behind. One
# loop-carried 128-register bank is Q in A and O in B (register allocation is per kernel). Bit-identical. Used for the
# chunks at >= 84K keys of context (emit_prefill_pp DEEP_FLAGS), where 3 workgroups per WGP allow its ~225 VGPRs.
PAIR = False
PAIR_MAP = "mod"    # "mod": pair = waves (p, p + 4), one A and one B per SIMD; "adj": (2p, 2p + 1)
PAIR_LATE = True    # PAIR: K / V staging loads in phase 2 (fewer VGPRs, but their latency is exposed: off when deployed)
PAIR_BFIRST = False # PAIR: the P.V (B) arm first in the role branch
PAIR_I32 = True     # PAIR: carry the bank as vector<8xi32>
PVF = 2             # PAIR: V fragments in flight in wave B's P.V (a schedule fence every PVF MMAs)
DOWHILE = True      # PAIR: bottom-tested loop (scf.while with the body in the condition region): the post-loop code then
                    # reads the last body outputs, not the loop-header block arguments, which Loom's linear live
                    # intervals (no lifetime holes) otherwise keep live across the whole body (PAIR: > 256 VGPRs)
DOWHILE_MAIN = False   # the same for the head_dim-split loop (measured neutral)
# SOFT1..4 (measured, not adopted, docs/results.md): keep the head_dim split and run the softmax in wave A only, P^T and
# alpha through LDS. Bit-identical, -35% VALU, but the waiting partner raises barrier idle: +1.4..+4.4% at 126K.
SOFT1 = False
SOFT2 = False       # P.V(j - 1) beside QK(j) in phase 1, softmax(j) in phase 2 (2 barriers)
SOFT3 = False       # SOFT2 with wave B's P.V moved into phase 2 (overlapping A's softmax)
SOFT4 = False       # phase 1 QK(j), phase 2 both P.V(j - 1) + A's softmax(j)
S1_PASS_FIRST = True
SOFT1_MAP = "mod"
LDS_MIN = 0         # LDS pool floor in bytes (> 32 KB caps a WGP at 3 attention workgroups)
# DEC_LSHADD: in the int4 / int8 K and V decoders, a left-shifted nibble pair ((w << s) & M) | C becomes
# ((w & (M >> s)) << s) + C (the fields do not overlap, so | is +), which Loom selects as v_and_b32 + v_lshl_add_u32:
# 2 VALU instead of 3, same bits
DEC_LSHADD = True


def dec_pair(e, ind, out, w, sh, mask, gname):
    """out = ((w << sh or w >> -sh) & mask) | g, as i32 (mask the f16 mantissa field of both lanes, then OR the 1.0)"""
    if DEC_LSHADD and sh > 0:
        e(f"{ind}{out}_m = scalar.constant {mask >> sh} : i32")
        e(f"{ind}{out}_a = scalar.andi {w}, {out}_m : i32")
        e(f"{ind}{out}_p = scalar.constant {1 << sh} : i32")
        e(f"{ind}{out} = scalar.fmai {out}_a, {out}_p, {gname} : i32")
        return
    op = "shli" if sh >= 0 else "shrui"
    e(f"{ind}{out}_c = scalar.constant {abs(sh)} : i32")
    e(f"{ind}{out}_t = scalar.{op} {w}, {out}_c : i32")
    e(f"{ind}{out}_mk = scalar.constant {mask} : i32")
    e(f"{ind}{out}_x = scalar.andi {out}_t, {out}_mk : i32")
    e(f"{ind}{out} = scalar.ori {out}_x, {gname} : i32")
# Page lookups are scalar (SMEM) loads of the global table; an LDS copy of the table was slower.
# No in-kernel clamp (it cost 1%): the host validates every entry (< npages) before upload,
# and the cache writers clamp page indices into their pools.
assert not (K4 and K8)
# Q16: the query arrives as f16(q * 0.0625) from RoPE (emit_prefill_pp.rope_q16), the scale and rounding this kernel
# applied to the f32 query: half the Q bytes, same bits. Not with K4 (its H256 rotates the scaled f32 Q first).
Q16 = not K4
# QROT (kv4): the Q prologue's H256 rotation runs in its own kernel (gen_qrot, the same operations), which writes f16
# rotated Q; the attention then takes it as Q16 with direct Q fragments (no 33.8 KB Q stage: 4 workgroups per WGP).
# Set QROT and Q16 together.
QROT = False
S_OFF = V_OFF + 256 * VT_PITCH * 2   # S partials: 2 planes x NT x 4 f32
Q_PITCH = 264
Q_END = NQB * 16 * Q_PITCH * 2           # Q stage (prologue only)
# Q-drain dummy store: past the Q stage, or in the S slots when they are past it
DRAIN_OFF = S_OFF if S_OFF >= Q_END else Q_END
K2_OFF = S_OFF + NT * 32                # second K buffer (PIPE2)
POOL = max(S_OFF + NT * 32, DRAIN_OFF + NT * 16)
assert POOL <= 65536


def configure(hpw, qt):
    """Workgroup shape: hpw query heads (of one GQA group) x qt query tokens; recomputes the derived sizes.
    The default is 2 x 32; 6 x 16 packs a whole GQA group (each K / V tile staged once for all 6 heads)."""
    global HPW, QT, NQB, NT, Q_END, DRAIN_OFF, POOL, K2_OFF
    HPW, QT = hpw, qt
    NQB = HPW * QT // 16
    NT = 64 * NQB
    Q_END = NQB * 16 * Q_PITCH * 2
    DRAIN_OFF = S_OFF if S_OFF >= Q_END else Q_END
    K2_OFF = S_OFF + NT * 32
    POOL = max(S_OFF + NT * 32, DRAIN_OFF + NT * 16)
    # QDIRECT has no Q stage or drain slot: gen() sizes the pool S_OFF + NT * 32 then
    assert (S_OFF + NT * 32 if (Q16 if QDIRECT is None else QDIRECT) else POOL) <= 65536


def had64(e, pre, vecs, xors, scale):
    """Walsh-Hadamard times `scale`: stages 1..32 inside this thread's 64 dims (16 vector<4xf32> in dim order),
    then across threads (lane xor x flips qpart bit log2(x): dims 64, 128 apart).
    Butterfly as in yah_kq4 (lower = a + b, upper = lower - upper), so both sides use the same orthogonal H."""
    xs = []
    for c in range(16):
        for i in range(4):
            e(f"  %{pre}0_{4 * c + i} = vector.extract {vecs[c]}[{i}] : {V4} -> f32")
            xs.append(f"%{pre}0_{4 * c + i}")
    st = 0
    for h in (1, 2, 4, 8, 16, 32):
        st += 1
        nx = []
        for i in range(64):
            j = i ^ h
            if i & h:
                e(f"  %{pre}{st}_{i} = scalar.subf {xs[j]}, {xs[i]} : f32")
            else:
                e(f"  %{pre}{st}_{i} = scalar.addf {xs[i]}, {xs[j]} : f32")
            nx.append(f"%{pre}{st}_{i}")
        xs = nx
    cur = []
    for c in range(16):
        e(f"  %{pre}v{c} = vector.from_elements {', '.join(xs[4 * c:4 * c + 4])} : {V4}")
        cur.append(f"%{pre}v{c}")
    for x in xors:
        e(f"  %{pre}x{x}c = scalar.constant {x} : i32")
        e(f"  %{pre}b{x}a = index.div %qpart, %c{x} : index")
        e(f"  %{pre}b{x}b = index.rem %{pre}b{x}a, %c2 : index")
        e(f"  %{pre}u{x} = index.cmp eq, %{pre}b{x}b, %c1 : index")
        nxt = []
        for c in range(16):
            e(f"  %{pre}s{x}_{c}, %{pre}sv{x}_{c} = kernel.subgroup.shuffle<xor> {cur[c]}, %{pre}x{x}c, %x32 : {V4}, i32, i32")
            e(f"  %{pre}a{x}_{c} = vector.addf {cur[c]}, %{pre}s{x}_{c} : {V4}")
            e(f"  %{pre}d{x}_{c} = vector.subf %{pre}s{x}_{c}, {cur[c]} : {V4}")
            e(f"  %{pre}f{x}_{c} = scf.select %{pre}u{x}, %{pre}d{x}_{c}, %{pre}a{x}_{c} : {V4}")
            nxt.append(f"%{pre}f{x}_{c}")
        cur = nxt
    e(f"  %{pre}rs = scalar.constant {scale} : f32")
    e(f"  %{pre}rs4 = vector.splat %{pre}rs : {V4}")
    out = []
    for c in range(16):
        e(f"  %{pre}r{c} = vector.mulf {cur[c]}, %{pre}rs4 : {V4}")
        out.append(f"%{pre}r{c}")
    return out


def gen():
    L = []
    e = L.append
    e("// GENERATED by tools/gen_attn_fa.py -- edit the generator.")
    e("// FlashAttention-style causal prefill attention, register softmax.")
    e("amdgpu.target<gfx1151> @attn_fa_w32 {subgroup_size = 32}")
    e("")
    e("config.def @attention_prefill.cache_capacity = 2048 : index")
    e(f"config.decl @attention_prefill.token_count : %value: index where [range(%value, 1, {MAX_TOKENS})]")
    e("config.decl @attention_prefill.num_heads : %value: index where [range(%value, 1, 4096)]")
    e("config.decl @attention_prefill.num_kv_heads : %value: index where [range(%value, 1, 4096)]")
    e("config.decl @attention_prefill.gqa : %value: index where [range(%value, 1, 4096)]")
    e("config.decl @attention_prefill.head_dim : %value: index where [range(%value, 1, 1024)]")
    e("config.decl @attention_prefill.start_pos : %value: index where [range(%value, 0, 1073741824)]")
    e("")
    e("kernel.def target(@attn_fa_w32) @yah_attn_wmma() {")
    e("  %token_count = config.get @attention_prefill.token_count : index")
    e("  %c1 = index.constant 1 : index")
    e("  %c2 = index.constant 2 : index")
    e("  %c31 = index.constant 31 : index")
    e("  %c32 = index.constant 32 : index")
    e("  %c256 = index.constant 256 : index")
    e("  %nh = config.get @attention_prefill.num_heads : index")
    e(f"  %chpw = index.constant {HPW} : index")
    e(f"  %cqt1 = index.constant {QT - 1} : index")
    e(f"  %cqt = index.constant {QT} : index")
    e(f"  %cnt = index.constant {NT} : index")
    e("  %pairs = index.div %nh, %chpw : index")
    e("  %tp = index.add %token_count, %cqt1 : index")
    e("  %qblocks = index.div %tp, %cqt : index")
    e("  kernel.launch.config workgroups(%qblocks, %pairs, %c1) workgroup_size(%cnt, %c1, %c1) : index")
    e("} launch(%query: buffer, %gate: buffer, %key_cache: buffer, %value_cache: buffer, %output: buffer, %lse: buffer"
      + (", %kscale: buffer" if KDEC else "")
      + (", %vstat4: buffer" if VQ4 or VQ8 else "") + (", %ptab: buffer" if PAGED else "") + ") {")
    e("  %base = index.constant 0 : offset")
    for v in (0, 1, 2, 3, 4, 6, 8, 15, 16, 31, 32, 64, 128, 256, 1024, 4096, 6144):
        e(f"  %c{v} = index.constant {v} : index")
    e(f"  %chpw = index.constant {HPW} : index")
    e(f"  %cqt = index.constant {QT} : index")
    e(f"  %cqt1 = index.constant {QT - 1} : index")
    e(f"  %cnt = index.constant {NT} : index")
    e("  %cache_capacity = config.get @attention_prefill.cache_capacity : index")
    e("  %token_count0 = config.get @attention_prefill.token_count : index")
    e(f"  %B = index.assume %token_count0 [range(%token_count0, 1, {MAX_TOKENS})] : index")
    e("  %start_pos = config.get @attention_prefill.start_pos : index")
    e("  %zero = scalar.constant 0.0 : f32")
    e("  %one = scalar.constant 1.0 : f32")
    e("  %ninf = scalar.constant -3.4028234663852886e+38 : f32")
    e("  %log2e = scalar.constant 1.4426950408889634 : f32")
    e("  %qscale = scalar.constant 0.0625 : f32")
    e("  %zh8 = vector.constant 0.0 : vector<8xf16>")
    e(f"  %zeros8 = vector.constant 0.0 : {V8}")
    e(f"  %zq16 = vector.constant 0.0 : {V16H}")
    e(f"  %ones8 = vector.constant 1.0 : {V8}")
    e(f"  %log2e8 = vector.splat %log2e : {V8}")
    e(f"  %ninf8 = vector.splat %ninf : {V8}")
    e("  %m = index.constant 16 : index")
    e("  %n = index.constant 16 : index")
    e("  %k = index.constant 16 : index")
    e("  %x16 = scalar.constant 16 : i32")
    e("  %x32 = scalar.constant 32 : i32")
    e("  %qtot = index.mul %B, %c6144 : index")
    CAPV = "%cache_capacity"
    if PAGED:
        e("  %c255p = index.constant 255 : index")
        e("  %npg0 = index.add %cache_capacity, %c255p : index")
        e("  %npages = index.div %npg0, %c256 : index")
        e("  %pcap = index.mul %npages, %c256 : index")       # pool rows: whole pages
        CAPV = "%pcap"
    e(f"  %kvtot = index.mul {CAPV}, %c1024 : index")
    e("  %q_na, %g_na, %k_na, %v_na, %o_na = buffer.assume.noalias %query, %gate, %key_cache, %value_cache, %output : buffer, buffer, buffer, buffer, buffer")
    e(f"  %q_flat = buffer.view %q_na[%base] : buffer -> view<[%qtot]x{'f16' if Q16 else 'f32'}>")
    e("  %g_flat = buffer.view %g_na[%base] : buffer -> view<[%qtot]xf32>")
    e("  %o_flat = buffer.view %o_na[%base] : buffer -> view<[%qtot]xf16>")
    # kv4a16: int4 K [token][128 dwords], f16x2 (16 s, lo - 16 s) [token][32]
    # kv8a16: int8 K [token][256 dwords], f16x2 (256 s, -384 s) [token][8]
    if KDEC:
        e(f"  %kq32tot = index.mul {CAPV}, %c{128 if K4 else 256} : index")
        e(f"  %kstot = index.mul {CAPV}, %c{32 if K4 else 8} : index")
        e("  %k_flat = buffer.view %k_na[%base] : buffer -> view<[%kq32tot]xi32>")
        e("  %ks_na = buffer.assume.noalias %kscale : buffer")
        e("  %ks_flat = buffer.view %ks_na[%base] : buffer -> view<[%kstot]xi32>")
    else:
        e("  %k_flat = buffer.view %k_na[%base] : buffer -> view<[%kvtot]xf16>")
    if PAGED:
        e("  %pt_na = buffer.assume.noalias %ptab : buffer")
        e("  %pt_flat = buffer.view %pt_na[%base] : buffer -> view<[%npages]xi32>")
    e(f"  %cap15 = index.add {CAPV}, %c15 : index")
    e("  %vtiles = index.div %cap15, %c16 : index")
    e("  %vpitch = index.mul %vtiles, %c16 : index")
    e("  %vtot = index.mul %vpitch, %c1024 : index")
    e("  %vlast = index.sub %vpitch, %c16 : index")
    if VQ4 or VQ8:   # V^T [kvh][tile][256 dims] x 2 (nibbles) or 4 (bytes) dwords, stats 1 dword
        assert not (VQ4 and VQ8)
        e("  %vq4tot0 = index.mul %vtiles, %c1024 : index")
        e(f"  %vq4tot = index.mul %vq4tot0, %c{2 if VQ4 else 4} : index")
        e("  %v_flat = buffer.view %v_na[%base] : buffer -> view<[%vq4tot]xi32>")
        e("  %vs4_na = buffer.assume.noalias %vstat4 : buffer")
        e("  %vs4_flat = buffer.view %vs4_na[%base] : buffer -> view<[%vq4tot0]xi32>")
    else:
        e("  %v_flat = buffer.view %v_na[%base] : buffer -> view<[%vtot]xf16>")
    pool = S_OFF + NT * 32 if (Q16 if QDIRECT is None else QDIRECT) else POOL      # QDIRECT: no Q stage / drain slot
    pool = max(pool, K2_OFF + KT * KT_PITCH * 2) if PIPE2 else pool
    if PAIR:   # no S exchange: P^T slots 2 x 4 pairs x 512 B, alpha 2 x 256 B, row sums 256 B
        pool = S_OFF + 4096 + 512 + 256
    if SOFT1:  # S partials (B's), then P^T 4 pairs x 512 B, alpha 256 B, row sums 256 B (x2 slots under SOFT3)
        pool = S_OFF + NT * 32 + (4096 + 768 if SOFT3 or SOFT4 else 2048 + 512)
    pool = max(pool, LDS_MIN)
    assert pool <= 65536
    e(f"  %pool_bytes = index.constant {pool} : offset")
    e("  %pool = buffer.alloca<workgroup> align(16) %pool_bytes : buffer")
    e(f"  %qs_view = buffer.view %pool[%base] : buffer -> view<{NQB * 16}x{Q_PITCH}xf16>")
    e(f"  %q_lay = encoding.layout.strided [1, {Q_PITCH}] : encoding<layout>")
    e(f"  %q_fr = buffer.view %pool[%base] : buffer -> view<256x{NQB * 16}xf16, %q_lay>")
    e(f"  %k_o = index.constant {K_OFF} : offset")
    e(f"  %k_view = buffer.view %pool[%k_o] : buffer -> view<{KT}x{KT_PITCH}xf16>")
    e(f"  %v_o = index.constant {V_OFF} : offset")
    e(f"  %v_view = buffer.view %pool[%v_o] : buffer -> view<256x{VT_PITCH}xf16>")
    e(f"  %s_o = index.constant {S_OFF} : offset")
    # two planes of 4 f32 per lane, so each b128 access is contiguous across lanes
    e(f"  %s_view = buffer.view %pool[%s_o] : buffer -> view<{2 * NT}x4xf32>")
    e(f"  %s8_view = buffer.view %pool[%s_o] : buffer -> view<{NT}x8xf32>")
    PRS = 256 if (SOFT3 or SOFT4) else 128   # SOFT1 P^T rows (2 slots under SOFT3)
    PRS2 = PRS // 2
    if SOFT1:
        PR = 256 if (SOFT3 or SOFT4) else 128
        e(f"  %pp_o = index.constant {S_OFF + NT * 32} : offset")
        e(f"  %pp_view = buffer.view %pool[%pp_o] : buffer -> view<{PR}x8xf16>")
        e(f"  %al_o = index.constant {S_OFF + NT * 32 + PR * 16} : offset")
        e(f"  %al_view = buffer.view %pool[%al_o] : buffer -> view<{PR // 2}xf32>")
        e(f"  %ls_o = index.constant {S_OFF + NT * 32 + PR * 16 + PR * 2} : offset")
        e("  %ls_view = buffer.view %pool[%ls_o] : buffer -> view<64xf32>")
    if PAIR:
        e("  %pp_view = buffer.view %pool[%s_o] : buffer -> view<256x8xf16>")
        e(f"  %al_o = index.constant {S_OFF + 4096} : offset")
        e("  %al_view = buffer.view %pool[%al_o] : buffer -> view<128xf32>")
        e(f"  %ls_o = index.constant {S_OFF + 4608} : offset")
        e("  %ls_view = buffer.view %pool[%ls_o] : buffer -> view<64xf32>")
    # ids: head pair fastest, longest query blocks first
    e("  %wgx = kernel.workgroup.id<x> : index")
    e("  %wgy = kernel.workgroup.id<y> : index")
    e("  %nh_ = config.get @attention_prefill.num_heads : index")
    e("  %npairs = index.div %nh_, %chpw : index")
    e("  %tpq = index.add %B, %cqt1 : index")
    e("  %nqb = index.div %tpq, %cqt : index")
    e("  %wgl0 = index.mul %wgy, %nqb : index")
    e("  %wgl = index.add %wgl0, %wgx : index")
    if KVMAJOR:   # KV head slowest: co-resident workgroups stream the same K / V (one KV head's pairs fastest)
        e("  %nkvh_ = config.get @attention_prefill.num_kv_heads : index")
        e("  %pkv = index.div %npairs, %nkvh_ : index")
        e("  %kvblk = index.mul %pkv, %nqb : index")
        e("  %kvh_m = index.div %wgl, %kvblk : index")
        e("  %kvr = index.rem %wgl, %kvblk : index")
        e("  %pig_m = index.rem %kvr, %pkv : index")
        e("  %qb0 = index.div %kvr, %pkv : index")
        e("  %hp0 = index.mul %kvh_m, %pkv : index")
        e("  %hp = index.add %hp0, %pig_m : index")
    else:
        e("  %hp = index.rem %wgl, %npairs : index")
        e("  %qb0 = index.div %wgl, %npairs : index")
    e("  %nqb1 = index.sub %nqb, %c1 : index")
    e("  %qb = index.sub %nqb1, %qb0 : index")
    e("  %tid = kernel.workitem.id<x> : index")
    e("  %wave = index.div %tid, %c32 : index")
    e("  %lane = index.rem %tid, %c32 : index")
    e("  %sub = index.rem %lane, %c16 : index")
    e("  %half = index.div %lane, %c16 : index")
    if PAIR or SOFT1:   # the role from the subgroup id: wave-uniform to Loom, so the role branch is a scalar branch
        e("  %psg = kernel.subgroup.id : index")
    if PAIR and PAIR_MAP == "mod":
        e("  %wqb = index.rem %psg, %c4 : index")       # query block 0..3
        e("  %prole = index.div %psg, %c4 : index")     # 0: A (QK + softmax), 1: B (P.V)
        e("  %hd = index.add %c0, %c0 : index")
    elif PAIR:
        e("  %wqb = index.div %psg, %c2 : index")
        e("  %prole = index.rem %psg, %c2 : index")
        e("  %hd = index.add %c0, %c0 : index")
    elif SOFT1 and SOFT1_MAP == "mod":
        e("  %wqb = index.rem %psg, %c4 : index")
        e("  %hd = index.div %psg, %c4 : index")
    elif SOFT1:
        e("  %wqb = index.div %psg, %c2 : index")
        e("  %hd = index.rem %psg, %c2 : index")
    else:
        e("  %wqb = index.div %wave, %c2 : index")       # query block 0..3
        e("  %hd = index.rem %wave, %c2 : index")        # head-dim half
    if PAIR:
        e("  %isA = index.cmp eq, %prole, %c0 : index")
    if SOFT1:
        e("  %isA = index.cmp eq, %hd, %c0 : index")
        e("  %isBs = index.cmp ne, %hd, %c0 : index")
    e("  %hd128 = index.mul %hd, %c128 : index")
    e("  %hd64 = index.mul %hd, %c64 : index")
    if SOFT1:   # only A reads the partner's (B's) partial
        e(f"  %pt0 = index.add %tid, %c{128 if SOFT1_MAP == 'mod' else 32} : index")
        e("  %ptid = index.add %pt0, %c0 : index")
    else:
        e("  %pt0 = index.add %tid, %c32 : index")
        e("  %ptid = index.sub %pt0, %hd64 : index")     # partner lane (tid ^ 32)
    e("  %tid2 = index.add %tid, %cnt : index")
    e("  %ptid2 = index.add %ptid, %cnt : index")
    e("  %h0 = index.cmp eq, %half, %c0 : index")
    # opaque 1.0 / 0.0 (lane & ~lane is 0, unprovable to the folder)
    e("  %lanei = index.cast %lane : index to i32")
    e("  %xm1 = scalar.constant -1 : i32")
    e("  %onebits = scalar.constant 1065353216 : i32")
    e("  %lanen = scalar.xori %lanei, %xm1 : i32")
    e("  %lz = scalar.andi %lanei, %lanen : i32")
    e("  %lob = scalar.ori %lz, %onebits : i32")
    e("  %one_o = scalar.bitcast %lob : i32 to f32")
    e("  %zero_o = scalar.bitcast %lz : i32 to f32")
    e("  %qs = index.mul %qb, %cqt : index")
    if HPW == 6:   # one workgroup = a whole GQA group
        e("  %kvh = index.add %hp, %c0 : index")
        e("  %head0 = index.mul %kvh, %c6 : index")
    else:
        assert HPW == 2
        e("  %kvh = index.div %hp, %c3 : index")
        e("  %pig = index.rem %hp, %c3 : index")
        e("  %kvh6 = index.mul %kvh, %c6 : index")
        e("  %pig2 = index.mul %pig, %c2 : index")
        e("  %head0 = index.add %kvh6, %pig2 : index")
    e("  %kvbase = index.mul %kvh, %c256 : index")
    e("  %vhb0 = index.mul %kvh, %vtiles : index")
    e("  %vhb = index.mul %vhb0, %c4096 : index")
    # V staging: lane = dim, the last 256 threads (K: the first 256)
    e(f"  %vtoff = index.constant {NT - 256} : index")
    if NT > 256:   # max(): threads below vtoff only stage K (their V stores are guarded) but stay in range
        e("  %vtm = index.max %tid, %vtoff : index")
        e("  %vt = index.sub %vtm, %vtoff : index")
    else:
        e("  %vt = index.add %tid, %c0 : index")
    if VSWZ and (VSWZ_T or not TILED_OUT):   # with TILED_OUT the paged build allocates 236 VGPRs (192 without)
        if VSWZF == 0:
            e("  %vsl = index.rem %vt, %c16 : index")
            e("  %vsg = index.div %vt, %c16 : index")
            e("  %vsb = index.mul %vsg, %c16 : index")
            e("  %vsl2 = index.rem %vsl, %c2 : index")
            e("  %vsl8 = index.mul %vsl2, %c8 : index")
            e("  %vslh = index.div %vsl, %c2 : index")
            e("  %vdimo = index.add %vsl8, %vslh : index")
            e("  %vdim0 = index.add %vsb, %vdimo : index")
        elif VSWZF == 1:   # dim = (t & ~15) | ((t & 1) << 3) | ((t >> 1) & 7) on the lane id
            e("  %vsl2 = index.rem %vt, %c2 : index")
            e("  %vsl8 = index.mul %vsl2, %c8 : index")
            e("  %vsh = index.div %vt, %c2 : index")
            e("  %vsh8 = index.rem %vsh, %c8 : index")
            e("  %vsg = index.div %vt, %c16 : index")
            e("  %vsb = index.mul %vsg, %c16 : index")
            e("  %vdimo = index.add %vsl8, %vsh8 : index")
            e("  %vdim0 = index.add %vsb, %vdimo : index")
        else:              # as a permutation of the lane only: the 16-group base comes from the unswizzled t
            e("  %vsl = index.rem %vt, %c16 : index")
            e("  %vsb = index.sub %vt, %vsl : index")
            e("  %vsl2 = index.rem %vt, %c2 : index")
            e("  %vsl8 = index.mul %vsl2, %c8 : index")
            e("  %vslh = index.div %vsl, %c2 : index")
            e("  %vdimo = index.add %vsl8, %vslh : index")
            e("  %vdim0 = index.add %vsb, %vdimo : index")
        e("  %c255v = index.constant 255 : index")
        e("  %vdim = index.min %vdim0, %c255v : index")   # always in range; states it for the bound proof
    else:
        e("  %vdim = index.add %vt, %c0 : index")
    e("  %vlane = index.mul %vdim, %c16 : index")
    e("  %vhl = index.add %vhb, %vlane : index")
    if VQ4 or VQ8:
        e("  %vhb4 = index.mul %kvh, %vtiles : index")
        if VQ4:
            e("  %v4m = scalar.constant 62915520 : i32")        # 0x03c003c0
        else:
            e("  %v8m = scalar.constant 66847740 : i32")        # 0x03fc03fc
        e("  %v4g = scalar.constant 1006648320 : i32")     # 0x3c003c00
    e("  %ctx_end = index.add %start_pos, %B : index")
    e("  %qs32 = index.add %qs, %cqt : index")
    e("  %vis0 = index.add %start_pos, %qs32 : index")
    e("  %max_vis = index.min %ctx_end, %vis0 : index")
    e("  %B_1 = index.sub %B, %c1 : index")
    e("  %cap_1 = index.sub %cache_capacity, %c1 : index")
    if KCLAMP_S:   # (only then: even an unused op perturbs the allocator)
        e("  %klast = index.sub %cache_capacity, %c16 : index")
    # this lane's query: block wqb = (head0 + wqb%2, tokens qs + (wqb/2)*16 + sub)
    e("  %wqh = index.rem %wqb, %chpw : index")
    e("  %wqt0 = index.div %wqb, %chpw : index")
    e("  %wqt1 = index.mul %wqt0, %c16 : index")
    e("  %wqt = index.add %qs, %wqt1 : index")
    e("  %r_lq = index.add %wqt, %sub : index")
    e("  %r_abs = index.add %start_pos, %r_lq : index")
    e("  %r_live = index.cmp ult, %r_lq, %B : index")
    e("  %r_head = index.add %head0, %wqh : index")
    qdirect = Q16 if QDIRECT is None else QDIRECT
    if qdirect:
        assert Q16 and (not K4 or QROT)
        e("  %qdq = index.min %r_lq, %B_1 : index")
        e("  %qdrow = index.mul %qdq, %c6144 : index")
        e("  %qdh = index.mul %r_head, %c256 : index")
        e("  %qdb0 = index.add %qdrow, %qdh : index")
        e("  %qdb = index.add %qdb0, %hd128 : index")
        e(f"  %zq16x = vector.concat<0> %zh8, %zh8 : {V8H}, {V8H} -> {V16H}")
        for c in range(16 if PAIR else 8):
            e(f"  %qdc{c} = index.constant {16 * c} : index")
            e(f"  %qda{c} = index.add %qdb, %qdc{c} : index")
            e(f"  %qdv{c} = vector.load %q_flat[%qda{c}] : view<[%qtot]xf16> -> {V16H}")
            e(f"  %qdz{c} = scf.select %r_live, %qdv{c}, %zq16x : {V16H}")
            if PAIR:   # one 128-register bank: Q bits in wave A, zero O in wave B
                bk = "vector<8xi32>" if PAIR_I32 else V8
                e(f"  %qbits{c} = vector.bitcast %qdz{c} : {V16H} to {bk}")
                if c == 0:
                    e(f"  %bzero = vector.bitcast %zeros8 : {V8} to {bk}")
                e(f"  %rinit{c} = scf.select %isA, %qbits{c}, %bzero : {bk}")
            else:
                e(f"  %qf{c} = vector.fragment<rhs> %qdz{c} shape [%k, %n] : {V16H}")
    else:
        # ---- Q stage: row r = rb*16 + rr
        e("  %qr = index.div %tid, %c4 : index")
        e("  %qpart = index.rem %tid, %c4 : index")
        e("  %qrb = index.div %qr, %c16 : index")
        e("  %qrr = index.rem %qr, %c16 : index")
        e("  %qrb2 = index.rem %qrb, %chpw : index")
        e("  %qrh = index.add %head0, %qrb2 : index")
        e("  %qrq0 = index.div %qrb, %chpw : index")
        e("  %qrq1 = index.mul %qrq0, %c16 : index")
        e("  %qrq2 = index.add %qs, %qrq1 : index")
        e("  %qlq = index.add %qrq2, %qrr : index")
        e("  %qlive = index.cmp ult, %qlq, %B : index")
        e("  %qlqc = index.min %qlq, %B_1 : index")
        e("  %qrow0 = index.mul %qlqc, %c6144 : index")
        e("  %qhb = index.mul %qrh, %c256 : index")
        e("  %qrow = index.add %qrow0, %qhb : index")
        e("  %qsc_v = vector.splat %qscale : " + V4)
        if K4:
            e(f"  %z4h = vector.constant 0.0 : {V4}")
        e("  %qdb0a = index.mul %qpart, %c64 : index")
        rotq = {}
        if K4:   # H256 of the scaled f32 Q (dims qpart*64 + 8c + 0..7) first
            raw = []
            for c in range(8):
                t = f"0_{c}"
                e(f"  %qdc{t} = index.constant {8 * c} : index")
                e(f"  %qd{t} = index.add %qdb0a, %qdc{t} : index")
                e(f"  %qa{t} = index.add %qrow, %qd{t} : index")
                e(f"  %qa{t}b = index.add %qa{t}, %c4 : index")
                e(f"  %qv{t}a = vector.load %q_flat[%qa{t}] : view<[%qtot]xf32> -> {V4}")
                e(f"  %qv{t}b = vector.load %q_flat[%qa{t}b] : view<[%qtot]xf32> -> {V4}")
                e(f"  %qm{t}a0 = vector.mulf %qv{t}a, %qsc_v : {V4}")
                e(f"  %qm{t}b0 = vector.mulf %qv{t}b, %qsc_v : {V4}")
                raw += [f"%qm{t}a0", f"%qm{t}b0"]
            rot = had64(e, "fh", raw, (1, 2), 0.0625)
            for c in range(8):
                rotq[c] = (rot[2 * c], rot[2 * c + 1])
        for c in range(8):
            t = f"0_{c}"
            if K4:
                e(f"  %qm{t}a = vector.addf {rotq[c][0]}, %z4h : {V4}")
                e(f"  %qm{t}b = vector.addf {rotq[c][1]}, %z4h : {V4}")
            else:
                e(f"  %qdc{t} = index.constant {8 * c} : index")
                e(f"  %qd{t} = index.add %qdb0a, %qdc{t} : index")
                e(f"  %qdg{t}c = index.constant 0 : index")
                e(f"  %qdg{t} = index.add %qd{t}, %qdg{t}c : index")
                e(f"  %qa{t} = index.add %qrow, %qdg{t} : index")
                if Q16:
                    e(f"  %qh{t} = vector.load %q_flat[%qa{t}] : view<[%qtot]xf16> -> {V8H}")
                else:
                    e(f"  %qa{t}b = index.add %qa{t}, %c4 : index")
                    e(f"  %qv{t}a = vector.load %q_flat[%qa{t}] : view<[%qtot]xf32> -> {V4}")
                    e(f"  %qv{t}b = vector.load %q_flat[%qa{t}b] : view<[%qtot]xf32> -> {V4}")
                    e(f"  %qm{t}a = vector.mulf %qv{t}a, %qsc_v : {V4}")
                    e(f"  %qm{t}b = vector.mulf %qv{t}b, %qsc_v : {V4}")
            if not (Q16 and not K4):
                e(f"  %qh{t}a = vector.fptrunc %qm{t}a : {V4} to vector<4xf16>")
                e(f"  %qh{t}b = vector.fptrunc %qm{t}b : {V4} to vector<4xf16>")
                e(f"  %qh{t} = vector.concat<0> %qh{t}a, %qh{t}b : vector<4xf16>, vector<4xf16> -> {V8H}")
            e(f"  %qz{t} = scf.select %qlive, %qh{t}, %zh8 : {V8H}")
            e(f"  vector.store %qz{t}, %qs_view[%qr, %qd{t}] : {V8H}, view<{NQB * 16}x{Q_PITCH}xf16>")
        e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        # Q^T fragments (B: [dim, query]) of this wave's 128 dims, in registers
        e("  %wq16 = index.mul %wqb, %c16 : index")
        names = []
        for c in range(8):
            e(f"  %qfd0{c}c = index.constant {16 * c} : index")
            e(f"  %qfd0{c} = index.add %hd128, %qfd0{c}c : index")
            e(f"  %ql0f{c} = vector.fragment.load<rhs> %q_fr[%qfd0{c}, %wq16] shape [%k, %n] : view<256x{NQB * 16}xf16, %q_lay> -> {V16H}")
            names.append(f"%ql0f{c}")
        # Loom does not drain these LDS loads before the barrier below (no lgkmcnt(0) ahead of s_barrier).
        # The prologue then stages K/V over the Q stage, so a fast wave can overwrite Q rows a slow wave still reads.
        # An LDS store of a value built from both halves of every fragment forces the drain first.
        acc = None
        for c in range(8):
            for j in (0, 8):
                e(f"  %qx0{c}_{j} = vector.extract {names[c]}[{j}] : {V16H} -> f16")
                if acc is None:
                    acc = f"%qx0{c}_{j}"
                else:
                    e(f"  %qy0{c}_{j} = scalar.addf {acc}, %qx0{c}_{j} : f16")
                    acc = f"%qy0{c}_{j}"
        e(f"  %qdrain0 = vector.splat {acc} : {V8H}")
        e(f"  %qdr0_o = index.constant {DRAIN_OFF} : offset")
        e(f"  %qdr0_v = buffer.view %pool[%qdr0_o] : buffer -> view<{NT}x8xf16>")
        e(f"  vector.store %qdrain0, %qdr0_v[%tid, %c0] : {V8H}, view<{NT}x8xf16>")
        e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        for c in range(8):
            e(f"  %qf{c} = vector.fragment<rhs> %ql0f{c} shape [%k, %n] : {V16H}")

    # Staging maps. K: item j = tid + 256nn, key = j/32, d8 = (j%32)*8, LDS row 2*(key%8) + key/8.
    # One key row per wave: coalesced global, contiguous LDS.
    # V^T: lane = dim, LDS row (dim/16)*16 + 2*(dim%8) + (dim%16)/8.
    e("  %vdl = index.rem %vt, %c16 : index")
    e("  %vdb = index.sub %vt, %vdl : index")
    e("  %vdl8 = index.rem %vdl, %c8 : index")
    e("  %vdh = index.div %vdl, %c8 : index")
    e("  %vdr0 = index.mul %vdl8, %c2 : index")
    e("  %vdr1 = index.add %vdr0, %vdh : index")
    if VSWZ and (VSWZ_T or not TILED_OUT):   # with TILED_OUT the paged build allocates 236 VGPRs (192 without)
        e("  %vrow = index.add %vt, %c0 : index")
    else:
        e("  %vrow = index.add %vdb, %vdr1 : index")

    for nn in range(2):
        e(f"  %ki{nn}c = index.constant {256 * nn} : index")
        e(f"  %ki{nn} = index.add %tid, %ki{nn}c : index")
        e(f"  %kk{nn} = index.div %ki{nn}, %c32 : index")
        e(f"  %kd{nn}a = index.rem %ki{nn}, %c32 : index")
        e(f"  %kd{nn} = index.mul %kd{nn}a, %c8 : index")
        e(f"  %kk{nn}l = index.rem %kk{nn}, %c8 : index")
        e(f"  %kk{nn}h = index.div %kk{nn}, %c8 : index")
        e(f"  %kk{nn}r0 = index.mul %kk{nn}l, %c2 : index")
        e(f"  %kr{nn} = index.add %kk{nn}r0, %kk{nn}h : index")

    if KDEC:
        # thread = (key tid/16, 16-dim chunk tid%16); LDS row 2*(key%8) + key/8
        e("  %kdk = index.div %tid, %c16 : index")
        e("  %kdc = index.rem %tid, %c16 : index")
        e("  %kdkl = index.rem %kdk, %c8 : index")
        e("  %kdkh = index.div %kdk, %c8 : index")
        e("  %kdr0 = index.mul %kdkl, %c2 : index")
        e("  %kdr = index.add %kdr0, %kdkh : index")
        e("  %kdc16 = index.mul %kdc, %c16 : index")
        e("  %kdc16b = index.add %kdc16, %c8 : index")
        if K4:   # 2 dwords at head*32 + chunk*2; scale group head*8 + chunk/2
            e("  %kdh0 = index.mul %kvh, %c32 : index")
            e("  %kdc2 = index.mul %kdc, %c2 : index")
            e("  %kdga = index.add %kdh0, %kdc2 : index")
            e("  %kdh1 = index.mul %kvh, %c8 : index")
            e("  %kdg = index.div %kdc, %c2 : index")
            e("  %kdso = index.add %kdh1, %kdg : index")
            e("  %kdm = scalar.constant 62915520 : i32")     # 0x03c003c0
        else:      # 4 dwords at head*64 + chunk*4; scale head*2 + half (chunk/8)
            e("  %kdh0 = index.mul %kvh, %c64 : index")
            e("  %kdc4 = index.mul %kdc, %c4 : index")
            e("  %kdga = index.add %kdh0, %kdc4 : index")
            e("  %kdh1 = index.mul %kvh, %c2 : index")
            e("  %kdg = index.div %kdc, %c8 : index")
            e("  %kdso = index.add %kdh1, %kdg : index")
            e("  %kdm = scalar.constant 66847740 : i32")     # 0x03fc03fc
        e("  %kdgm = scalar.constant 1006648320 : i32")      # 0x3c003c00

    def page_of(start, p, ind, tag):
        """Physical page of the (uniform) tile start; start is already clamped."""
        e(f"{ind}%{p}{tag}lp = index.div {start}, %c256 : index")
        e(f"{ind}%{p}{tag}pg0 = view.load %pt_flat[%{p}{tag}lp] : view<[%npages]xi32> -> i32")
        e(f"{ind}%{p}{tag}pgr = index.cast %{p}{tag}pg0 : i32 to index")
        e(f"{ind}%{p}{tag}pg = index.assume %{p}{tag}pgr [range(%{p}{tag}pgr, 0, 65535), lt(%{p}{tag}pgr, %npages)] : index")
        return f"%{p}{tag}pg"

    def phys_row(kc, ks, p, ind, tag):
        """Row kc (clamped, in the same page as the clamped ks) -> physical row."""
        if not PAGED:
            return kc
        e(f"{ind}%{p}{tag}ksc = index.min {ks}, %cap_1 : index")
        pg = page_of(f"%{p}{tag}ksc", p, ind, tag)
        e(f"{ind}%{p}{tag}pb = index.mul {pg}, %c256 : index")
        e(f"{ind}%{p}{tag}po = index.rem {kc}, %c256 : index")
        e(f"{ind}%{p}{tag}pr = index.add %{p}{tag}pb, %{p}{tag}po : index")
        return f"%{p}{tag}pr"

    def phys_tilestart(vks, p, ind, tag):
        """V^T tile start vks (a multiple of 16) -> physical tile start (in keys)."""
        if not PAGED:
            return vks
        pg = page_of(vks, p, ind, tag)
        # physical tile = page * 16 + (tile % 16): bounded by npages * 16 = vtiles
        e(f"{ind}%{p}{tag}pb = index.mul {pg}, %c16 : index")
        e(f"{ind}%{p}{tag}t16 = index.div {vks}, %c16 : index")
        e(f"{ind}%{p}{tag}po = index.rem %{p}{tag}t16, %c16 : index")
        e(f"{ind}%{p}{tag}pt = index.add %{p}{tag}pb, %{p}{tag}po : index")
        e(f"{ind}%{p}{tag}pr = index.mul %{p}{tag}pt, %c16 : index")
        return f"%{p}{tag}pr"

    def load_k(ks, p, ind):
        if KDEC:
            nw = 2 if K4 else 4
            e(f"{ind}%{p}kp = index.add {ks}, %kdk : index")
            if PAGED:
                e(f"{ind}%{p}kpc0 = index.min %{p}kp, %cap_1 : index")
                kpc = phys_row(f"%{p}kpc0", ks, p, ind, "kq")
                e(f"{ind}%{p}kpc = index.add {kpc}, %c0 : index")
            else:
                e(f"{ind}%{p}kpc = index.min %{p}kp, %cap_1 : index")
            e(f"{ind}%{p}kr = index.mul %{p}kpc, %c{128 if K4 else 256} : index")
            e(f"{ind}%{p}ka = index.add %{p}kr, %kdga : index")
            e(f"{ind}%{p}kv = vector.load %k_flat[%{p}ka] : view<[%kq32tot]xi32> -> vector<{nw}xi32>")
            e(f"{ind}%{p}sr = index.mul %{p}kpc, %c{32 if K4 else 8} : index")
            e(f"{ind}%{p}sa = index.add %{p}sr, %kdso : index")
            e(f"{ind}%{p}sv = view.load %ks_flat[%{p}sa] : view<[%kstot]xi32> -> i32")
            return [f"%{p}kv", f"%{p}sv"]
        names = []
        if KCLAMP_S:   # rows past ctx_end are zeroed at staging from the unclamped key, so any in-range row will do
            e(f"{ind}%{p}ksb = index.min {ks}, %klast : index")
        for nn in range(2):
            e(f"{ind}%{p}kp{nn} = index.add {f'%{p}ksb' if KCLAMP_S else ks}, %kk{nn} : index")
            if PAGED and KCLAMP_S:
                kpc = phys_row(f"%{p}kp{nn}", f"%{p}ksb", p, ind, f"k{nn}")
                e(f"{ind}%{p}kpc{nn} = index.add {kpc}, %c0 : index")
            elif PAGED:
                e(f"{ind}%{p}kpc{nn}0 = index.min %{p}kp{nn}, %cap_1 : index")
                kpc = phys_row(f"%{p}kpc{nn}0", ks, p, ind, f"k{nn}")
                e(f"{ind}%{p}kpc{nn} = index.add {kpc}, %c0 : index")
            else:
                e(f"{ind}%{p}kpc{nn} = index.min %{p}kp{nn}, %cap_1 : index")
            e(f"{ind}%{p}kr{nn} = index.mul %{p}kpc{nn}, %c1024 : index")
            e(f"{ind}%{p}kr{nn}b = index.add %{p}kr{nn}, %kvbase : index")
            e(f"{ind}%{p}ka{nn} = index.add %{p}kr{nn}b, %kd{nn} : index")
            e(f"{ind}%{p}kv{nn} = vector.load %k_flat[%{p}ka{nn}] : view<[%kvtot]xf16> -> {V8H}")
            names.append(f"%{p}kv{nn}")
        return names

    def load_v(ks, p, ind):
        if VQ4 or VQ8:   # [kvh][tile][dim]: data 2 (VQ4) or 4 (VQ8) dwords, stats 1 dword
            nw = 2 if VQ4 else 4
            if PAGED:
                e(f"{ind}%{p}vks0 = index.min {ks}, %vlast : index")
                vks = phys_tilestart(f"%{p}vks0", p, ind, "vq")
                e(f"{ind}%{p}vks = index.add {vks}, %c0 : index")
            else:
                e(f"{ind}%{p}vks = index.min {ks}, %vlast : index")
            e(f"{ind}%{p}vt16 = index.div %{p}vks, %c16 : index")
            e(f"{ind}%{p}vti0 = index.add %vhb4, %{p}vt16 : index")
            e(f"{ind}%{p}vti1 = index.mul %{p}vti0, %c256 : index")
            e(f"{ind}%{p}vti = index.add %{p}vti1, %vdim : index")
            e(f"{ind}%{p}vda = index.mul %{p}vti, %c{nw} : index")
            e(f"{ind}%{p}vq = vector.load %v_flat[%{p}vda] : view<[%vq4tot]xi32> -> vector<{nw}xi32>")
            e(f"{ind}%{p}vst = view.load %vs4_flat[%{p}vti] : view<[%vq4tot0]xi32> -> i32")
            return [f"%{p}vq", f"%{p}vst"]
        if PAGED:
            e(f"{ind}%{p}vks0 = index.min {ks}, %vlast : index")
            vks = phys_tilestart(f"%{p}vks0", p, ind, "v")
            e(f"{ind}%{p}vks = index.add {vks}, %c0 : index")
        else:
            e(f"{ind}%{p}vks = index.min {ks}, %vlast : index")
        e(f"{ind}%{p}vtb = index.mul %{p}vks, %c256 : index")
        if VSWZR:
            e(f"{ind}%{p}vln = index.mul %vdim, %c16 : index")
            e(f"{ind}%{p}vhl = index.add %vhb, %{p}vln : index")
            e(f"{ind}%{p}va0 = index.add %{p}vhl, %{p}vtb : index")
        else:
            e(f"{ind}%{p}va0 = index.add %vhl, %{p}vtb : index")
        e(f"{ind}%{p}va1 = index.add %{p}va0, %c8 : index")
        names = []
        for nn in range(2):
            e(f"{ind}%{p}vv{nn} = vector.load %v_flat[%{p}va{nn}] : view<[%vtot]xf16> -> {V8H}")
            names.append(f"%{p}vv{nn}")
        return names

    if NT > 256:
        # wave-uniform role guards from the subgroup id; loads stay unconditional (out-of-role threads load clamped,
        # in-range rows), only the LDS stores are guarded: an scf.if around loads drains vmcnt(0) at its exit
        e("  %sgid = kernel.subgroup.id : index")
        e(f"  %vsg0 = index.constant {(NT - 256) // 32} : index")
        e("  %kstg = index.cmp ult, %sgid, %c8 : index")
        e("  %vstg = index.cmp uge, %sgid, %vsg0 : index")

    def stage_k(ks, cur, p, ind, kv="%k_view"):
        if NT == 256:
            return stage_k_(ks, cur, p, ind, kv)
        e(f"{ind}scf.if %kstg {{")
        stage_k_(ks, cur, p, ind + "  ", kv)
        e(f"{ind}}}")

    def stage_k_(ks, cur, p, ind, kv="%k_view"):
        """K tile at key ks (rows permuted, zero past ctx_end) to LDS."""
        if KDEC:
            nw = 2 if K4 else 4
            shifts = (6, 2, -2, -6) if K4 else (2, -6)
            e(f"{ind}%{p}qk = index.add {ks}, %kdk : index")
            e(f"{ind}%{p}ql = index.cmp ult, %{p}qk, %ctx_end : index")
            e(f"{ind}%{p}s1 = vector.from_elements {cur[1]} : vector<1xi32>")
            e(f"{ind}%{p}sh = vector.bitcast %{p}s1 : vector<1xi32> to vector<2xf16>")
            e(f"{ind}%{p}sS = vector.extract %{p}sh[0] : vector<2xf16> -> f16")
            e(f"{ind}%{p}sC = vector.extract %{p}sh[1] : vector<2xf16> -> f16")
            ws = []
            for d in range(nw):
                e(f"{ind}%{p}w{d} = vector.extract {cur[0]}[{d}] : vector<{nw}xi32> -> i32")
                for k, sh in enumerate(shifts):
                    if DEC_LSHADD:
                        dec_pair(e, ind, f"%{p}g{d}{k}", f"%{p}w{d}", sh, 0x03c003c0 if K4 else 0x03fc03fc, "%kdgm")
                        ws.append(f"%{p}g{d}{k}")
                        continue
                    op = "shli" if sh >= 0 else "shrui"
                    e(f"{ind}%{p}t{d}{k}c = scalar.constant {abs(sh)} : i32")
                    e(f"{ind}%{p}t{d}{k} = scalar.{op} %{p}w{d}, %{p}t{d}{k}c : i32")
                    e(f"{ind}%{p}m{d}{k} = scalar.andi %{p}t{d}{k}, %kdm : i32")
                    e(f"{ind}%{p}g{d}{k} = scalar.ori %{p}m{d}{k}, %kdgm : i32")
                    ws.append(f"%{p}g{d}{k}")
            # ONE 16-wide fma: two 8-wide ones miscompile (see vq4_unpack)
            e(f"{ind}%{p}pk = vector.from_elements {', '.join(ws)} : vector<8xi32>")
            e(f"{ind}%{p}pf = vector.bitcast %{p}pk : vector<8xi32> to {V16H}")
            e(f"{ind}%{p}S16 = vector.splat %{p}sS : {V16H}")
            e(f"{ind}%{p}C16 = vector.splat %{p}sC : {V16H}")
            e(f"{ind}%{p}dk = vector.fmaf %{p}pf, %{p}S16, %{p}C16 : {V16H}")
            e(f"{ind}%{p}dz = scf.select %{p}ql, %{p}dk, %zq16 : {V16H}")
            e(f"{ind}%{p}d0 = vector.slice %{p}dz[0] : {V16H} -> {V8H}")
            e(f"{ind}%{p}d1 = vector.slice %{p}dz[8] : {V16H} -> {V8H}")
            e(f"{ind}vector.store %{p}d0, {kv}[%kdr, %kdc16] : {V8H}, view<{KT}x{KT_PITCH}xf16>")
            e(f"{ind}vector.store %{p}d1, {kv}[%kdr, %kdc16b] : {V8H}, view<{KT}x{KT_PITCH}xf16>")
            return
        for nn in range(2):
            e(f"{ind}%{p}sp{nn} = index.add {ks}, %kk{nn} : index")
            e(f"{ind}%{p}sl{nn} = index.cmp ult, %{p}sp{nn}, %ctx_end : index")
            e(f"{ind}%{p}sv{nn} = scf.select %{p}sl{nn}, {cur[nn]}, %zh8 : {V8H}")
            e(f"{ind}vector.store %{p}sv{nn}, {kv}[%kr{nn}, %kd{nn}] : {V8H}, view<{KT}x{KT_PITCH}xf16>")

    def vq8_unpack(cur, p, ind):
        # bytes (k0, k2, k1, k3) per dword -> f16 1 + u/256 pairs (keys 2k, 2k+1) via (w << 2, w >> 6) & 0x03fc03fc | 0x3c003c00
        # then ONE 16-wide fma f * S + C' (see vq4_unpack)
        e(f"{ind}%{p}vs1 = vector.from_elements {cur[1]} : vector<1xi32>")
        e(f"{ind}%{p}vsv = vector.bitcast %{p}vs1 : vector<1xi32> to vector<2xf16>")
        e(f"{ind}%{p}vss = vector.extract %{p}vsv[0] : vector<2xf16> -> f16")
        e(f"{ind}%{p}vsc = vector.extract %{p}vsv[1] : vector<2xf16> -> f16")
        ws = []
        for d in range(4):
            e(f"{ind}%{p}vw{d}x = vector.extract {cur[0]}[{d}] : vector<4xi32> -> i32")
            for k, sh in enumerate((2, -6)):
                op = "shli" if sh >= 0 else "shrui"
                e(f"{ind}%{p}vsh{d}{k}c = scalar.constant {abs(sh)} : i32")
                e(f"{ind}%{p}vsh{d}{k} = scalar.{op} %{p}vw{d}x, %{p}vsh{d}{k}c : i32")
                e(f"{ind}%{p}vmk{d}{k} = scalar.andi %{p}vsh{d}{k}, %v8m : i32")
                e(f"{ind}%{p}vmg{d}{k} = scalar.ori %{p}vmk{d}{k}, %v4g : i32")
                ws.append(f"%{p}vmg{d}{k}")
        e(f"{ind}%{p}vpk = vector.from_elements {', '.join(ws)} : vector<8xi32>")
        e(f"{ind}%{p}vf = vector.bitcast %{p}vpk : vector<8xi32> to vector<16xf16>")
        e(f"{ind}%{p}vss16 = vector.splat %{p}vss : vector<16xf16>")
        e(f"{ind}%{p}vsc16 = vector.splat %{p}vsc : vector<16xf16>")
        e(f"{ind}%{p}vd = vector.fmaf %{p}vf, %{p}vss16, %{p}vsc16 : vector<16xf16>")
        hv = []
        for d in range(2):
            e(f"{ind}%{p}vd{d} = vector.slice %{p}vd[{8 * d}] : vector<16xf16> -> {V8H}")
            hv.append(f"%{p}vd{d}")
        return hv

    def vq4_unpack(cur, p, ind):
        # nibble pairs (2k, 2k+1) -> f16 1024 + u, then (f - 1031) * s + c
        e(f"{ind}%{p}vs1 = vector.from_elements {cur[1]} : vector<1xi32>")
        e(f"{ind}%{p}vsv = vector.bitcast %{p}vs1 : vector<1xi32> to vector<2xf16>")
        e(f"{ind}%{p}vss = vector.extract %{p}vsv[0] : vector<2xf16> -> f16")
        e(f"{ind}%{p}vsc = vector.extract %{p}vsv[1] : vector<2xf16> -> f16")
        ws = []
        for d in range(2):
            e(f"{ind}%{p}vw{d}x = vector.extract {cur[0]}[{d}] : vector<2xi32> -> i32")
            for k in range(4):
                # nibbles (4k, 16 + 4k) -> bits (6.., 22..): f16 1 + u/16
                sh = 6 - 4 * k
                if DEC_LSHADD:
                    dec_pair(e, ind, f"%{p}vmg{d}{k}", f"%{p}vw{d}x", sh, 0x03c003c0, "%v4g")
                    ws.append(f"%{p}vmg{d}{k}")
                    continue
                op = "shli" if sh >= 0 else "shrui"
                e(f"{ind}%{p}vsh{d}{k}c = scalar.constant {abs(sh)} : i32")
                e(f"{ind}%{p}vsh{d}{k} = scalar.{op} %{p}vw{d}x, %{p}vsh{d}{k}c : i32")
                e(f"{ind}%{p}vmk{d}{k} = scalar.andi %{p}vsh{d}{k}, %v4m : i32")
                e(f"{ind}%{p}vmg{d}{k} = scalar.ori %{p}vmk{d}{k}, %v4g : i32")
                ws.append(f"%{p}vmg{d}{k}")
        # ONE 16-wide fma: with two 8-wide fmas Loom CSEs their identical addend splats
        # and ties both in-place v_pk_fmac to one register, so keys 8..15 get f * S + (the result of keys 0..7).
        e(f"{ind}%{p}vpk = vector.from_elements {', '.join(ws)} : vector<8xi32>")
        e(f"{ind}%{p}vf = vector.bitcast %{p}vpk : vector<8xi32> to vector<16xf16>")
        e(f"{ind}%{p}vss16 = vector.splat %{p}vss : vector<16xf16>")
        e(f"{ind}%{p}vsc16 = vector.splat %{p}vsc : vector<16xf16>")
        e(f"{ind}%{p}vd = vector.fmaf %{p}vf, %{p}vss16, %{p}vsc16 : vector<16xf16>")
        hv = []
        for d in range(2):
            e(f"{ind}%{p}vd{d} = vector.slice %{p}vd[{8 * d}] : vector<16xf16> -> {V8H}")
            hv.append(f"%{p}vd{d}")
        return hv

    def stage_v(cur, vb, p, ind):
        if VQ4:   # unpack outside the guard (an scf.if reading loaded registers drains vmcnt(0) at entry)
            cur = vq4_unpack(cur, p, ind)
        if VQ8:
            cur = vq8_unpack(cur, p, ind)
        if NT == 256:
            return stage_v_(cur, vb, p, ind)
        e(f"{ind}scf.if %vstg {{")
        stage_v_(cur, vb, p, ind + "  ")
        e(f"{ind}}}")

    def stage_v_(cur, vb, p, ind):
        e(f"{ind}%{p}vr = index.add {vb}, %vrow : index")
        for j, v in enumerate(cur):
            e(f"{ind}vector.store {v}, %v_view[%{p}vr, %c{8 * j}] : {V8H}, view<256x{VT_PITCH}xf16>")

    # prologue: K(0) staged
    k0 = load_k("%c0", "p0", "  ")
    if PIPE2:
        k1 = load_k("%c16", "p1", "  ")
        v0 = load_v("%c0", "pv0", "  ")
    stage_k("%c0", k0, "p0s", "  ")
    if SOFT1 and (SOFT2 or SOFT4):
        stage_v_(["%zh8", "%zh8"], "%c0", "pz", "  ")
        e("  %pzlo = index.cmp ult, %tid, %c128 : index")
        e("  scf.if %pzlo {")
        e(f"    %pzrow = index.add %tid, %c{128 if SOFT3 or SOFT4 else 0} : index")
        e(f"    vector.store %zh8, %pp_view[%pzrow, %c0] : vector<8xf16>, view<{PRS}x8xf16>")
        e("  }")
        e("  %pza = index.cmp ult, %tid, %c64 : index")
        e("  scf.if %pza {")
        e(f"    %pzai = index.add %tid, %c{64 if SOFT3 or SOFT4 else 0} : index")
        e(f"    view.store %one, %al_view[%pzai] : f32, view<{PRS2}xf32>")
        e("  }")
    if PAIR:
        stage_v_(["%zh8", "%zh8"], "%c0", "pz", "  ")
        e("  %pzlo = index.cmp ult, %tid, %c128 : index")
        e("  scf.if %pzlo {")
        e("    %pzr = index.add %tid, %c128 : index")
        e("    vector.store %zh8, %pp_view[%pzr, %c0] : vector<8xf16>, view<256x8xf16>")
        e("  }")
        e("  %pza = index.cmp ult, %tid, %c64 : index")
        e("  scf.if %pza {")
        e("    %pzai = index.add %tid, %c64 : index")
        e("    view.store %one, %al_view[%pzai] : f32, view<128xf32>")
        e("  }")
    e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    if PIPE2:
        e(f"  %k2_o = index.constant {K2_OFF} : offset")
        e(f"  %k_view2 = buffer.view %pool[%k2_o] : buffer -> view<{KT}x{KT_PITCH}xf16>")

    onames = [f"%o{f}" for f in range(8)]
    types = ", ".join([V8] * 8 + ["f32", "f32"])

    def emit_qk(kv, pp, ind):
        """S^T partial over this wave's 128 dims: one 8-MMA chain from zero (the accumulation order is fixed)."""
        e(f"{ind}%{pp}zeros8s = vector.fragment<init> %zeros8 shape [%m, %n] : {V8}")
        acc = f"%{pp}zeros8s"
        for c in range(8):
            e(f"{ind}%{pp}kfc{c}c = index.constant {16 * c} : index")
            e(f"{ind}%{pp}kfc{c} = index.add %hd128, %{pp}kfc{c}c : index")
            e(f"{ind}%{pp}kf{c} = vector.fragment.load<lhs> {kv}[%c0, %{pp}kfc{c}] shape [%m, %k] : view<{KT}x{KT_PITCH}xf16> -> {V16H}")
            e(f"{ind}%{pp}sa{c} = vector.mma %{pp}kf{c}, %qf{c}, {acc} : {V16H}, {V16H}, {V8}")
            acc = f"%{pp}sa{c}"
            if c + 1 < 8 and (c + 1) % QKF == 0:
                e(f"{ind}scf.schedule.fence")
        return acc

    def emit_softmax_pv(acc, tail, mid=None, early=None, pair_slot=None):
        """Softmax of this tile (own partial acc + the partner's from LDS) and P.V; returns the O accumulators.
        mid(): emitted right after the rescale branch (the pipelined QK of the next tile and its loads).
        pair_slot (PAIR, wave A): %s0 is the full S; P^T and alpha go to that LDS slot (rows, alpha base); no P.V."""
        if pair_slot is None:
            e(f"    %spa = vector.load %s_view[%ptid, %c0] : view<{2 * NT}x4xf32> -> {V4}")
            e(f"    %spb = vector.load %s_view[%ptid2, %c0] : view<{2 * NT}x4xf32> -> {V4}")
            e(f"    %spart = vector.concat<0> %spa, %spb : {V4}, {V4} -> {V8}")
            e(f"    %s0 = vector.addf {acc}, %spart : {V8}")
        s = "%s0"
        if tail:
            # key 8*half + i visible iff < lim = min(r_abs + 1, ctx_end) - ks - 8*half
            e("    %lim0 = index.add %r_abs, %c1 : index")
            e("    %lim1 = index.min %lim0, %ctx_end : index")
            e("    %h8 = index.mul %half, %c8 : index")
            e("    %kh = index.add %ks_, %h8 : index")
            e("    %limi = index.cast %lim1 : index to i32")
            e("    %khi = index.cast %kh : index to i32")
            e("    %lim = scalar.subi %limi, %khi : i32")
            ind = "    "
            if DIAG_MASK:
                # some lane has a key at or past its limit (lim < 8) or a dead row
                e("    %mc8 = scalar.constant 8 : i32")
                if DIAG_SWAP:   # no lane has a key at or past its limit and no dead row: the scores pass through
                    e("    %mok8 = scalar.cmpi sge, %lim, %mc8 : i32")
                    e("    %mlive = index.cmp ult, %r_lq, %B : index")
                    e("    %mok = scalar.andi %mok8, %mlive : i1")
                    e("    %mallok = kernel.subgroup.vote.all %mok : i1")
                    e(f"    %smask = scf.if %mallok -> ({V8}) {{")
                    e(f"      scf.yield %s0 : {V8}")
                    e("    } else {")
                else:
                    e("    %mpart = scalar.cmpi slt, %lim, %mc8 : i32")
                    e("    %mdead = index.cmp uge, %r_lq, %B : index")
                    e("    %mneed = scalar.ori %mpart, %mdead : i1")
                    e("    %manyneed = kernel.subgroup.vote.any %mneed : i1")
                    e(f"    %smask = scf.if %manyneed -> ({V8}) {{")
                ind = "      "
            els = []
            for i in range(8):
                e(f"{ind}%mi{i} = scalar.constant {i} : i32")
                e(f"{ind}%mv{i}a = scalar.cmpi slt, %mi{i}, %lim : i32")
                e(f"{ind}%mv{i} = scalar.andi %mv{i}a, %r_live_i1 : i1")
                e(f"{ind}%se{i} = vector.extract %s0[{i}] : {V8} -> f32")
                e(f"{ind}%sm{i} = scf.select %mv{i}, %se{i}, %ninf : f32")
                els.append(f"%sm{i}")
            if DIAG_MASK:
                e(f"      %smk = vector.from_elements {', '.join(els)} : {V8}")
                e(f"      scf.yield %smk : {V8}")
                if not DIAG_SWAP:
                    e("    } else {")
                    e(f"      scf.yield %s0 : {V8}")
                e("    }")
            else:
                e(f"    %smask = vector.from_elements {', '.join(els)} : {V8}")
            s = "%smask"
        e(f"    %tmax0 = vector.reduce<maxnumf> {s}, %ninf : {V8}, f32")
        e("    %tmi = scalar.bitcast %tmax0 : f32 to i32")
        e("    %tmx, %tmv = kernel.subgroup.shuffle<xor> %tmi, %x16, %x32 : i32, i32, i32")
        e("    %tmf = scalar.bitcast %tmx : i32 to f32")
        e("    %tmax = scalar.maxnumf %tmax0, %tmf : f32")
        if early:
            early()
        # one wave-uniform branch: raise the max and rescale O / l only when some row needs it (FA4 conditional rescale)
        e("    %grow = scalar.cmpf ogt, %tmax, %rmax : f32")
        e("    %anygrow = kernel.subgroup.vote.any %grow : i1")
        if pair_slot is not None:   # O lives in wave B: only the max and alpha here
            e("    %rsm, %rss = scf.if %anygrow -> (f32, f32) {")
            e("      %gmx = scalar.maxnumf %rmax, %tmax : f32")
            e("      %gpd = scalar.subf %rmax, %gmx : f32")
            e("      %gpdl = scalar.mulf %gpd, %log2e : f32")
            e("      %galpha = scalar.exp2f<afn> %gpdl : f32")
            e("      scf.yield %gmx, %galpha : f32, f32")
            e("    } else {")
            e("      scf.yield %rmax, %one : f32, f32")
            e("    }")
        else:
            otypes = ", ".join([V8] * 8 + ["f32", "f32"])
            e(f"    %rso0, %rso1, %rso2, %rso3, %rso4, %rso5, %rso6, %rso7, %rsm, %rss = scf.if %anygrow -> ({otypes}) {{")
            e("      %gmx = scalar.maxnumf %rmax, %tmax : f32")
            e("      %gpd = scalar.subf %rmax, %gmx : f32")
            e("      %gpdl = scalar.mulf %gpd, %log2e : f32")
            e("      %galpha = scalar.exp2f<afn> %gpdl : f32")
            e(f"      %galpha8 = vector.splat %galpha : {V8}")
            for f in range(8):
                e(f"      %gos{f} = vector.mulf %o{f}, %galpha8 : {V8}")
            # yields alpha; 1.0 when skipped: O * 1.0 and fma(sum, 1.0, part) are exact, so the skip is bit-identical
            e(f"      scf.yield {', '.join(f'%gos{f}' for f in range(8))}, %gmx, %galpha : {otypes}")
            e("    } else {")
            e(f"      scf.yield {', '.join(f'%o{f}' for f in range(8))}, %rmax, %one : {otypes}")
            e("    }")
        if mid:
            mid()
        e("    %nmax = scalar.maxnumf %rsm, %ninf : f32")
        e("    %nml = scalar.mulf %nmax, %log2e : f32")
        e("    %nnml = scalar.negf %nml : f32")
        e("    %pd = scalar.subf %rmax, %nmax : f32")
        e("    %pdl = scalar.mulf %pd, %log2e : f32")
        e("    %alpha = scalar.exp2f<afn> %pdl : f32")
        e(f"    %nnml8 = vector.splat %nnml : {V8}")
        e(f"    %nmax8 = vector.splat %nmax : {V8}")
        e(f"    %pd8 = vector.subf {s}, %nmax8 : {V8}")
        e(f"    %pl = vector.mulf %pd8, %log2e8 : {V8}")
        e(f"    %p = vector.exp2f<afn> %pl : {V8}")
        # keys 8h..8h+3 and 8h+4..8h+7: HIP's 4-lane segments
        for g in range(2):
            cur = "%zero"
            for i in range(4):
                e(f"    %pg{g}_{i} = vector.extract %p[{4 * g + i}] : {V8} -> f32")
                e(f"    %pa{g}_{i} = scalar.addf {cur}, %pg{g}_{i} : f32")
                cur = f"%pa{g}_{i}"
        e("    %pab = scalar.addf %pa0_3, %pa1_3 : f32")
        e("    %pabi = scalar.bitcast %pab : f32 to i32")
        e("    %pcdi, %pcdv = kernel.subgroup.shuffle<xor> %pabi, %x16, %x32 : i32, i32, i32")
        e("    %pcd = scalar.bitcast %pcdi : i32 to f32")
        e("    %psum = scalar.addf %pab, %pcd : f32")
        e("    %nsum = scalar.fmaf %rsum, %rss, %psum : f32")
        # P^T as the B operand: keys 0..7 from the h=0 lane, 8..15 from h=1.
        # f16(p) as fptrunc(fma(p, 1, 0)) with an opaque 1 and 0, which selects v_fma_mix (as Q4FMIX in the GEMMs).
        # v_cvt_f16_f32 results must sit in v0..v127, where Q^T and O live: each conversion evicts a Q^T fragment to scratch.
        for i in range(8):
            e(f"    %pe{i} = vector.extract %p[{i}] : {V8} -> f32")
            e(f"    %pm{i} = scalar.fmaf %pe{i}, %one_o, %zero_o : f32")
            e(f"    %pt{i} = scalar.fptrunc %pm{i} : f32 to f16")
        e(f"    %ph = vector.from_elements {', '.join(f'%pt{i}' for i in range(8))} : {V8H}")
        if pair_slot is not None:
            rows, alb = pair_slot
            e(f"    %pwr = index.add {rows}, %prow_own : index")
            prows = PRS if SOFT1 else 256
            e(f"    vector.store %ph, %pp_view[%pwr, %c0] : {V8H}, view<{prows}x8xf16>")
            e(f"    %pwa = index.add {alb}, %pa_own : index")
            e(f"    view.store %rss, %al_view[%pwa] : f32, view<{PRS2 if SOFT1 else 128}xf32>")
            return None
        e(f"    %phi = vector.bitcast %ph : {V8H} to {V4I}")
        e(f"    %ppi, %ppv = kernel.subgroup.shuffle<xor> %phi, %x16, %x32 : {V4I}, i32, i32")
        e(f"    %pph = vector.bitcast %ppi : {V4I} to {V8H}")
        e(f"    %plo = scf.select %h0, %ph, %pph : {V8H}")
        e(f"    %phi2 = scf.select %h0, %pph, %ph : {V8H}")
        e(f"    %pb0 = vector.concat<0> %plo, %phi2 : {V8H}, {V8H} -> {V16H}")
        e(f"    %pb = vector.fragment<rhs> %pb0 shape [%k, %n] : {V16H}")
        # P.V into the (conditionally) rescaled O
        e(f"    %alpha8 = vector.splat %alpha : {V8}")
        outs = []
        for f in range(8):
            e(f"    %vfr{f}c = index.constant {16 * f} : index")
            e(f"    %vfr{f}a = index.add %hd128, %vfr{f}c : index")
            e(f"    %vfr{f} = index.add %vcur, %vfr{f}a : index")
            e(f"    %vf{f} = vector.fragment.load<lhs> %v_view[%vfr{f}, %c0] shape [%m, %k] : view<256x{VT_PITCH}xf16> -> {V16H}")
            e(f"    %nx{f} = vector.mma %vf{f}, %pb, %rso{f} : {V16H}, {V16H}, {V8}")
            outs.append(f"%nx{f}")
        return outs


    NFR = 16 if PAIR else 8
    if PAIR or SOFT1:
        e("  %pw32 = index.mul %wqb, %c32 : index")
        e("  %pw16 = index.mul %wqb, %c16 : index")
        e("  %prow_own = index.add %pw32, %lane : index")     # A: this lane's P^T row
        e("  %prow_lo = index.add %pw32, %sub : index")       # B: the h0 lane's row ...
        e("  %prow_hi = index.add %prow_lo, %c16 : index")    # ... and the h1 lane's
        e("  %pa_own = index.add %pw16, %sub : index")
        e("  %c128p = index.constant 128 : index")
    if PAIR:
        rnames = [f"%rb{f}" for f in range(16)]
        BK = "vector<8xi32>" if PAIR_I32 else V8
        rtypes = ", ".join([BK] * 16)

    def pair_pv(slot_rows, slot_al, R, pp, ind):
        """B: rescale the O bank by A's alpha of that tile (only when some row's max grew, as the A-side branch did),
        P^T from LDS, 16 P.V MMAs. Returns the new bank names."""
        e(f"{ind}%{pp}ai = index.add {slot_al}, %pa_own : index")
        e(f"{ind}%{pp}al = view.load %al_view[%{pp}ai] : view<128xf32> -> f32")
        # pass-through arm first (as DIAG_SWAP): the bank's last use is then the multiply, which can overwrite it in place
        e(f"{ind}%{pp}gr = scalar.cmpf oeq, %{pp}al, %one : f32")
        e(f"{ind}%{pp}ag = kernel.subgroup.vote.all %{pp}gr : i1")
        if PAIR_I32:
            Rf = []
            for f in range(16):
                e(f"{ind}%{pp}rf{f} = vector.bitcast {R[f]} : {BK} to {V8}")
                Rf.append(f"%{pp}rf{f}")
            R = Rf
        ftypes = ", ".join([V8] * 16)
        sc = [f"%{pp}sc{f}" for f in range(16)]
        if True:
          e(f"{ind}{', '.join(sc)} = scf.if %{pp}ag -> ({ftypes}) {{")
          e(f"{ind}  scf.yield {', '.join(R)} : {ftypes}")
          e(f"{ind}}} else {{")
          e(f"{ind}  %{pp}al8 = vector.splat %{pp}al : {V8}")
          for f in range(16):
              e(f"{ind}  %{pp}os{f} = vector.mulf {R[f]}, %{pp}al8 : {V8}")
          e(f"{ind}  scf.yield {', '.join(f'%{pp}os{f}' for f in range(16))} : {ftypes}")
          e(f"{ind}}}")
        e(f"{ind}%{pp}rl = index.add {slot_rows}, %prow_lo : index")
        e(f"{ind}%{pp}rh = index.add {slot_rows}, %prow_hi : index")
        e(f"{ind}%{pp}plo = vector.load %pp_view[%{pp}rl, %c0] : view<256x8xf16> -> {V8H}")
        e(f"{ind}%{pp}phi = vector.load %pp_view[%{pp}rh, %c0] : view<256x8xf16> -> {V8H}")
        e(f"{ind}%{pp}pb0 = vector.concat<0> %{pp}plo, %{pp}phi : {V8H}, {V8H} -> {V16H}")
        e(f"{ind}%{pp}pb = vector.fragment<rhs> %{pp}pb0 shape [%k, %n] : {V16H}")
        outs = []
        for f in range(16):
            e(f"{ind}%{pp}vr{f} = index.constant {16 * f} : index")
            e(f"{ind}%{pp}vf{f} = vector.fragment.load<lhs> %v_view[%{pp}vr{f}, %c0] shape [%m, %k] : view<256x{VT_PITCH}xf16> -> {V16H}")
            e(f"{ind}%{pp}oi{f} = vector.fragment<init> {sc[f]} shape [%m, %n] : {V8}")
            e(f"{ind}%{pp}nx{f} = vector.mma %{pp}vf{f}, %{pp}pb, %{pp}oi{f} : {V16H}, {V16H}, {V8}")
            if PAIR_I32:
                e(f"{ind}%{pp}nxi{f} = vector.bitcast %{pp}nx{f} : {V8} to {BK}")
                outs.append(f"%{pp}nxi{f}")
            else:
                outs.append(f"%{pp}nx{f}")
            if f < 15 and (f + 1) % PVF == 0:
                e(f"{ind}scf.schedule.fence")
        return outs

    def pair_a(R, ind):
        """A: S^T = chain(dims 0..127) + chain(128..255), masked softmax; P^T (f16) and alpha of this tile to LDS slot."""
        for c in range(16):
            e(f"{ind}%qb16_{c} = vector.bitcast {R[c]} : {BK} to {V16H}")
            e(f"{ind}%qf{c} = vector.fragment<rhs> %qb16_{c} shape [%k, %n] : {V16H}")
        chains = []
        for h in range(2):
            e(f"{ind}%sz{h} = vector.fragment<init> %zeros8 shape [%m, %n] : {V8}")
            acc = f"%sz{h}"
            for c in range(8 * h, 8 * h + 8):
                e(f"{ind}%kfc{c} = index.constant {16 * c} : index")
                e(f"{ind}%kf{c} = vector.fragment.load<lhs> %k_view[%c0, %kfc{c}] shape [%m, %k] : view<{KT}x{KT_PITCH}xf16> -> {V16H}")
                e(f"{ind}%sa{c} = vector.mma %kf{c}, %qf{c}, {acc} : {V16H}, {V16H}, {V8}")
                acc = f"%sa{c}"
                if c % 8 != 7 and (c + 1) % QKF == 0:
                    e(f"{ind}scf.schedule.fence")
            chains.append(acc)
        e(f"{ind}%s0 = vector.addf {chains[0]}, {chains[1]} : {V8}")


    def soft1_pv(ind, onm=None, px="s1", rows=None, alb=None):
        """SOFT1, both waves: rescale the O half by A's alpha (pass-through first), P^T from LDS, 8 P.V MMAs."""
        onm = onm or onames
        if alb is not None:
            e(f"{ind}%{px}ai = index.add {alb}, %pa_own : index")
        e(f"{ind}%{px}al = view.load %al_view[{f'%{px}ai' if alb is not None else '%pa_own'}] : view<{PRS2}xf32> -> f32")
        sc = [f"%{px}sc{f}" for f in range(8)]
        otypes8 = ", ".join([V8] * 8)

        def arm_pass():
            e(f"{ind}  scf.yield {', '.join(onm)} : {otypes8}")

        def arm_mul():
            e(f"{ind}  %{px}al8 = vector.splat %{px}al : {V8}")
            for f in range(8):
                e(f"{ind}  %{px}os{f} = vector.mulf {onm[f]}, %{px}al8 : {V8}")
            e(f"{ind}  scf.yield {', '.join(f'%{px}os{f}' for f in range(8))} : {otypes8}")
        if S1_PASS_FIRST:
            e(f"{ind}%{px}gr = scalar.cmpf oeq, %{px}al, %one : f32")
            e(f"{ind}%{px}ag = kernel.subgroup.vote.all %{px}gr : i1")
        else:
            e(f"{ind}%{px}gr = scalar.cmpf une, %{px}al, %one : f32")
            e(f"{ind}%{px}ag = kernel.subgroup.vote.any %{px}gr : i1")
        e(f"{ind}{', '.join(sc)} = scf.if %{px}ag -> ({otypes8}) {{")
        (arm_pass if S1_PASS_FIRST else arm_mul)()
        e(f"{ind}}} else {{")
        (arm_mul if S1_PASS_FIRST else arm_pass)()
        e(f"{ind}}}")
        rl, rh = "%prow_lo", "%prow_hi"
        if rows is not None:
            e(f"{ind}%{px}rl = index.add {rows}, %prow_lo : index")
            e(f"{ind}%{px}rh = index.add {rows}, %prow_hi : index")
            rl, rh = f"%{px}rl", f"%{px}rh"
        e(f"{ind}%{px}plo = vector.load %pp_view[{rl}, %c0] : view<{PRS}x8xf16> -> {V8H}")
        e(f"{ind}%{px}phi = vector.load %pp_view[{rh}, %c0] : view<{PRS}x8xf16> -> {V8H}")
        e(f"{ind}%{px}pb0 = vector.concat<0> %{px}plo, %{px}phi : {V8H}, {V8H} -> {V16H}")
        e(f"{ind}%{px}pb = vector.fragment<rhs> %{px}pb0 shape [%k, %n] : {V16H}")
        outs = []
        for f in range(8):
            e(f"{ind}%{px}vc{f} = index.constant {16 * f} : index")
            e(f"{ind}%{px}vr{f} = index.add %hd128, %{px}vc{f} : index")
            e(f"{ind}%{px}vf{f} = vector.fragment.load<lhs> %v_view[%{px}vr{f}, %c0] shape [%m, %k] : view<256x{VT_PITCH}xf16> -> {V16H}")
            e(f"{ind}%{px}nx{f} = vector.mma %{px}vf{f}, %{px}pb, {sc[f]} : {V16H}, {V16H}, {V8}")
            outs.append(f"%{px}nx{f}")
        return outs

    def body_soft1(tail):
        I = "    "
        e(f"{I}%vcur = index.add %c0, %c0 : index")
        e(f"{I}%ks16 = index.add %ks_, %c16 : index")
        nk = load_k("%ks16", "nk", I)
        nv = load_v("%ks_", "nv", I)
        acc = emit_qk("%k_view", "", I)
        e(f"{I}scf.if %isBs {{")
        e(f"{I}  %sst0 = vector.slice {acc}[0] : {V8} -> {V4}")
        e(f"{I}  %sst1 = vector.slice {acc}[4] : {V8} -> {V4}")
        e(f"{I}  vector.store %sst0, %s_view[%tid, %c0] : {V4}, view<{2 * NT}x4xf32>")
        e(f"{I}  vector.store %sst1, %s_view[%tid2, %c0] : {V4}, view<{2 * NT}x4xf32>")
        e(f"{I}}}")
        stage_v(nv, "%c0", "stv", I)
        e(f"{I}kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        stage_k("%ks16", nk, "st", I)
        e(f"{I}%qmax, %qsum = scf.if %isA -> (f32, f32) {{")
        e(f"{I}  %spa = vector.load %s_view[%ptid, %c0] : view<{2 * NT}x4xf32> -> {V4}")
        e(f"{I}  %spb = vector.load %s_view[%ptid2, %c0] : view<{2 * NT}x4xf32> -> {V4}")
        e(f"{I}  %spart = vector.concat<0> %spa, %spb : {V4}, {V4} -> {V8}")
        e(f"{I}  %s0 = vector.addf {acc}, %spart : {V8}")
        emit_softmax_pv(None, tail, pair_slot=("%c0", "%c0"))
        e(f"{I}  scf.yield %nmax, %nsum : f32, f32")
        e(f"{I}}} else {{")
        e(f"{I}  scf.yield %rmax, %rsum : f32, f32")
        e(f"{I}}}")
        e(f"{I}kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        outs = soft1_pv(I)
        e(f"{I}kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        return outs


    def body_soft2(tail):
        I = "    "
        e(f"{I}%vcur = index.add %c0, %c0 : index")
        e(f"{I}%ks16 = index.add %ks_, %c16 : index")
        nk = load_k("%ks16", "nk", I)
        nv = load_v("%ks_", "nv", I)
        acc = emit_qk("%k_view", "", I)
        e(f"{I}scf.if %isBs {{")
        e(f"{I}  %sst0 = vector.slice {acc}[0] : {V8} -> {V4}")
        e(f"{I}  %sst1 = vector.slice {acc}[4] : {V8} -> {V4}")
        e(f"{I}  vector.store %sst0, %s_view[%tid, %c0] : {V4}, view<{2 * NT}x4xf32>")
        e(f"{I}  vector.store %sst1, %s_view[%tid2, %c0] : {V4}, view<{2 * NT}x4xf32>")
        e(f"{I}}}")
        otypes8 = ", ".join([V8] * 8)
        cur_slot = prev_slot = (None, None)
        if SOFT3:   # P^T / alpha slot: tile parity (A writes this tile's, P.V reads the previous tile's)
            e(f"{I}%s3t = index.div %ks_, %c16 : index")
            e(f"{I}%s3p = index.rem %s3t, %c2 : index")
            e(f"{I}%s3q = index.sub %c1, %s3p : index")
            e(f"{I}%s3rows = index.mul %s3p, %c128 : index")
            e(f"{I}%s3al = index.mul %s3p, %c64 : index")
            e(f"{I}%s3prows = index.mul %s3q, %c128 : index")
            e(f"{I}%s3pal = index.mul %s3q, %c64 : index")
            cur_slot, prev_slot = ("%s3rows", "%s3al"), ("%s3prows", "%s3pal")
        if SOFT3:   # phase 1: only A's P.V(j - 1)
            ph1 = [f"%p1o{f}" for f in range(8)]
            # pass-through arm first: O's last use is then the P.V, which accumulates in place
            e(f"{I}{', '.join(ph1)} = scf.if %isBs -> ({otypes8}) {{")
            e(f"{I}  scf.yield {', '.join(onames)} : {otypes8}")
            e(f"{I}}} else {{")
            o1 = soft1_pv(I + "  ", onames, "pa", *prev_slot)
            e(f"{I}  scf.yield {', '.join(o1)} : {otypes8}")
            e(f"{I}}}")
            outs = ph1
        else:
            outs = soft1_pv(I)            # P.V of the previous tile (zero P / V before the first)
        e(f"{I}kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        stage_v_late = SOFT3   # under SOFT3, B still reads V(j - 1) in phase 2: stage V(j) after the barrier below
        if not stage_v_late:
            stage_v(nv, "%c0", "stv", I)
        stage_k("%ks16", nk, "st", I)
        e(f"{I}%qmax, %qsum = scf.if %isA -> (f32, f32) {{")
        e(f"{I}  %spa = vector.load %s_view[%ptid, %c0] : view<{2 * NT}x4xf32> -> {V4}")
        e(f"{I}  %spb = vector.load %s_view[%ptid2, %c0] : view<{2 * NT}x4xf32> -> {V4}")
        e(f"{I}  %spart = vector.concat<0> %spa, %spb : {V4}, {V4} -> {V8}")
        e(f"{I}  %s0 = vector.addf {acc}, %spart : {V8}")
        emit_softmax_pv(None, tail, pair_slot=cur_slot if SOFT3 else ("%c0", "%c0"))
        e(f"{I}  scf.yield %nmax, %nsum : f32, f32")
        e(f"{I}}} else {{")
        e(f"{I}  scf.yield %rmax, %rsum : f32, f32")
        e(f"{I}}}")
        if SOFT3:   # phase 2: B's P.V(j - 1) (P^T(j - 1) / alpha(j - 1) are only rewritten after this barrier pair)
            ph2 = [f"%p2o{f}" for f in range(8)]
            e(f"{I}{', '.join(ph2)} = scf.if %isA -> ({otypes8}) {{")
            e(f"{I}  scf.yield {', '.join(outs)} : {otypes8}")
            e(f"{I}}} else {{")
            o2 = soft1_pv(I + "  ", outs, "pb", *prev_slot)
            e(f"{I}  scf.yield {', '.join(o2)} : {otypes8}")
            e(f"{I}}}")
            outs = ph2
        e(f"{I}kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        if SOFT3:   # V(j) after everyone's P.V(j - 1); a third barrier before the next tile reads it
            stage_v(nv, "%c0", "stv", I)
            e(f"{I}kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        return outs


    def body_soft4(tail):
        I = "    "
        e(f"{I}%vcur = index.add %c0, %c0 : index")
        e(f"{I}%ks16 = index.add %ks_, %c16 : index")
        e(f"{I}%ksm0 = index.max %ks_, %c16 : index")
        e(f"{I}%ksm = index.sub %ksm0, %c16 : index")      # V(j - 1); tile 0 reloads tile 0 (finite, times a zero P)
        nk = load_k("%ks16", "nk", I)
        nv = load_v("%ksm", "nv", I)
        e(f"{I}%s3t = index.div %ks_, %c16 : index")
        e(f"{I}%s3p = index.rem %s3t, %c2 : index")
        e(f"{I}%s3q = index.sub %c1, %s3p : index")
        e(f"{I}%s3rows = index.mul %s3p, %c128 : index")
        e(f"{I}%s3al = index.mul %s3p, %c64 : index")
        e(f"{I}%s3prows = index.mul %s3q, %c128 : index")
        e(f"{I}%s3pal = index.mul %s3q, %c64 : index")
        acc = emit_qk("%k_view", "", I)
        e(f"{I}scf.if %isBs {{")
        e(f"{I}  %sst0 = vector.slice {acc}[0] : {V8} -> {V4}")
        e(f"{I}  %sst1 = vector.slice {acc}[4] : {V8} -> {V4}")
        e(f"{I}  vector.store %sst0, %s_view[%tid, %c0] : {V4}, view<{2 * NT}x4xf32>")
        e(f"{I}  vector.store %sst1, %s_view[%tid2, %c0] : {V4}, view<{2 * NT}x4xf32>")
        e(f"{I}}}")
        stage_v(nv, "%c0", "stv", I)
        e(f"{I}kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        stage_k("%ks16", nk, "st", I)
        outs = soft1_pv(I, onames, "s4", "%s3prows", "%s3pal")
        e(f"{I}%qmax, %qsum = scf.if %isA -> (f32, f32) {{")
        e(f"{I}  %spa = vector.load %s_view[%ptid, %c0] : view<{2 * NT}x4xf32> -> {V4}")
        e(f"{I}  %spb = vector.load %s_view[%ptid2, %c0] : view<{2 * NT}x4xf32> -> {V4}")
        e(f"{I}  %spart = vector.concat<0> %spa, %spb : {V4}, {V4} -> {V8}")
        e(f"{I}  %s0 = vector.addf {acc}, %spart : {V8}")
        emit_softmax_pv(None, tail, pair_slot=("%s3rows", "%s3al"))
        e(f"{I}  scf.yield %nmax, %nsum : f32, f32")
        e(f"{I}}} else {{")
        e(f"{I}  scf.yield %rmax, %rsum : f32, f32")
        e(f"{I}}}")
        e(f"{I}kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        return outs

    def store_s(acc, pp, ind):
        e(f"{ind}%{pp}sst0 = vector.slice {acc}[0] : {V8} -> {V4}")
        e(f"{ind}%{pp}sst1 = vector.slice {acc}[4] : {V8} -> {V4}")
        e(f"{ind}vector.store %{pp}sst0, %s_view[%tid, %c0] : {V4}, view<{2 * NT}x4xf32>")
        e(f"{ind}vector.store %{pp}sst1, %s_view[%tid2, %c0] : {V4}, view<{2 * NT}x4xf32>")

    def body2(tail):
        """PIPE2 tile ks: region 1 (softmax(ks) + QK(ks+16) + P.V(ks)), barrier, region 2 (stores), barrier."""
        I = "    "
        e(f"{I}%vcur = index.add %c0, %c0 : index")
        e(f"{I}%ks16 = index.add %ks_, %c16 : index")
        e(f"{I}%ks32 = index.add %ks_, %c32 : index")
        # tile t = ks / 16 sits in K buffer t % 2: QK(t+1) reads buffer (t+1) % 2, K(t+2) is staged into t % 2
        e(f"{I}%kt16 = index.div %ks_, %c16 : index")
        e(f"{I}%kpar = index.rem %kt16, %c2 : index")
        e(f"{I}%kpe = index.cmp eq, %kpar, %c0 : index")
        e(f"{I}%kqo0 = index.constant {K_OFF} : index")
        e(f"{I}%kqo1 = index.constant {K2_OFF} : index")
        e(f"{I}%kqoi = scf.select %kpe, %kqo1, %kqo0 : index")
        e(f"{I}%ksoi = scf.select %kpe, %kqo0, %kqo1 : index")
        e(f"{I}%kqo = index.cast %kqoi : index to offset")
        e(f"{I}%kso = index.cast %ksoi : index to offset")
        e(f"{I}%kq_view = buffer.view %pool[%kqo] : buffer -> view<{KT}x{KT_PITCH}xf16>")
        e(f"{I}%ks_view = buffer.view %pool[%kso] : buffer -> view<{KT}x{KT_PITCH}xf16>")
        got = {}

        def qk_next():
            got["s"] = emit_qk("%kq_view", "nq", I)

        def mid():
            if not PIPE2_EARLY:
                qk_next()
            got["v"] = load_v("%ks16", "nv", I)
            got["k"] = load_k("%ks32", "nk", I)
        outs = emit_softmax_pv("%sown", tail, mid, qk_next if PIPE2_EARLY else None)
        e(f"{I}kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        store_s(got["s"], "ns", I)
        stage_v(got["v"], "%c0", "stv", I)
        stage_k("%ks32", got["k"], "st", I, kv="%ks_view")
        e(f"{I}kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        return outs, got["s"]

    def body(tail):
        e("    %vcur = index.add %c0, %c0 : index")
        e("    %ks16 = index.add %ks_, %c16 : index")
        nk = load_k("%ks16", "nk", "    ")
        nv = load_v("%ks_", "nv", "    ")
        acc = emit_qk("%k_view", "", "    ")
        e(f"    %sst0 = vector.slice {acc}[0] : {V8} -> {V4}")
        e(f"    %sst1 = vector.slice {acc}[4] : {V8} -> {V4}")
        e(f"    vector.store %sst0, %s_view[%tid, %c0] : {V4}, view<{2 * NT}x4xf32>")
        e(f"    vector.store %sst1, %s_view[%tid2, %c0] : {V4}, view<{2 * NT}x4xf32>")
        # V(ks): the previous tile's P.V finished at the last barrier
        if VQ8 or VQ4:
            # keep the unpack after QK: hoisted between the QK MMAs, it waits for the V load ~4 MMAs after issue
            e("    scf.schedule.fence")
        stage_v(nv, "%c0", "stv", "    ")
        e("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        # ---- B: K(ks+16) (QK of this tile is done) to LDS
        stage_k("%ks16", nk, "st", "    ")
        outs = emit_softmax_pv(acc, tail)
        e("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        return outs


    def pair_tile_head(I):
        e(f"{I}%ks16 = index.add %ks_, %c16 : index")
        if PAIR_LATE:
            nk = nv = None
        else:
            nk = load_k("%ks16", "nk", I)
            nv = load_v("%ks_", "nv", I)
        e(f"{I}%kt16 = index.div %ks_, %c16 : index")
        e(f"{I}%kpar = index.rem %kt16, %c2 : index")
        e(f"{I}%kpar1 = index.sub %c1, %kpar : index")
        return nk, nv

    def pair_tile_tail(nk, nv, I):
        e(f"{I}kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        if nk is None:
            nk = load_k("%ks16", "nk", I)
            nv = load_v("%ks_", "nv", I)
        stage_v(nv, "%c0", "stv", I)
        stage_k("%ks16", nk, "st", I)
        e(f"{I}kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")

    def body_pair():
        """PAIR tile ks. Phase 1, A: QK(ks) + softmax -> P^T / alpha slot ks % 2; B: P.V(ks - 16) from the other slot
        and the V tile (zero-initialized for ks = 0). Phase 2: stage V(ks), K(ks + 16)."""
        I = "    "
        nk, nv = pair_tile_head(I)
        e(f"{I}%slot_rows = index.mul %kpar, %c128p : index")
        e(f"{I}%slot_al = index.mul %kpar, %c64 : index")
        e(f"{I}%prev_rows = index.mul %kpar1, %c128p : index")
        e(f"{I}%prev_al = index.mul %kpar1, %c64 : index")
        nb = [f"%nb{f}" for f in range(16)]
        def arm_a():
            pair_a(rnames, I + "  ")
            emit_softmax_pv(None, True, pair_slot=("%slot_rows", "%slot_al"))
            e(f"{I}  scf.yield {', '.join(rnames)}, %nmax, %nsum : {rtypes}, f32, f32")

        def arm_b():
            outs = pair_pv("%prev_rows", "%prev_al", rnames, "bp", I + "  ")
            e(f"{I}  scf.yield {', '.join(outs)}, %rmax, %rsum : {rtypes}, f32, f32")
        e(f"{I}%isBr = index.cmp ne, %prole, %c0 : index")
        e(f"{I}{', '.join(nb)}, %qmax, %qsum = scf.if {'%isBr' if PAIR_BFIRST else '%isA'} -> ({rtypes}, f32, f32) {{")
        (arm_b if PAIR_BFIRST else arm_a)()
        e(f"{I}}} else {{")
        (arm_a if PAIR_BFIRST else arm_b)()
        e(f"{I}}}")
        pair_tile_tail(nk, nv, I)
        return nb

    e("  %r_live_i1 = index.cmp ult, %r_lq, %B : index")
    e(f"  %oinit = vector.fragment<init> %zeros8 shape [%m, %n] : {V8}")
    e("  %spq = index.add %start_pos, %qs : index")
    e("  %spq1 = index.add %spq, %c1 : index")
    e("  %split0 = index.div %spq1, %c16 : index")
    e("  %split1 = index.mul %split0, %c16 : index")
    # One masked loop over every tile (split = 0, so the unmasked loop runs no iteration).
    # Two loops (unmasked up to the diagonal, then masked) corrupt the O accumulators across the hand-off (Loom).
    e("  %split = index.min %c0, %max_vis : index")

    def loop(lo, hi, init_o, init_m, init_s, tail, res):
        init = ", ".join(f"{nm} = {v} : {V8}" for nm, v in zip(onames, init_o))
        init += f", %rmax = {init_m} : f32, %rsum = {init_s} : f32"
        if DOWHILE_MAIN:
            e(f"  %wkf, {', '.join(res)} = scf.while(%ksw = {lo} : index, {init}) -> (index, {types}) {{")
            e("    %ks_ = index.assume %ksw [range(%ksw, 0, 1048576), lt(%ksw, %max_vis)] : index")
            if SOFT1:
                outs = (body_soft4 if SOFT4 else body_soft2 if SOFT2 else body_soft1)(tail)
                nm_, ns_ = "%qmax", "%qsum"
            else:
                outs = body(tail)
                nm_, ns_ = "%nmax", "%nsum"
            e(f"    %ksn = index.add %ks_, %c{KT} : index")
            e(f"    %kcont = index.cmp ult, %ksn, {hi} : index")
            e(f"    scf.condition %kcont, %ksn, {', '.join(outs)}, {nm_}, {ns_} : i1, index, {types}")
            wargs = ", ".join(["%wk: index"] + [f"%wo{f}: {V8}" for f in range(8)] + ["%wm: f32", "%ws: f32"])
            e(f"  }} do({wargs}) {{")
            e(f"    scf.yield %wk, {', '.join(f'%wo{f}' for f in range(8))}, %wm, %ws : index, {types}")
            e("  }")
            return
        e(f"  {', '.join(res)} = scf.for %ks_ = [{lo} to {hi} step %c{KT}]({init}) -> ({types})  {{")
        if SOFT1:
            outs = (body_soft4 if SOFT4 else body_soft2 if SOFT2 else body_soft1)(tail)
            e(f"    scf.yield {', '.join(outs)}, %qmax, %qsum : {types}")
        else:
            outs = body(tail)
            e(f"    scf.yield {', '.join(outs)}, %nmax, %nsum : {types}")
        e("  }")

    if PAIR:
        init = ", ".join(f"%rb{f} = %rinit{f} : {BK}" for f in range(16)) + ", %rmax = %ninf : f32, %rsum = %zero : f32"
        fb = [f"%fb{f}" for f in range(16)]
        if DOWHILE:
            e(f"  %wkf, {', '.join(fb)}, %gmax, %gsum = scf.while(%ksw = %c0 : index, {init}) -> (index, {rtypes}, f32, f32) {{")
            # the while carry has no induction range: restate it (ks < max_vis <= ctx_end, a tile start)
            e("    %ks_ = index.assume %ksw [range(%ksw, 0, 1048576), lt(%ksw, %max_vis)] : index")
            nb = body_pair()
            e(f"    %ksn = index.add %ks_, %c{KT} : index")
            e("    %kcont = index.cmp ult, %ksn, %max_vis : index")
            e(f"    scf.condition %kcont, %ksn, {', '.join(nb)}, %qmax, %qsum : i1, index, {rtypes}, f32, f32")
            wargs = ", ".join([f"%wk: index"] + [f"%wb{f}: {BK}" for f in range(16)] + ["%wm: f32", "%ws: f32"])
            e(f"  }} do({wargs}) {{")
            e(f"    scf.yield %wk, {', '.join(f'%wb{f}' for f in range(16))}, %wm, %ws : index, {rtypes}, f32, f32")
            e("  }")
        else:
            e(f"  {', '.join(fb)}, %gmax, %gsum = scf.for %ks_ = [%c0 to %max_vis step %c{KT}]({init}) -> ({rtypes}, f32, f32)  {{")
            nb = body_pair()
            e(f"    scf.yield {', '.join(nb)}, %qmax, %qsum : {rtypes}, f32, f32")
            e("  }")
        e("  scf.if %isA {")
        e("    view.store %gsum, %ls_view[%pa_own] : f32, view<64xf32>")
        e("  }")
        e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        e("  %ltl0 = index.add %max_vis, %c15 : index")
        e("  %ltl1 = index.div %ltl0, %c16 : index")
        e("  %ltl = index.sub %ltl1, %c1 : index")
        e("  %lpar = index.rem %ltl, %c2 : index")
        e("  %lrows = index.mul %lpar, %c128p : index")
        e("  %lal = index.mul %lpar, %c64 : index")
        # every wave (A's result is never stored): no merge, so the bank is not duplicated across arms
        outs = pair_pv("%lrows", "%lal", fb, "fp", "  ")
        for f in range(16):
            e(f"  %og{f} = vector.bitcast {outs[f]} : {BK} to {V8}")
        e("  %lsum_l = view.load %ls_view[%pa_own] : view<64xf32> -> f32")
        e("  %isB = index.cmp ne, %prole, %c0 : index")
        e("  %ewr = scalar.andi %r_live, %isB : i1")
    elif PIPE2:
        # prologue part 2: QK(0) -> own S partial (carried), store it, stage V(0) and K(1) (second buffer)
        s0 = emit_qk("%k_view", "p0q", "  ")
        store_s(s0, "p0s", "  ")
        stage_v(v0, "%c0", "p0v", "  ")
        stage_k("%c16", k1, "p1s", "  ", kv="%k_view2")
        e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        init = ", ".join(f"{nm} = %oinit : {V8}" for nm in onames)
        init += f", %rmax = %ninf : f32, %rsum = %zero : f32, %sown = {s0} : {V8}"
        r2 = [f"%og{i}" for i in range(8)] + ["%gmax", "%gsum", "%gsown"]
        e(f"  {', '.join(r2)} = scf.for %ks_ = [%c0 to %max_vis step %c{KT}]({init}) -> ({types}, {V8})  {{")
        outs, snext = body2(True)
        e(f"    scf.yield {', '.join(outs)}, %nmax, %nsum, {snext} : {types}, {V8}")
        e("  }")
    else:
        r1 = [f"%of{i}" for i in range(8)] + ["%fmax", "%fsum"]
        if DOWHILE_MAIN:   # the unmasked loop runs no iteration: its results are its inits
            r1 = ["%oinit"] * 8 + ["%ninf", "%zero"]
        else:
            loop("%c0", "%split", ["%oinit"] * 8, "%ninf", "%zero", False, r1)
        r2 = [f"%og{'l' if SOFT1 and (SOFT2 or SOFT4) else ''}{i}" for i in range(8)] + ["%gmax", "%gsum"]
        loop("%split", "%max_vis", r1[:8], r1[8], r1[9], True, r2)
        if SOFT1 and SOFT4:   # the last tile: stage its V, then its P.V from slot (tiles - 1) % 2
            e("  %f4a = index.add %max_vis, %c15 : index")
            e("  %f4b = index.div %f4a, %c16 : index")
            e("  %f4c = index.sub %f4b, %c1 : index")
            e("  %f4ks = index.mul %f4c, %c16 : index")
            fv = load_v("%f4ks", "fv", "  ")
            stage_v(fv, "%c0", "fsv", "  ")
            e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
            e("  %f4p = index.rem %f4c, %c2 : index")
            e("  %f4rows = index.mul %f4p, %c128 : index")
            e("  %f4al = index.mul %f4p, %c64 : index")
            fin = soft1_pv("  ", [f"%ogl{i}" for i in range(8)], "fin", "%f4rows", "%f4al")
            for i in range(8):
                e(f"  %og{i} = vector.bitcast {fin[i]} : {V8} to {V8}")
        if SOFT1 and SOFT2 and not SOFT4:   # the last tile's P.V (its P^T / alpha / V are in LDS after the loop's last barrier)
            fslot = (None, None)
            if SOFT3:
                e("  %fs0 = index.add %max_vis, %c15 : index")
                e("  %fs1 = index.div %fs0, %c16 : index")
                e("  %fs2 = index.sub %fs1, %c1 : index")
                e("  %fs3 = index.rem %fs2, %c2 : index")
                e("  %fsrows = index.mul %fs3, %c128 : index")
                e("  %fsal = index.mul %fs3, %c64 : index")
                fslot = ("%fsrows", "%fsal")
            fin = soft1_pv("  ", [f"%ogl{i}" for i in range(8)], "fin", *fslot)
            for i in range(8):
                e(f"  %og{i} = vector.bitcast {fin[i]} : {V8} to {V8}")
        if SOFT1:
            e("  scf.if %isA {")
            e("    view.store %gsum, %ls_view[%pa_own] : f32, view<64xf32>")
            e("  }")
            e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
            e("  %lsum_l = view.load %ls_view[%pa_own] : view<64xf32> -> f32")

    # ---- epilogue: o / sum * sigmoid(gate); lane writes dims 8h..8h+7 of each 16
    e(f"  %lsum = scalar.addf {'%lsum_l' if PAIR or SOFT1 else '%gsum'}, %zero : f32")   # already the full row sum
    e("  %lpos = scalar.cmpf ogt, %lsum, %zero : f32")
    e("  %linv0 = scalar.divf %one, %lsum : f32")
    e("  %linv = scf.select %lpos, %linv0, %zero : f32")
    e(f"  %linv8 = vector.splat %linv : {V8}")
    e(f"  %lsum8 = vector.splat %lsum : {V8}")
    e(f"  %nlog2e8 = vector.splat %log2e : {V8}")
    e("  %elc = index.min %r_lq, %B_1 : index")
    e("  %eoff0 = index.mul %elc, %c6144 : index")
    e("  %ehb = index.mul %r_head, %c256 : index")
    e("  %eoff1 = index.add %eoff0, %ehb : index")
    e("  %eh8 = index.mul %half, %c8 : index")
    e("  %eoff2 = index.add %eoff1, %hd128 : index")
    e("  %eoff = index.add %eoff2, %eh8 : index")
    def gate_addr(f):
        e(f"  %eo{f}c = index.constant {16 * f} : index")
        e(f"  %eo{f} = index.add %eoff, %eo{f}c : index")
        e(f"  %eo{f}b = index.add %eo{f}, %c4 : index")

    def gate_load(f):
        e(f"  %eg{f}a = vector.load %g_flat[%eo{f}] : view<[%qtot]xf32> -> {V4}")
        e(f"  %eg{f}b = vector.load %g_flat[%eo{f}b] : view<[%qtot]xf32> -> {V4}")

    def gate_loads(f):
        gate_addr(f)
        gate_load(f)
    gate_each = GATE_EACH or PAIR   # PAIR: 16 output fragments are live, so each gate fragment right before its use

    def gate_batch(fs):
        # all gate loads together (latency); GATE_EACH: each fragment's right before its use (fewer live registers)
        if gate_each:
            return
        if GATE_FENCE and not GATE_EACH:
            for f in fs:
                gate_addr(f)
            e("  scf.schedule.fence")
            for f in fs:
                gate_load(f)
        else:
            for f in (() if GATE_EACH else fs):
                gate_loads(f)
    gate_batch(range(8))
    if TILED_OUT:
        e("  %et16 = index.constant 16 : index")
        e("  %et384 = index.constant 384 : index")
        e("  %etk = index.div %elc, %et16 : index")
        e("  %etr = index.rem %elc, %et16 : index")
        e("  %etkb = index.mul %etk, %et384 : index")
        e("  %etrb = index.mul %etr, %et16 : index")
        e("  %etc0a = index.add %ehb, %hd128 : index")   # the lane's column: head * 256 + d
        e("  %etc0 = index.add %etc0a, %eh8 : index")
        e("  %etlast = index.sub %qtot, %c8 : index")
        # fragment f's column is etc0 + 16 f: tile column etc0 / 16 + f, offset etc0 % 16, so its address is base + 256 f
        e("  %ett0 = index.div %etc0, %et16 : index")
        e("  %etm0 = index.rem %etc0, %et16 : index")
        e("  %eti0 = index.add %etkb, %ett0 : index")
        e("  %etb0 = index.mul %eti0, %c256 : index")
        e("  %ets0 = index.add %etb0, %etrb : index")
        e("  %etu0 = index.add %ets0, %etm0 : index")
        e(f"  %et1792 = index.constant {256 * (NFR - 1)} : index")
        e("  %etmax = index.sub %etlast, %et1792 : index")
        e("  %etbase = index.min %etu0, %etmax : index")   # always in range; states it for the bound proof

    def tiled_addr(f):
        """fragment-major output address of fragment f: base + 256 f"""
        e(f"    %eto{f}c = index.constant {256 * f} : index")
        e(f"    %eto{f} = index.add %etbase, %eto{f}c : index")
    for f in range(NFR):
        if f == 8:
            gate_batch(range(8, 16))
        if gate_each:
            gate_loads(f)
        e(f"  %eg{f} = vector.concat<0> %eg{f}a, %eg{f}b : {V4}, {V4} -> {V8}")
        e(f"  %egn{f} = vector.negf %eg{f} : {V8}")
        e(f"  %egm{f} = vector.mulf %egn{f}, %nlog2e8 : {V8}")
        e(f"  %egx{f} = vector.exp2f<afn> %egm{f} : {V8}")
        e(f"  %egd{f} = vector.addf %ones8, %egx{f} : {V8}")
        e(f"  %egr{f} = vector.divf %ones8, %egd{f} : {V8}")
        e(f"  %eod{f} = vector.divf %og{f}, %lsum8 : {V8}")
        e(f"  %eov{f} = scf.select %lpos, %eod{f}, %zeros8 : {V8}")
        e(f"  %eout{f} = vector.mulf %eov{f}, %egr{f} : {V8}")
        e(f"  scf.if {'%ewr' if PAIR else '%r_live'} {{")
        if TILED_OUT:
            tiled_addr(f)
        e(f"    %eoh{f} = vector.fptrunc %eout{f} : {V8} to {V8H}")
        e(f"    vector.store %eoh{f}, %o_flat[%{'eto' if TILED_OUT else 'eo'}{f}] : {V8H}, view<[%qtot]xf16>")
        e("  }")
    e("  kernel.return")
    e("}")
    return "\n".join(L) + "\n"


def gen_qrot():
    """yah_qrot (QROT): the kv4 attention's Q prologue as its own kernel: per (token, head) row of 256 dims, 4 threads of
    64 dims compute f16(H256(q * 1/16) * 1/16) with the staged path's operations (had64, the +0.0, the rounding), so
    the attention loads the same bits as direct Q fragments. f32 query [token][6144] -> f16 [token][6144]."""
    L = []
    e = L.append
    e("// GENERATED by tools/gen_attn_fa.py (gen_qrot) -- edit the generator.")
    e("amdgpu.target<gfx1151> @qrot_w32 {subgroup_size = 32}")
    e("config.decl @yah_qrot.token_count : %value: index where [range(%value, 1, 1048576)]")
    e("")
    e("kernel.def target(@qrot_w32) @yah_qrot() {")
    e("  %ntok = config.get @yah_qrot.token_count : index")
    e("  %c1 = index.constant 1 : index")
    e("  %c96 = index.constant 96 : index")
    e("  %c255 = index.constant 255 : index")
    e("  %c256 = index.constant 256 : index")
    e("  %th0 = index.mul %ntok, %c96 : index")       # 24 rows x 4 threads per token
    e("  %th1 = index.add %th0, %c255 : index")
    e("  %nwg = index.div %th1, %c256 : index")
    e("  kernel.launch.config workgroups(%nwg, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index")
    e("} launch(%src: buffer, %dst: buffer) {")
    e("  %base = index.constant 0 : offset")
    for v in (0, 1, 2, 4, 24, 64, 256, 6144):
        e(f"  %c{v} = index.constant {v} : index")
    e("  %x32 = scalar.constant 32 : i32")
    e("  %ntok = config.get @yah_qrot.token_count : index")
    e("  %qtot = index.mul %ntok, %c6144 : index")
    e("  %rows = index.mul %ntok, %c24 : index")
    e("  %rows1 = index.sub %rows, %c1 : index")
    e("  %s_na, %d_na = buffer.assume.noalias %src, %dst : buffer, buffer")
    e("  %q_flat = buffer.view %s_na[%base] : buffer -> view<[%qtot]xf32>")
    e("  %o_flat = buffer.view %d_na[%base] : buffer -> view<[%qtot]xf16>")
    e("  %wg = kernel.workgroup.id<x> : index")
    e("  %tid = kernel.workitem.id<x> : index")
    e("  %g0 = index.mul %wg, %c256 : index")
    e("  %g = index.add %g0, %tid : index")
    e("  %row = index.div %g, %c4 : index")
    e("  %qpart = index.rem %g, %c4 : index")
    e("  %live = index.cmp ult, %row, %rows : index")
    e("  %rowc = index.min %row, %rows1 : index")
    e("  %qrow = index.mul %rowc, %c256 : index")    # [token][head][256]: row-major rows of 256
    e("  %qscale = scalar.constant 0.0625 : f32")
    e(f"  %qsc_v = vector.splat %qscale : {V4}")
    e(f"  %z4h = vector.constant 0.0 : {V4}")
    e("  %qdb0a = index.mul %qpart, %c64 : index")
    raw = []
    for c in range(8):
        t = f"0_{c}"
        e(f"  %qdc{t} = index.constant {8 * c} : index")
        e(f"  %qd{t} = index.add %qdb0a, %qdc{t} : index")
        e(f"  %qa{t} = index.add %qrow, %qd{t} : index")
        e(f"  %qa{t}b = index.add %qa{t}, %c4 : index")
        e(f"  %qv{t}a = vector.load %q_flat[%qa{t}] : view<[%qtot]xf32> -> {V4}")
        e(f"  %qv{t}b = vector.load %q_flat[%qa{t}b] : view<[%qtot]xf32> -> {V4}")
        e(f"  %qm{t}a0 = vector.mulf %qv{t}a, %qsc_v : {V4}")
        e(f"  %qm{t}b0 = vector.mulf %qv{t}b, %qsc_v : {V4}")
        raw += [f"%qm{t}a0", f"%qm{t}b0"]
    rot = had64(e, "fh", raw, (1, 2), 0.0625)
    e("  scf.if %live {")
    for c in range(8):
        t = f"0_{c}"
        e(f"    %qm{t}a = vector.addf {rot[2 * c]}, %z4h : {V4}")
        e(f"    %qm{t}b = vector.addf {rot[2 * c + 1]}, %z4h : {V4}")
        e(f"    %qh{t}a = vector.fptrunc %qm{t}a : {V4} to vector<4xf16>")
        e(f"    %qh{t}b = vector.fptrunc %qm{t}b : {V4} to vector<4xf16>")
        e(f"    %qh{t} = vector.concat<0> %qh{t}a, %qh{t}b : vector<4xf16>, vector<4xf16> -> {V8H}")
        e(f"    vector.store %qh{t}, %o_flat[%qa{t}] : {V8H}, view<[%qtot]xf16>")
    e("  }")
    e("  kernel.return")
    e("}")
    return "\n".join(L) + "\n"


def _deq_header(e, name, sym, extra_params):
    e(f"// GENERATED by tools/gen_attn_fa.py ({name}) -- edit the generator.")
    e("amdgpu.target<gfx1151> @deq_w32 {subgroup_size = 32}")
    e(f"config.decl @{sym}.cache_capacity : %value: index where [range(%value, 1, 1048576)]")
    e(f"config.decl @{sym}.rows : %value: index where [range(%value, 1, 1048576)]")
    e("")
    e(f"kernel.def target(@deq_w32) @{sym}() {{")
    e(f"  %rows = config.get @{sym}.rows : index")
    for v in (1, 15, 16, 64, 255, 256):
        e(f"  %c{v} = index.constant {v} : index")
    e(extra_params)
    e("  %th1 = index.add %th, %c255 : index")
    e("  %nwg = index.div %th1, %c256 : index")
    e("  kernel.launch.config workgroups(%nwg, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index")


def _deq_common(e, sym):
    e("  %base = index.constant 0 : offset")
    for v in (0, 1, 2, 8, 15, 16, 32, 64, 128, 255, 256, 1024, 4096):
        e(f"  %c{v} = index.constant {v} : index")
    e(f"  %cap = config.get @{sym}.cache_capacity : index")
    e(f"  %rows0 = config.get @{sym}.rows : index")
    e("  %rows = index.min %rows0, %cap : index")
    e("  %npg0 = index.add %cap, %c255 : index")
    e("  %npages = index.div %npg0, %c256 : index")
    e("  %pcap = index.mul %npages, %c256 : index")
    e("  %wg = kernel.workgroup.id<x> : index")
    e("  %tid = kernel.workitem.id<x> : index")
    e("  %g0 = index.mul %wg, %c256 : index")
    e("  %g = index.add %g0, %tid : index")


def gen_kdeq(bits=4):
    """yah_kdeq4 (kv4 prefill): the paged int4 K pool -> the paged f16 K pool [physical row][1024] (rows 0..rows-1), with the
    attention's own staging decode (stage_k_ K4: nibble pairs to f16 1 + u/16, one packed fma with the row group's
    (16 s, lo - 16 s)), so the fp16 attention reading it computes what the kv4 attention did. Thread: (row, KV head,
    16-dim chunk). Bindings: kq (i32), kscale (i32), ptab, out (f16)."""
    assert bits in (4, 8)
    sym = f"yah_kdeq{bits}"
    nw = 2 if bits == 4 else 4                  # dwords per 16-dim chunk
    rw, sw = (128, 32) if bits == 4 else (256, 8)   # code / scale dwords per row
    L = []
    e = L.append
    _deq_header(e, f"gen_kdeq{bits}", sym, "  %th = index.mul %rows, %c64 : index")
    e("} launch(%kq: buffer, %ks: buffer, %ptab: buffer, %out: buffer) {")
    _deq_common(e, sym)
    if bits == 8:
        e("  %c4 = index.constant 4 : index")
    e(f"  %kq32tot = index.mul %pcap, %c{rw} : index")
    e(f"  %kstot = index.mul %pcap, %c{sw} : index")
    e("  %otot = index.mul %pcap, %c1024 : index")
    e("  %kq_na, %ks_na, %pt_na, %o_na = buffer.assume.noalias %kq, %ks, %ptab, %out : buffer, buffer, buffer, buffer")
    e("  %kq_flat = buffer.view %kq_na[%base] : buffer -> view<[%kq32tot]xi32>")
    e("  %ks_flat = buffer.view %ks_na[%base] : buffer -> view<[%kstot]xi32>")
    e("  %pt_flat = buffer.view %pt_na[%base] : buffer -> view<[%npages]xi32>")
    e("  %o_flat = buffer.view %o_na[%base] : buffer -> view<[%otot]xf16>")
    e("  %r0 = index.div %g, %c64 : index")
    e("  %j = index.rem %g, %c64 : index")
    e("  %live = index.cmp ult, %r0, %rows : index")
    e("  %cap1 = index.sub %cap, %c1 : index")
    e("  %r = index.min %r0, %cap1 : index")
    e("  %h = index.div %j, %c16 : index")
    e("  %ch = index.rem %j, %c16 : index")
    e("  %lp = index.div %r, %c256 : index")
    e("  %pg0 = view.load %pt_flat[%lp] : view<[%npages]xi32> -> i32")
    e("  %pgr = index.cast %pg0 : i32 to index")
    e("  %pg = index.assume %pgr [range(%pgr, 0, 65535), lt(%pgr, %npages)] : index")
    e("  %pb = index.mul %pg, %c256 : index")
    e("  %po = index.rem %r, %c256 : index")
    e("  %pr = index.add %pb, %po : index")
    e(f"  %kr = index.mul %pr, %c{rw} : index")
    e(f"  %kh = index.mul %h, %c{16 * nw} : index")
    e(f"  %kc = index.mul %ch, %c{nw} : index")
    e("  %ka0 = index.add %kr, %kh : index")
    e("  %ka = index.add %ka0, %kc : index")
    e(f"  %kv = vector.load %kq_flat[%ka] : view<[%kq32tot]xi32> -> vector<{nw}xi32>")
    e(f"  %sr = index.mul %pr, %c{sw} : index")
    e(f"  %sh8 = index.mul %h, %c{sw // 4} : index")
    e(f"  %sc2 = index.div %ch, %c{2 if bits == 4 else 8} : index")
    e("  %sa0 = index.add %sr, %sh8 : index")
    e("  %sa = index.add %sa0, %sc2 : index")
    e("  %sv = view.load %ks_flat[%sa] : view<[%kstot]xi32> -> i32")
    e(f"  %kdm = scalar.constant {62915520 if bits == 4 else 66847740} : i32")       # 0x03c003c0 / 0x03fc03fc
    e("  %kdgm = scalar.constant 1006648320 : i32")    # 0x3c003c00
    e("  %s1 = vector.from_elements %sv : vector<1xi32>")
    e("  %sh = vector.bitcast %s1 : vector<1xi32> to vector<2xf16>")
    e("  %sS = vector.extract %sh[0] : vector<2xf16> -> f16")
    e("  %sC = vector.extract %sh[1] : vector<2xf16> -> f16")
    ws = []
    for d in range(nw):
        e(f"  %w{d} = vector.extract %kv[{d}] : vector<{nw}xi32> -> i32")
        for k, sh in enumerate((6, 2, -2, -6) if bits == 4 else (2, -6)):
            dec_pair(e, "  ", f"%g{d}{k}", f"%w{d}", sh, 0x03c003c0 if bits == 4 else 0x03fc03fc, "%kdgm")
            ws.append(f"%g{d}{k}")
    e(f"  %pk = vector.from_elements {', '.join(ws)} : vector<8xi32>")
    e(f"  %pf = vector.bitcast %pk : vector<8xi32> to {V16H}")
    e(f"  %S16 = vector.splat %sS : {V16H}")
    e(f"  %C16 = vector.splat %sC : {V16H}")
    e(f"  %dk = vector.fmaf %pf, %S16, %C16 : {V16H}")
    e("  %orr = index.mul %pr, %c1024 : index")   # the f16 pool row = the int4 pool's physical row
    e("  %oh = index.mul %h, %c256 : index")
    e("  %oc = index.mul %ch, %c16 : index")
    e("  %oa0 = index.add %orr, %oh : index")
    e("  %oa = index.add %oa0, %oc : index")
    e("  scf.if %live {")
    e(f"    vector.store %dk, %o_flat[%oa] : {V16H}, view<[%otot]xf16>")
    e("  }")
    e("  kernel.return")
    e("}")
    return "\n".join(L) + "\n"


def gen_vdeq(bits=4):
    """yah_vdeq4 (kv4 prefill): the paged int4 V^T pool -> the paged f16 V^T pool [KV head][physical tile][256 dims]
    [16 keys] (vtpage's layout), tiles 0..ceil(rows / 16) - 1, with the attention's own staging decode (vq4_unpack). Thread: (KV head,
    tile, dim). Bindings: vq (i32), vstat (i32), ptab, out (f16)."""
    assert bits in (4, 8)
    sym = f"yah_vdeq{bits}"
    nw = 2 if bits == 4 else 4                  # dwords per (tile, dim)
    L = []
    e = L.append
    _deq_header(e, f"gen_vdeq{bits}", sym,
                "  %ntl0 = index.add %rows, %c15 : index\n  %ntl = index.div %ntl0, %c16 : index\n"
                "  %th0 = index.mul %ntl, %c256 : index\n  %c4 = index.constant 4 : index\n  %th = index.mul %th0, %c4 : index")
    e("} launch(%vq: buffer, %vs: buffer, %ptab: buffer, %out: buffer) {")
    _deq_common(e, sym)
    e("  %c4 = index.constant 4 : index")
    e("  %vtiles = index.div %pcap, %c16 : index")          # pool tiles per KV head
    e("  %vq4tot0 = index.mul %vtiles, %c1024 : index")
    e(f"  %vq4tot = index.mul %vq4tot0, %c{nw} : index")
    e("  %otl0 = index.add %cap, %c15 : index")
    e("  %otl = index.div %otl0, %c16 : index")              # logical tiles per KV head
    e("  %otot0 = index.mul %vtiles, %c4096 : index")
    e("  %otot = index.mul %otot0, %c4 : index")
    e("  %vq_na, %vs_na, %pt_na, %o_na = buffer.assume.noalias %vq, %vs, %ptab, %out : buffer, buffer, buffer, buffer")
    e("  %vq_flat = buffer.view %vq_na[%base] : buffer -> view<[%vq4tot]xi32>")
    e("  %vs_flat = buffer.view %vs_na[%base] : buffer -> view<[%vq4tot0]xi32>")
    e("  %pt_flat = buffer.view %pt_na[%base] : buffer -> view<[%npages]xi32>")
    e("  %o_flat = buffer.view %o_na[%base] : buffer -> view<[%otot]xf16>")
    e("  %ntl0 = index.add %rows, %c15 : index")
    e("  %ntl = index.div %ntl0, %c16 : index")
    e("  %dim = index.rem %g, %c256 : index")
    e("  %gt = index.div %g, %c256 : index")
    e("  %t0 = index.rem %gt, %ntl : index")
    e("  %kvh0 = index.div %gt, %ntl : index")
    e("  %live = index.cmp ult, %kvh0, %c4 : index")
    e("  %c3 = index.constant 3 : index")
    e("  %kvh = index.min %kvh0, %c3 : index")
    e("  %otl1 = index.sub %otl, %c1 : index")
    e("  %t = index.min %t0, %otl1 : index")
    e("  %vks = index.mul %t, %c16 : index")
    e("  %lp = index.div %vks, %c256 : index")
    e("  %pg0 = view.load %pt_flat[%lp] : view<[%npages]xi32> -> i32")
    e("  %pgr = index.cast %pg0 : i32 to index")
    e("  %pg = index.assume %pgr [range(%pgr, 0, 65535), lt(%pgr, %npages)] : index")
    e("  %pb = index.mul %pg, %c16 : index")
    e("  %po = index.rem %t, %c16 : index")
    e("  %pt = index.add %pb, %po : index")
    e("  %vti0a = index.mul %kvh, %vtiles : index")
    e("  %vti0 = index.add %vti0a, %pt : index")
    e("  %vti1 = index.mul %vti0, %c256 : index")
    e("  %vti = index.add %vti1, %dim : index")
    e(f"  %vda = index.mul %vti, %c{nw} : index")
    e(f"  %vqv = vector.load %vq_flat[%vda] : view<[%vq4tot]xi32> -> vector<{nw}xi32>")
    e("  %vst = view.load %vs_flat[%vti] : view<[%vq4tot0]xi32> -> i32")
    e("  %v4g = scalar.constant 1006648320 : i32")    # 0x3c003c00
    e("  %vs1 = vector.from_elements %vst : vector<1xi32>")
    e("  %vsv = vector.bitcast %vs1 : vector<1xi32> to vector<2xf16>")
    e("  %vss = vector.extract %vsv[0] : vector<2xf16> -> f16")
    e("  %vsc = vector.extract %vsv[1] : vector<2xf16> -> f16")
    ws = []
    for d in range(nw):
        e(f"  %vw{d} = vector.extract %vqv[{d}] : vector<{nw}xi32> -> i32")
        for k, sh in enumerate((6, 2, -2, -6) if bits == 4 else (2, -6)):
            dec_pair(e, "  ", f"%vmg{d}{k}", f"%vw{d}", sh, 0x03c003c0 if bits == 4 else 0x03fc03fc, "%v4g")
            ws.append(f"%vmg{d}{k}")
    e(f"  %vpk = vector.from_elements {', '.join(ws)} : vector<8xi32>")
    e(f"  %vf = vector.bitcast %vpk : vector<8xi32> to {V16H}")
    e(f"  %vss16 = vector.splat %vss : {V16H}")
    e(f"  %vsc16 = vector.splat %vsc : {V16H}")
    e(f"  %vd = vector.fmaf %vf, %vss16, %vsc16 : {V16H}")
    e("  %oi0 = index.add %vti0, %c0 : index")   # the f16 pool tile = the int4 pool's physical tile
    e("  %oi1 = index.mul %oi0, %c256 : index")
    e("  %oi2 = index.add %oi1, %dim : index")
    e("  %oi = index.mul %oi2, %c16 : index")
    e("  scf.if %live {")
    e(f"    vector.store %vd, %o_flat[%oi] : {V16H}, view<[%otot]xf16>")
    e("  }")
    e("  kernel.return")
    e("}")
    return "\n".join(L) + "\n"


def gen_vtrans():
    """yah_transpose_v16: token-major f16 V cache [token][1024] -> V^T blocked [4 kv heads][ceil(capacity/16) tiles][256 dims][16 keys].
    Tokens >= token_count are zero. 32 x 32 tiles through LDS; grid (1024/32, pitch/32 rounded up)."""
    L = []
    e = L.append
    e("// GENERATED by tools/gen_attn_fa.py (gen_vtrans) -- edit the generator.")
    e("amdgpu.target<gfx1151> @vtrans_w32 {subgroup_size = 32}")
    e("config.decl @yah_vtrans.token_count : %value: index where [range(%value, 1, 1048576)]")
    e("config.decl @yah_vtrans.cache_capacity : %value: index where [range(%value, 1, 1048576)]")
    e("")
    e("kernel.def target(@vtrans_w32) @yah_transpose_v16() {")
    e("  %cap = config.get @yah_vtrans.cache_capacity : index")
    e("  %c1 = index.constant 1 : index")
    e("  %c31 = index.constant 31 : index")
    e("  %c32 = index.constant 32 : index")
    e("  %c256 = index.constant 256 : index")
    e("  %t0 = index.add %cap, %c31 : index")
    e("  %tiles = index.div %t0, %c32 : index")
    e("  kernel.launch.config workgroups(%c32, %tiles, %c1) workgroup_size(%c256, %c1, %c1) : index")
    e("} launch(%src: buffer, %dst: buffer) {")
    e("  %base = index.constant 0 : offset")
    for v in (0, 1, 4, 8, 15, 16, 32, 1024):
        e(f"  %c{v} = index.constant {v} : index")
    e("  %ntok0 = config.get @yah_vtrans.token_count : index")
    e("  %cap = config.get @yah_vtrans.cache_capacity : index")
    e("  %ntok = index.min %ntok0, %cap : index")
    e("  %p15 = index.add %cap, %c15 : index")
    e("  %p16 = index.div %p15, %c16 : index")
    e("  %pitch = index.mul %p16, %c16 : index")
    e("  %stot = index.mul %cap, %c1024 : index")
    e("  %dtot = index.mul %pitch, %c1024 : index")
    e("  %s_na, %d_na = buffer.assume.noalias %src, %dst : buffer, buffer")
    e("  %s_flat = buffer.view %s_na[%base] : buffer -> view<[%stot]xf16>")
    e("  %d_flat = buffer.view %d_na[%base] : buffer -> view<[%dtot]xf16>")
    e("  %tb = index.constant 2304 : offset")
    e("  %tile = buffer.alloca<workgroup> align(16) %tb : buffer")
    e("  %tv = buffer.view %tile[%base] : buffer -> view<32x36xf16>")
    e("  %cb = kernel.workgroup.id<x> : index")
    e("  %tbk = kernel.workgroup.id<y> : index")
    e("  %tid = kernel.workitem.id<x> : index")
    e("  %col0 = index.mul %cb, %c32 : index")
    e("  %tok0 = index.mul %tbk, %c32 : index")
    e("  %r = index.div %tid, %c8 : index")
    e("  %sg0 = index.rem %tid, %c8 : index")
    e("  %sg = index.mul %sg0, %c4 : index")
    e("  %zh4 = vector.constant 0.0 : vector<4xf16>")
    # load: token tok0 + r, cols col0 + sg .. +3 (zero past the tokens)
    e("  %tok = index.add %tok0, %r : index")
    e("  %live = index.cmp ult, %tok, %ntok : index")
    e("  %cap1 = index.sub %cap, %c1 : index")
    e("  %tokc = index.min %tok, %cap1 : index")
    e("  %sa0 = index.mul %tokc, %c1024 : index")
    e("  %sa1 = index.add %sa0, %col0 : index")
    e("  %sa = index.add %sa1, %sg : index")
    e("  %raw = vector.load %s_flat[%sa] : view<[%stot]xf16> -> vector<4xf16>")
    e("  %val = scf.select %live, %raw, %zh4 : vector<4xf16>")
    e("  vector.store %val, %tv[%r, %sg] : vector<4xf16>, view<32x36xf16>")
    e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    # store: col col0 + r, tokens tok0 + sg .. +3
    els = []
    for j in range(4):
        e(f"  %tj{j}c = index.constant {j} : index")
        e(f"  %tj{j} = index.add %sg, %tj{j}c : index")
        e(f"  %x{j} = view.load %tv[%tj{j}, %r] : view<32x36xf16> -> f16")
        els.append(f"%x{j}")
    e(f"  %ov = vector.from_elements {', '.join(els)} : vector<4xf16>")
    e("  %col = index.add %col0, %r : index")
    e("  %dt = index.add %tok0, %sg : index")
    e("  %dlive = index.cmp ult, %dt, %pitch : index")
    # [kv_head][tile][dim][16]: (kvh * tiles + dt/16) * 4096 + d * 16 + dt%16
    e("  %c256b = index.constant 256 : index")
    e("  %c4096 = index.constant 4096 : index")
    e("  %kvh = index.div %col, %c256b : index")
    e("  %dd = index.rem %col, %c256b : index")
    e("  %dtl = index.div %dt, %c16 : index")
    e("  %dj = index.rem %dt, %c16 : index")
    e("  %da0 = index.mul %kvh, %p16 : index")
    e("  %da1 = index.add %da0, %dtl : index")
    e("  %da2 = index.mul %da1, %c4096 : index")
    e("  %da3 = index.mul %dd, %c16 : index")
    e("  %da4 = index.add %da2, %da3 : index")
    e("  %da5 = index.add %da4, %dj : index")
    e("  %dlim = index.sub %dtot, %c4 : index")
    e("  %da = index.min %da5, %dlim : index")  # in range already; for the prover
    e("  scf.if %dlive {")
    e("    vector.store %ov, %d_flat[%da] : vector<4xf16>, view<[%dtot]xf16>")
    e("  }")
    e("  kernel.return")
    e("}")
    return "\n".join(L) + "\n"


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "vtrans":
        out = sys.argv[2] if len(sys.argv) > 2 else "yah_transpose_v16.loom"
        open(out, "w").write(gen_vtrans())
        print(out)
        return
    out = sys.argv[1] if len(sys.argv) > 1 else "yah_attn_fa.loom"
    open(out, "w").write(gen())
    print(out)


if __name__ == "__main__":
    main()


gen_kdeq4 = gen_kdeq
gen_vdeq4 = gen_vdeq
