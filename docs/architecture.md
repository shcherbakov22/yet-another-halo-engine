# Architecture

The engine runs one model (Qwen3.8-27B, IQ4_XS GGUF) on one GPU (Strix Halo, gfx1151). Every kernel is written in Loom and compiled for the exact shapes of this model. Python generators write the Loom source, `emit_hal.py` compiles it to an HRX HAL executable, and two small C++ drivers dispatch the HAL sets through the HRX native API. There is no general tensor runtime: the layer loop is hand-written.

This page describes the model, the prefill pipeline, the decode step, the data layouts and the HAL sets. Numbers are in [results.md](results.md), measured hardware facts in [hardware.md](hardware.md).

## The model

Qwen3.8-27B is a hybrid: every 4th layer is full (softmax) attention, the others are Gated DeltaNet (a linear-attention recurrence). The GGUF stores it as architecture `qwen35`. `engine/core/config.hpp` reads every field from the file.

| item | value |
|---|---|
| layers | 64 main layers + 1 MTP (`nextn`) layer, which the engine does not run |
| full-attention layers | 16: layers 3, 7, 11, ..., 63 (`(l + 1) % 4 == 0`) |
| Gated DeltaNet layers | 48 |
| hidden | 5120 |
| FFN | SwiGLU, 17408 |
| attention | 24 query heads, 4 KV heads (GQA 6), head dim 256, RoPE on 64 of 256 dims, per-head QK RMSNorm, sigmoid output gate |
| attention q projection | 12288 rows: per head 256 q rows then 256 gate rows |
| DeltaNet | 16 key heads, 48 value heads, head dim 128, state 128 x 128 per value head, short conv width 4 over 10240 channels (q 2048, k 2048, v 6144), z gate 6144, alpha / beta 48 each |
| vocab | 248320; output head Q6_K, token embedding IQ4_XS |
| weights | 12.17 GiB, 3.84 bpw, 11 formats: IQ4_XS, IQ3_S, IQ3_XXS, IQ2_XXS, IQ2_XS, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0 |

The weights are the GGUF's own mmap, imported once into HRX as one device-visible buffer. Every tensor is an offset into that import. There is no second copy and no repacking.

## Prefill

`engine/model/loom_prefill.hpp` (`LoomPrefill`) runs a prompt through all 64 layers as one batch of B tokens (B = 2048 by default); `loom_forward_pp` is its command-line driver. Every kernel has the token count in its grid, so there is no per-tile loop. All activations are token-major (`[token][row]`). The residual stream is f32; GEMM inputs are f16.

### Kernels per layer, in order

Every layer starts with `yah_half_norm` (RMSNorm, writes the f16 GEMM input).

Full-attention layer:

1. `gemm_kqg` (attn_q, 12288 rows): the GEMM epilogue writes q and gate to separate buffers.
2. `gemm_kstore` attn_k and attn_v (1024 rows each).
3. `yah_fused_qk_rope_batched`: QK RMSNorm and RoPE. It writes roped q, and writes K straight into its page of the K pool. V goes to a one-chunk f16 scratch.
4. KV writers: fp16 `yah_vtpage` (V^T tiles into the paged pool); quantized KV `yah_kmean` (first chunk only), `yah_kq8` or `yah_kq4`, then `yah_vq8` or `yah_vq4`.
5. `yah_attn_wmma` (FlashAttention-style, see below). It writes `f16(o / sum * sigmoid(gate))` straight into the o-projection input.
6. `gemm_kres` attn_output (5120 x 6144): the GEMM reads the residual and writes residual + W x.

Gated DeltaNet layer:

1. `gemm_kstore` attn_qkv (10240 rows), attn_gate (z, 6144), ssm_alpha and ssm_beta (48 rows each).
2. `yah_ssm_conv_kq`: the causal conv over the 10240 channels with the q / k L2 normalization (`prep_kq`) fused in.
3. `yah_deltanet_prep_ab`: decay alpha and beta per token and head, and the conv ring (the last 4 inputs) for the next chunk or the decoder.
4. `yah_deltanet`: chunked WY Gated DeltaNet (see below).
5. `yah_ssm_postnorm_fp16`: gated RMSNorm per head with z, f16 out.
6. `gemm_kres` ssm_out (5120 x 6144).

Then the FFN, for every layer:

1. `yah_half_norm` (post_attention_norm); `norm_t.hal` writes the fragment-major copy for afrag GEMMs (both when one of gate / up has no afrag form).
2. `gemm_kstore` ffn_gate (17408 rows, f32 out).
3. `gemm_swiglu` ffn_up: `silu(gate) * up`, f16 out (fragment-major, `.af.to.hal`, when ffn_down is afrag too).
4. `gemm_kres` ffn_down (5120 x 17408).

Layers whose ffn_gate and ffn_up share a format with a fused form (`gemm_ffn_<fmt>_..`, `AF_FFN`) run them as one GEMM writing silu(gate) * up. Where the set has afrag forms (below; `AF` in `emit_prefill_pp.py`), the GEMMs run them, also the DeltaNet qkv / gate (`norm_rt.hal` writes both layouts: alpha / beta stay row-major), ssm_out (`postnorm_t.hal`) and the attention o-projection (`wmma_t[_c<i>].hal`) and the attention q / k / v, each producer storing fragment-major. These GEMMs run their afrag forms on chunks where 512-token tiles pad no worse than the narrow ones (`LoomPrefill::AfChunk`).

The last layer's tail (o-proj / ssm_out and the FFN) runs only the GEMM token tiles holding the rows the caller reads (`RunLayers` keep: the last token, or none on chunks whose output nobody reads). After the last layer of the last chunk: `yah_rmsnorm` on the last token, the Q6_K output GEMV (`yah_gemv_q6k`, 248320 rows) and `yah_argmax`. A pp2048 pass is 867 dispatches. The token embedding is dequantized on the host and uploaded once per chunk.

### The tile GEMM

All big GEMMs come from `tools/gen_gemm_tile.py`. They are about 90% of prefill time.

- Workgroup tile: 128 rows x 256 tokens, wave32. 16 waves of 32 x 64 by default; 8 waves of 32 x 128 (`WAVE_FMTS`) for IQ4_XS, IQ3_S, IQ3_XXS, Q3_K, Q4_K, Q5_K.
- Per K phase, the decoded weight tile and the activation tile are both in LDS, so the MMA loop reads only LDS. LDS rows are padded (weights +8 f16, activations +8 f16) to remove bank conflicts.
- The next phase's weight bytes and activations are loaded into registers while the current phase computes, then stored to LDS after it. A schedule fence keeps the next loads behind the current LDS stores.
- Decode-ahead (IQ4_XS, Q4_K, Q6_K): the decoding waves decode phase p+1 into a second weight tile while every wave multiplies phase p. Off for the IQ4_XS and Q4_K residual GEMMs at K = 6144, where it serializes on `vmcnt(0)`.
- The weight decode is exact (bit-identical to the reference dequant). Decoders work on 32-bit words, read headers with one vector load per block, and narrow to f16 with `v_fma_mix` where `v_cvt_f16_f32` would hit the v0..v127 window.
- Epilogue kinds: `kstore` (plain store), `kqg` (q / gate split), `swiglu` (reads the f32 gate, LDS epilogue for IQ3), `kres` (fused residual add into a second hidden buffer; the driver swaps the two).
- On grids of 320 or more workgroups, the second workgroup on each WGP in the first round runs 8000 empty workgroup barriers before it starts. This breaks the lockstep epilogue bursts of the short-K residual GEMMs.
- Small shapes (the 48-row ssm_alpha / ssm_beta) use a 16-row x 64-token tile.
- Afrag forms (`<hal>.af.hal`, `Tile.afrag`; the FFN GEMMs of `AF_FMTS` in `emit_prefill_pp.py`): only the decoded weights go to LDS; the activations are WMMA B fragments loaded straight from a fragment-major input (16 x 16 tiles of 512 contiguous bytes, `[token / 16][k / 16][token % 16][k % 16]`). 128 x 512 workgroups of 16 waves x (128 rows x 32 tokens), KSUB 128, two workgroups per WGP at 192 VGPRs, staggered on every grid. Each k step loads the next step's B fragments; the next phase's weight loads go in the last k step. kstore / kres store through 16-row LDS slabs in the weight tile (free after the K loop), swiglu through `swiglu_epilogue`.

HAL names encode the shape: `gemm_<kind>_<fmt>_<m_tiles>_<k_blocks>.hal`, with m_tiles = rows / 16 and k_blocks = K / 256 (Q8_0: K / 32). For example `gemm_kstore_iq4xs_1088_20` is a 17408 x 5120 IQ4_XS GEMM.

### Attention

`tools/gen_attn_fa.py`, FlashAttention-style with the softmax in registers. A workgroup is 32 query tokens x 2 query heads of one GQA group, 8 waves; each wave pair owns 16 queries and each wave one half of the head dim.

- S^T = K Q^T, so a lane owns one query column; the pair adds its partial scores through a private LDS slot.
- P^T becomes the B operand of the P.V WMMA through one xor-16 swizzle. V is read as V^T tiles.
- The O rescale is skipped when no row max of the wave grew (exact).
- Workgroups run head-pair-fastest in longest-first order, so K/V tiles hit L2.
- One V buffer (loaded at the top of phase A, staged at its end); 3 workgroups per WGP.
- Chunked prefill emits one attention HAL per chunk (`wmma_c<i>.hal`, start_pos = i B).

It is not bit-identical to the HIP-order kernel; the tiered accuracy gate accepted it (see build-and-run.md).

### Gated DeltaNet

`tools/gen_gdn_chunk.py`: the chunked WY form (chunk C = 32 tokens). A workgroup takes 64 value rows of one head (grid 2 x 48), 8 waves; the state stays in f32 WMMA accumulators and the matmul inputs are f16. Gating is in log space; log2(alpha) is clamped at -100 because alpha underflows to 0 deep in the model. Output error vs the exact recurrence is ~2e-4 relative; end-to-end KLD ~3e-6.

### One HRX graph per chunk

`RunLayers` records a chunk's ~870 dispatches into one HRX graph (`LoomGraph` in `engine/model/loom_runtime.hpp`) and launches it once. Edges come from the byte ranges each dispatch reads and writes: the GEMMs declare their outputs, every other kernel counts all its bindings as written. Kernels with no data between them run at the same time: a layer's input projections, and the DeltaNet gate projection beside the conv and the DeltaNet. A stream dispatch always ends with an ordering barrier and the GPU has one queue, so the graph is how stock HRX overlaps kernels. Results are bit-identical to dispatching on the stream.

### Chunked prefill (long context)

A set emitted with `YAH_CTX=T` runs every kernel at the chunk size B but sizes the KV pools for T tokens. The prompt runs in passes of B tokens over the 64 layers, carrying the conv ring and the DeltaNet state between chunks. One-pass and chunked runs are bit-identical at 8K.

A prompt that ends inside a chunk runs a partial last chunk. The GEMMs launch only the token tiles that hold real tokens (the driver sizes their grid from the count), and chunked sets carry each GEMM also with 128- and 64-token tiles: `LoomPrefill::PickGemm` takes the variant with the fewest padded token rows weighted by its cost per row; every other kernel runs the whole chunk over padding rows, which are zero-filled at allocation and so always finite. `yah_deltanet_prep_ab` reads the real token count from a device scalar: padding tokens get decay 1 and update 0, so they leave the DeltaNet state unchanged, and the conv ring advances only past the real tokens. Attention is causal, so the real tokens never see the padding, and decode overwrites the padding's KV rows before reading them. The KV quantizers also read the real token count: `yah_kmean` averages K over the real rows only and `yah_vq8` / `yah_vq4` repeat the last real row into a partial 16-key tile, so a partial chunk's result does not depend on what the padding rows hold (which differs with the GEMM tile that ran and with earlier prompts). With quantized V, `yah_vseed` copies the real V rows of the last, partial 16-key tile into the decoder's open tile.

### Prefill calibration while serving (shelved)

Off since 2026-10-03: the server does not call `EnableCalibration`, and sets carry the menu only with `YAH_CALIB_MENU=1`. Kept for later; as built:

Chunked sets also carry a calibration menu: for each tile GEMM, up to 8 `<gemm>.m<i>.hal` tiles one knob away from its own (decode-ahead, KSUB, waves, token tile), built in parallel by the emitter and kept only if they hash the same as the GEMM at a full and a partial chunk, twice. The server (`model/prefill_calib.hpp`) picks among the GEMM, its narrow variants and its menu per token bucket (64 ... 2048) from real prefills: while a bucket is open, each occurrence runs the plain GEMM or the least-sampled open challenger, the chunk's graph is profiled in-process (HRX patch 0006, about 0.2% cost), and each challenger occurrence is timed by the span of its overlap group against the plain GEMM in the same chunk (one clock, no temperature drift). Clearly slower arms drop out; a bucket closes on one arm or 12 samples each and takes a challenger only if it is clearly faster. The state lives in `$XDG_CACHE_HOME/yah/calib-<set>.txt` (a re-emit starts over). Outputs are bit-identical whatever runs; only time moves. Buckets of GEMMs that run a few times per chunk need many prompts to close.

### NPU GEMM path (in progress)

Each large prefill GEMM can be split by output columns: the NPU (XDNA2, 32 compute tiles through HRX `.xdna`) computes the trailing rows, the GPU the rest. Wired into the driver for the DeltaNet qkv / gate GEMMs (below); the other GEMM kinds are next.

The NPU multiplies one-pass BFP16 (`bfp16ebs8`) operands with f32 accumulation. `tools/gen_npu_gemm.py` emits the NPU GEMM (array program + leaves, compiled by `loom-compile` with HRX patch 0013); `tools/npu_gemm_check.py` runs it through `iree-xdna-run` against a float64 oracle. A fragment is 8 rows x 8 consecutive k: per row `[E u8][8 x int8 m]`, value `m * 2^(E - 133)`, 72 bytes; E is the f32 exponent field of the block's max |x|. The NPU GEMM is a cascade: per column, 3 rows compute K-slice partials and pass them down the accumulator cascade, the 4th row adds its slice and holds C across passes (f32; on the last pass it packs each finished 16-row segment to bf16 in place, round to nearest even, and the C ring sends only that half: `LOOM_EXP_LS_SEND_PITCH=2` spaces the ring slots two records apart). A column computes 80 output columns (5 16-column weight slabs, `gen_npu_gemm.NP`) of 64-row M blocks: the kernel is bound by the operand streams into the tiles (36 * (1 / MP + 1 / NP) bytes per MMA over two ~7.7 B/cycle streams), and 5 slabs is the widest panel whose whole K = 5120 weights still fit a column's 512 KB memory tile; 8 columns make 640 output rows per call. `tools/gen_npu_dequant.py` is a weight decoder for one NPU tile (Q4_K: raw blocks in, these fragments out; not wired yet, see roadmap.md).

The GPU writes both operand streams in the NPU's layout (`tools/gen_bfp16_encode.py`): activations from the f16 GEMM input (`yah_bfp16_encode_act`; the norms that feed the K = 5120 sites write it themselves, `<norm>_bfp.hal` from `gen_half_norm.gen_split` bfp: 8 consecutive columns per lane, one row block each, staged in LDS and copied out as 36-byte runs, f16 output bit-identical, `tools/norm_bfp_check.py`; the DeltaNet postnorm over heads 0-39, ssm_out's NPU K chunk, too, `gen_postnorm_bfp` / `tools/postnorm_bfp_check.py`; the ffn unpack writes down's for its columns), weights straight from the quantized rows (`yah_dequant_<fmt>_bfp16`, `gen_gemm_tile.DQ_BFP`: the tile GEMM's dequant kind decodes each 64 x 64 phase into LDS as the GEMMs do, then every lane encodes row blocks (8 k of one row) into the phase's fragments staged in LDS, copied out as aligned 8-byte stores; one workgroup per (row group, K group)). The two-step form (`yah_dequant_<fmt>` f16 `[rows][K]`, then `yah_bfp16_encode_wgt`) is the checked reference. No weight copy persists. Both read K chunks of a longer row for the K = 17408 GEMMs, whose weight panel does not fit a memory tile (`DQ_BFP = (ks, passes, kb_start, kb_total)`; the encoder's `k_off` / `k_src`), and the encoder also reads the afrag GEMMs' fragment-major input (`tiled`). The kstore / kres / swiglu / fused-ffn / kqg split GEMMs (row-major and fragment-major output; kqg by whole heads, the unpack routing each NPU head's q and gate halves) and every encoder / decoder variant are checked byte for byte (`split_gemm_check.py`, `bfp16_check.py` with `BFP_SRC` / `BFP_KWIN`, `dq_bfp16_check.py` with `DQ_KCHUNK`). The GPU's share runs the kstore tile GEMM over the leading rows at the full output stride (`gen_gemm_tile.OSTRIDE`; `tools/split_gemm_check.py`, also in the prefill set's afrag form with f16 output for the DeltaNet qkv / gate: bit-identical rows for every format they use); `yah_npu_unpack` (`tools/gen_npu_unpack.py`, `tools/npu_unpack_check.py`) moves the NPU's bf16 C into the output's trailing columns (f32, or f16 rounded as the f16-output GEMMs round; for the other GEMM kinds also the kres residual add, the K chunks' partial sums, the swiglu `f16(silu(gate) * up)` with the GPU epilogue's ops, and fragment-major output), sweeping 4 output rows at a time across all column tiles (whole 128-byte lines read, 320 contiguous bytes stored per token and NPU column). `tools/bfp16_check.py` and `tools/dq_bfp16_check.py` check both against numpy oracles byte for byte, for every model format.

`engine/model/loom_npu.hpp` (`LoomNpu`) runs the NPU beside a `LoomDevice`: libamdf opens the XDNA device and the HRX xdna loader binds `.xdna` images. Shared operands are HRX device buffers, exported once as DMA-BUFs (`hsa_amd_portable_export_dmabuf` from HRX's own libhsa) and imported per binding view into the NPU, so GPU kernels touch them at full bandwidth; registered host pages work too, but they are snooped and cap GPU reads and writes at ~25 GB/s. `Enqueue` queues NPU calls behind the stream's current position; a relay thread waits for that timeline point, submits the calls (the control-only continuation once the image is resident, HRX patch 0014) and signals a semaphore. `Join` orders the stream after it: the host waits for the semaphore first, because HRX resolves a stream wait on a semaphore the host signals later in software, only after the stream has drained (an idle GPU gap of 0.1-0.2 ms); so the GPU work that runs beside the NPU is queued before `Join`. `engine/run/npu_split_run.cc`, driven by `tools/npu_split_check.py`, runs one split kstore GEMM this way on real weights and activations: the GPU's rows bit-identical to the full GEMM, the NPU's rows ~2e-7 from the BFP16 oracle.

In the driver (`YAH_NPU=1` with a set emitted with `YAH_NPU_SPLIT`; `emit_prefill_pp.npu_split`) every large GEMM of a full chunk can be split: the DeltaNet qkv and gate (one NPU job, the same activations), the attention q (kqg, whole heads; k / v run beside the NPU), ssm_out / attn_output (kres, K = 6144: the NPU takes K 0 - 5119), the FFN gate + up (the fused ffn or kstore + swiglu: the NPU computes the same rows of both, the unpack applies silu(gate) * up) and ffn_down (kres, K = 17408: the NPU takes three chunks of 5120). The NPU runs one image, K = 5120 per call (switching images costs ~0.55 ms each, 4 per layer before): the rest of out's and down's K (1024 / 2048) for the NPU's rows runs on the GPU beside the GPU's rows (`npurem_<site>_<fmt>.hal`, `gen_gemm_tile.KWIN`, f32 [tokens][rows]) and the unpack sums the chunks, that partial and the residual. Per site: the activation encoder writes the GEMM input (row-major or fragment-major, whichever the GPU GEMM reads) into the shared A, the NPU's rows are decoded straight into W panels, the graph is cut and the NPU calls (one per 640 rows and K chunk) are enqueued; the next segment runs the GPU's rows (`<hal>.npu.hal`: every form of the set's GEMM, OSTRIDE); then `LoomNpu::Join` and `yah_npu_unpack`. A chunk launches as ~507 graph segments; segment i + 1 is instantiated while segment i runs. Without the NPU it is one graph as before (bit-identical). `NpuSplit` (`engine/model/npu_split.hpp`) keeps libamdf out of `LoomPrefill`; `LoomNpuSplit` (`loom_npu.hpp`) binds one kernel instance per (image, A / W / C views), and `EnableNpu` binds all of a chunk's calls up front with a pass of the layers that dispatches nothing (a cold bind loads and patches the image's storage, ~3 ms each). A job's weight decode is recorded into the previous job's GPU segment (W has two slots), so it runs beside that segment's GEMM while the NPU works. The split out / down GEMMs have fewer row blocks than the 40 CUs: the persistent kres takes the first three token tiles and the tiled kres the last one beside it. C, which the NPU writes, is in host pages registered with libamdf and imported into HRX (`LoomNpu::CreateShared` host). Fabric / memory clock switching during NPU work raises deferred data-fabric machine checks that can corrupt an NPU output: run NPU work with the GPU performance level pinned high (results.md). Calibration is off for NPU chunks. GPU cache maintenance around the NPU's accesses: `LoomNpu::Enqueue` and `Join` each queue a 64-byte copy to / from a host-visible buffer, whose system-scope release / acquire fences write the encoders' output back before the NPU reads it and drop cached lines of C before the unpack reads it (the stream wait in `Join` sees a signaled semaphore and is resolved in software, without a fence; without the copies an unpack read stale C left by the previous site's unpack, so outputs varied run to run).

## Engine

`engine/model/engine.hpp` (`Engine`) loads the model, a chunked prefill set and a decode set once and serves `Generate(prompt, params, on_token)` calls one at a time (the `TextGenerator` interface in `engine/model/generator.hpp`; the server in `engine/serve/` talks only to that). Per request: reset the recurrent state, prefill the whole chunks, finish the tail with a partial chunk or with decode steps (whichever its measured costs say is cheaper), then decode. Greedy picks the argmax on the GPU; temperature / top_p sample on the host from the logits.

## Decode

`engine/model/loom_decoder.hpp` (`LoomDecoder`) is the single-token step. `loom_decode` feeds a prompt through it one token at a time; `loom_forward_pp` with `YAH_GEN` runs it after a prefill, on the prefill's KV pools and recurrent state. Decode reads all 12.40 GB of weights (token_embd excluded) once per token, so it is a bandwidth problem: the target is 240 GB/s.

### Step, per layer

1. `rmsnorm` (512 lanes).
2. Input projections as one band-fused GEMV (`gb_*`): attn_q / attn_k / attn_v, or attn_qkv / attn_gate / ssm_alpha / ssm_beta, in one dispatch.
3. Full attention: `unpack` (q / gate), `rope` (QK norm + RoPE), KV append (`dattn_kvappend`, or `dattn_kappend_q` + `dattn_vappend_q`), `dattn_part` (or `dattn_part_q`), `dattn_reduce`, then `gv_resid` attn_output.
4. DeltaNet: `deltanet_conv` (the conv fused into the DeltaNet step, gated norm inside), then `gv_resid` ssm_out.
5. `rmsnorm`, `gv_swiglu` (ffn_gate and ffn_up in one kernel), `gv_resid` ffn_down.

Head: `rmsnorm`, `gv_plain` output (Q6_K, 248320 rows), `argmax`. The argmax writes the next token id into a device token stream; the next step's `embed` kernel (IQ4_XS row) reads it. Positions come from a device array. So steps are enqueued back to back and the host waits only after the prompt and at the end.

### GEMV design

`tools/gen_gemv.py`, one kernel per (kind, formats, rows, K). The arithmetic is HIP's sub-16 decode: a row is cut into 16-element sub-blocks; each decodes to 16 small unsigned integers, a scale and an offset, and the lane accumulates `scale * dot(q, x) - offset * sum(x)`.

- Decode on 32-bit words (`vector<4xi32>`), never on `vector<16xi8>`, which Loom lowers byte by byte.
- Two rows per wave share each x load; 4 waves per workgroup (R x W = 2 x 4).
- `pipeline(2)` read-ahead on the sub-block loop.
- Butterfly reduce over the wave.
- Kinds: `plain`, `resid` (y += W x, the residual add fused), `swiglu` (gate and up of mixed formats in one kernel), and bands (`gen_bands`, several output matrices of one input in one dispatch).

The GEMV families run at 214-236 GB/s of the 240 GB/s peak.

### Decode attention

`tools/gen_decode_attn.py`, split-K over pages:

- `part`: one workgroup per (KV head, 256-key page), with the six query heads of the GQA group. Scores (thread = key), per-head block max and sum, p in LDS, then P.V (thread = dim). Read-ahead depth 3 on the K rows and 2 on the V^T tiles. Grid order (head, page) so the four KV heads of a page read together.
- `reduce`: one workgroup per query head, merges the pages and applies the sigmoid gate.
- Quantized KV (`part_q`): same grid and outputs, the reduce is shared. q goes to LDS in the codes' storage order (kv4: rotated in place by an 8-stage LDS butterfly). Each key thread dequantizes its own row. The open f16 V tile is added once after the tile loop. The K channel mean is never added back: q.m is constant over keys and cancels in the softmax.

## Data layouts

### Paged KV

All KV caches are paged with 256-token pages. One page table per sequence (i32, logical page -> physical page) is shared by all layers. A 16-key attention tile always lies in one page, so attention does one page-table load per K tile and per V tile. The host validates every entry (< number of pages) before upload; attention assumes the bound, the cache writers clamp to it. The context must be a multiple of 256.

Per full-attention layer, at context T (pool rows = T):

| format | K | V | bytes per token per layer |
|---|---|---|---:|
| fp16 | `[T][1024]` f16, row of position p = `ptab[p/256]*256 + p%256` | V^T `[4 kv heads][T/16 tiles][256 dims][16 keys]` f16, physical tile `ptab[t/16]*16 + t%16` | 4096 |
| kv8a16 | int8, one scale per (token, KV head, 128-dim half): codes `[T][1024 B]`, scales `[T][8]` f16 pairs | uint8 per channel per 16-key tile around the tile midrange: codes `[4][tiles][256] x 16 B`, stats `[4][tiles][256]` f16 pairs | 2336 |
| kv4a16 | H256 (Walsh-Hadamard over the whole head) of K, asymmetric int4 per 32-dim group: codes `[T][512 B]`, scales `[T][32]` f16 pairs | 15 levels per channel per 16-key tile: codes `[4][tiles][256] x 8 B`, stats as kv8 | 1408 |

Over 16 layers that is 64 KiB (fp16), 36.5 KiB (kv8a16) and 22 KiB (kv4a16) per token of context; at 32K context 2.0 / 1.14 / 0.69 GiB.

Details of the quantized formats (`tools/gen_kvq.py`):

- K is centred first: `yah_kmean` computes the per-channel mean of the first chunk's K (deterministic, no atomics), kept per layer. Subtracting a per-channel constant shifts each query row's scores uniformly, so softmax is unchanged.
- kv4 rotates K with H256; the attention rotates q the same way, so q.k is unchanged.
- Codes are stored in an order that lets the attention build f16 values with masks and ORs (`1 + u/16`, `1 + u/256`), then dequantize with one fma `f * S + C'`. All attention math stays f16 WMMA ("a16").
- With K and V quantized, the f16 KV cache is a one-layer, one-chunk scratch that RoPE writes and the quantizers read.
- Decode keeps each layer's open 16-key V tile in f16 and quantizes it when the 16th key arrives. Prefill runs end on whole chunks, so a handoff always starts on a tile boundary.

### Recurrent state

| buffer | layout |
|---|---|
| conv state | `[48 layers][10240 channels][4]` f32. Decode ping-pongs between two copies per token, so the three value heads that share a key head never read a half-updated state. |
| DeltaNet state | `[48 layers][48 value heads][128][128]` f32 (3 MiB per layer) |

Prefill and decode use the same layouts, so the decoder binds the prefill's buffers directly.

## HAL sets

A HAL set is a directory of compiled kernels plus one text file that records how to launch them. The drivers read the file and refuse any other grid: Loom drops index clamps it proves redundant from the launch grid, so a larger grid reads out of bounds (see build-and-run.md, GPU safety).

### Prefill set: `tools/emit_prefill_pp.py <model.gguf> <dir> [B]`

- One GEMM HAL per (kind, format, shape) on the shard, the fixed kernels (`norm.hal`, `convkq.hal`, `rowsplit.hal` = DeltaNet, `postnorm.hal`, `rope.hal`, `wmma.hal` = attention, `vtpage.hal`, `rmsnorm.hal`, `gemv.hal`, `argmax.hal`, ...), one rope / attention / KV-writer HAL per chunk, and the IQ grid tables (`grid_*.bin`, `ksigns_*.bin`).
- `dispatch.txt`, one line per HAL: `<hal> <tokens per workgroup> <row groups> <token tiles>`. Marker rows with zeros carry facts: `kv_paged`, `kv16_scratch`, `rope_kpaged`, `attn_f16out`, `attn_kq8` / `attn_kq4` / `attn_vq8` / `attn_vq4`, and `ctx <B> 0 <T>` for chunked sets.
- `tools/footprint_gate.py` runs for every GEMM before it is emitted and refuses a kernel whose declared operand footprint is larger than the buffer the driver binds.

### Decode set: `tools/emit_decode.py <model.gguf> <dir> [max_context]`

- `gv_<kind>_<fmts>_<M>_<K>.hal` and `gb_<fmts>_<Ms>_<K>.hal` (GEMVs and band GEMVs), `dattn_*` (attention), `rmsnorm`, `unpack`, `rope`, `deltanet_conv`, `embed`, `argmax`, and the IQ tables.
- `decode.txt`: `ctx <T>`, `kv q <K bits> <V bits>` for quantized KV, `rw <kind> <R> <W>` (GEMV geometry), and `grid <hal> <workgroups> 0` for every GEMV kernel.
- To decode after a prefill, emit the decode set with the same context as the prefill pools and the same `YAH_KV`. The driver refuses a mismatch.

### From generator to HAL

1. A generator (`tools/gen_*.py`) prints Loom text, one kernel per `gen_*` function. A few kernels are still hand-written `.loom` files in `engine/gpu/loom/`; the emitters compile those with fixed configs.
2. `engine/gpu/loom/emit_hal.py <file.loom> <outdir> sym=value ...` replaces every `config.get` with its constant, drops the `config.decl`, and has `iree-run-loom --emit-only --emit-hal-executable` compile the kernel for gfx1151. The kernel is built for exactly that config. The HRX/Loom tree is `external/hrx` (`engine/gpu/loom/hrx_paths.py`; `YAH_LOOM_HOME` points it elsewhere for diagnosis).
3. The emitter copies the result into the set and writes `dispatch.txt` or `decode.txt`.

Production sets build with the pinned HRX/Loom plus `engine/hrx/patches` (see build-and-run.md, HRX patches). Any other compiler change is for diagnosis only until it is a patch there.
