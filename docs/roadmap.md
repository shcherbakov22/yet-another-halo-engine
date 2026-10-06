# Roadmap

What is next, and what is planned but not built. Numbers behind the leads are in [results.md](results.md).

## Next leads

- Prefill autotuner: per-kernel `Tile` (BN, wave layout, KSUB, decode-ahead, ...) per token-count bucket, written as a `YAH_TILES` table. In: the knob refactor, `YAH_TILES`, masked last tiles, runtime token counts, the calibrated cost model (scratch). Next: the BN = 128 test (6 resident waves per SIMD vs 4), an occupancy sweep to calibrate the model's latency term, a persistent bench harness, the search driver.
- Prefill, partial chunks: only the GEMMs shrink to the real tokens; the norms and other small kernels still run the whole chunk (~250 ms). Shrinking them too changed the last token's logits by ~1e-4 when it sat alone in its 16-token group (likely the 32-token DeltaNet chunk); find that first.
- Kernel overlap: in an HRX graph a barrier drains all earlier work on the one GPU queue, so only kernels between the same barriers overlap. Targeted waits or a second queue (upstream HRX) would let the DeltaNet and attention overlap GEMMs and make cross-chunk pipelining pay (about 1-2% at 2048-token chunks, ~4% at 512).
- Decode, kv4a16 attention: `part_q` with kv4 is latency-bound (~91 GB/s, 347 us per call at 30.7K). Bandwidth-bound it would be ~160 us, about -3 ms per token at 30K. The per-workgroup q rotation, barriers and group sums are the suspects.
- Decode, dispatch overlap: re-measure the no-barrier overlap (-1.4 ms when it was added). A single uncooled round after the cleanup showed no gain.
- Speculative decoding (drafting): planned.
- Upstream the local HRX patches (no-ordering-barrier dispatch flag, stream profile metadata, counters mode), with the owner's agreement. See build-and-run.md.
- Other models of the same family: `UD-Q4_K_S` fails to emit (its q5k `ffn_up` has no `yah_ffn_gemm_q5k_swiglu_f16.loom`).

## Planned, not built

- NPU GEMMs (XDNA2 through HRX `.xdna`): GPU-side BFP16 operand encoders and one split kstore GEMM end to end in one HRX stream (`LoomNpu`, DMA-BUF-shared operands) are done (architecture.md); the dequant writes the BFP16 weights itself and the unpack is coalesced (-22% on the qkv GEMM in the harness); next the split in `LoomPrefill` (weight decode off the critical path, one activation encode for qkv and gate, segments joined with `LoomNpu::Join`), the per-GEMM ratio, the other GEMM kinds' epilogues, and the NPU's DRAM contention (each call re-streams the activations). K = 17408 needs K chunking (its weight panel exceeds the 512 KB memory tile).
- GPU + NPU scheduling: prefill GEMMs split by output columns at a runtime ratio (start ~35-40% NPU), everything else and decode on the GPU. Two async queues joined by events. Both engines share the 130 W budget. The NPU keeps full performance under concurrency as long as the thermal power cap is below the max power cap (the concurrency derate is then ignored). Gate: GPU stream idle < 5% with the split on.
- Vision: `mmproj-F16.gguf` (0.86 GiB) is the only projector; image tokens (64-16384 per image) merge after it. Gate: an image prompt gives correct output.
- Serving: CLI, HTTP and a sampler (greedy only today). Keep it thin.
