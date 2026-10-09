# Hardware

Measured facts about the box that decide kernel design and how to measure. Everything here was measured on this machine; the date is when it was last measured. Rates in cycles come from `SQ_BUSY_CYCLES`, which does not depend on the clock.

## The box

| item | value |
|---|---|
| APU | AMD Ryzen AI MAX+ 395 (Strix Halo), Radeon 8060S |
| GPU | gfx1151 (RDNA 3.5), 40 CUs = 20 WGPs = 80 SIMDs, wave32 and wave64 |
| memory | 32 GB LPDDR5X-8000, 256-bit (256 GB/s theoretical), shared with the CPU |
| GPU memory | GTT 24 GiB, VRAM carve-out 512 MiB, `iommu=pt`, THP always |
| MALL (last-level cache) | 32 MB |
| GPU clock levels | 600 / SMU-chosen / 2900 MHz (`pp_dpm_sclk`); the middle level is a live operating point, not a table entry |
| LDS | 64 KB per workgroup |
| VGPRs | 1536 per SIMD (allocation granule 24 on the gfx1151 target), at most 256 per wave |

## Memory bandwidth (2026-10-02)

Loom streaming kernels through HRX (`hal_run`), 1 GiB buffers, 16-byte accesses, grid-stride.

| kernel | launch | GB/s |
|---|---|---:|
| read | 16384 x 256, 1 access per iteration | 240.5 |
| read | 2048 x 256, unroll 4 | 230.1 |
| copy (read + write counted) | 2048 x 256, unroll 4 | 193.9 |
| write | 2048 x 256, unroll 4 | 170.5 |

- Read saturates near 240 GB/s (94% of theoretical) once a few thousand workgroups are in flight.
- Working sets under 32 MB stay in the MALL and read faster than DRAM. Long-context decode attention moves 128 MB per call at 32K; benchmark it there, not at 8K.
- The non-matrix prefill kernels (norms, conv, postnorm, unpack, RoPE) already run at 180-230 GB/s. Only fewer bytes (fusion) helps them.

## Matrix and vector rates (2026-10-01)

Microbenchmark `wmmarate.hip` under PMC, 16 waves per SIMD, 8 independent accumulator chains.

| instruction | cycles per instruction per SIMD |
|---|---:|
| `v_wmma_f32_16x16x16_f16` / `_bf16` | 34.0 |
| `v_wmma_i32_16x16x16_iu8` | 34.0 |
| `v_wmma_i32_16x16x16_iu4` | 17.0 |
| WMMA, dependent chain, 1 wave per SIMD | 34.07 (no extra latency) |

Issue costs next to WMMA (extra cycles of the WMMA pipe per added instruction):

| added per WMMA | extra cycles each |
|---|---:|
| independent VALU (`v_fma_f32`, `v_perm_b32`, `v_bfe_u32`, `v_cvt_*`) | 1.03-1.18 |
| VOPD pair (`v_dual_fmac`) | ~1.9 (no saving over two VALUs) |
| dependent VALU chain (`v_perm_b32`) | 2.8 |
| `v_mul_lo_u32`, dependent | 4.2 |
| `ds_load_b128`, conflict-free, 1 per WMMA | 2.4-2.8 |
| `ds_load_b128`, 4 per WMMA | 7.7 (LDS-bandwidth-bound, ~128 B/clk per WGP) |
| `ds_load_b128` with 64 B lane stride (bank conflicts) | ~4x the conflict-free cost |
| `ds_load_b32` / `_b64`, 4 per WMMA | 6.1 (LDS cost is per instruction, not per byte) |
| `global_load_b128`, L1-resident | 2.5 (not cheaper than LDS) |

Consequences:

- A WMMA kernel's cycles are close to `34 x WMMA + VALU issue + ~3 x LDS instructions` per SIMD. All production prefill kernels run at 98-102% of this bound: barrier waits are absorbed by other waves, only instructions per WMMA move cycles.
- In full GEMMs, a removed decode VALU saved only ~0.23 cycles, not ~1.1: most decode VALU already hides under other waves' WMMAs.
- iu8 WMMA runs at the f16 rate, so int8 buys no arithmetic, only bytes. iu4 is 2x but needs 4-bit activations.
- fp16 WMMA measured with a wall-clock harness: 48.35 TFLOPS (HIP, 2026-08; clock not recorded).

## Wave32 vs wave64 (2026-10-01)

| | wave32 | wave64 |
|---|---:|---:|
| WMMA per 16x16x16, cycles per SIMD | 34.0 | 34.0 |
| VALU instruction, cycles | 1.03 | 1.76 |
| LDS instruction, cycles | 1.49 | 2.53 |
| pure FP32 FMA, lane-FMAs per cycle per SIMD | 56.6 (needs VOPD pairing) | 60.0 (no pairing needed) |

- Wave64 WMMA keeps the full operand fragment per lane; only the accumulators halve. So per-WMMA overhead grows ~1.7x: wave64 tile GEMMs lose (IQ3_S +31%).
- Wave64 wins for pure FP32 VALU kernels (the recurrent DeltaNet: -16.5%).

## Register and LDS limits

| fact | consequence |
|---|---|
| `v_cvt_f16_f32` (and other low-half writes) can only target v0..v127 | with 128 VGPRs of accumulators live, every conversion evicts an accumulator to scratch. Narrow with `fptrunc(fma(x, 1.0, y))`, which selects `v_fma_mix` (any VGPR). |
| 64 KB LDS per workgroup | a 256 x 256 GEMM tile plus the IQ grid table does not fit; the tile GEMM is 128 x 256 |
| no `global_load_lds` (vmem-to-LDS) on gfx1151 | staging always goes through registers; Loom rejects async global-to-LDS copies on gfx11 |
| Loom's gfx11 encoding drops load cache hints other than device / regular | there is nothing to tune with cache hints |
| SMEM loads return out of order | every use of a scalar load drains all outstanding scalar loads; stage wave-uniform data through LDS instead |

## Dispatch and synchronization (2026-10-02)

| measure | value |
|---|---:|
| dependent dispatch on an HRX stream (decode, gap per boundary) | ~3.5-4 us |
| the same, replayed from an HRX graph | 3.32 us (stream 3.57 us) |
| grid barrier inside a resident Loom kernel, 20-40 / 80 / 160 workgroups | 0.26 / 0.37 / 0.49 us |
| prefill: GPU idle during pp2048 | 0.5% |
| prefill: dispatch gaps, median | 9 us (hidden: the host enqueues ahead) |

- Stock HRX records a full ordering barrier after every `hrx_stream_dispatch`, so independent dispatches never overlap (a local patch lifts this for decode; see build-and-run.md).
- Launch overhead does not matter for prefill. For decode (~560 dependent dispatches per token) it is ~3 ms per token; only fewer dependent dispatches reduce it.

## Clocks, heat and power (2026-10-01)

The GPU clock is limited by the SMU's GFX thermal controller, not by power. The GFX hotspot (Tgfx) goes from 47 to 87 C within ~30 ms of load and the SMU holds it at the tctl limit by lowering the clock ceiling. PPT, STAPM and PROCHOT do not fire at the default limit.

Steady state per instruction mix (5 s probe, default tctl):

| load | clock MHz | Tgfx C | socket W |
|---|---:|---:|---:|
| WMMA only | 2622 | 94.8 | 113 |
| VALU FMA only | 2067 | 94.8 | 101 |
| LDS only | 2634 | 94.8 | 98 |
| memory stream | 2841 | 67.1 | 62 |
| IQ3_S prefill GEMM | 2276 | 94.8 | 111 |
| pp2048 prefill (median) | ~2270 | 95-98 | ~115 |

- The sustained clock is set by heat per cycle: FP32 VALU is the hottest work. Fewer instructions per WMMA raise the clock as well as cutting cycles.
- tctl (`doas ryzenadj --tctl-temp=N`, resets on reboot): 105 lets Tgfx reach ~100 C (GEMM 2276 -> 2425 MHz) but trips PROCHOT every ~6 s under sustained load (600 MHz for ~1 s). 99 holds ~99 C without PROCHOT. The clean results in results.md were measured at tctl 95.
- Start temperature matters: pre-run Tgfx 40 C gave 2478 MHz, 49.5 C gave 2322 MHz. Cool the APU to <= 55 C before every timed run.
- Real weights draw more power than constant fills: the IQ4_XS GEMM ran 1.93 GHz on real bytes vs 2.56 GHz on fills, with the same cycle count.
- Standalone kernels run ~15-20% faster in ms than inside the pipeline, at the same cycle count: the difference is clock.
- Power: socket idle 13.8 W; WMMA 106 W, VALU 110 W, LDS 105 W, memory stream 69 W, IQ3_S GEMM 116 W. The PM table (ryzen_smu) gives socket = 1.06 x field[203] + 9.4 W; the compute rail carries ~98 of 117 W in the GEMM (limit 120). gpu_metrics' GFX power is a model estimate, not a measurement.

## Host side (2026-10-01)

| measure | value |
|---|---:|
| HRX device init | 180 ms |
| import of the 12 GiB GGUF mmap into HRX (warm page cache) | 250-450 ms (3.2 M 4 KB PTEs) |
| the same after warming the cache with `cat` | 830 ms (small page-cache folios) |
| host CPU during prefill, sleep-polled final wait | 3-4% of a core (busy-poll: 104%) |

HRX counters, rocprofv3, ATT and what does not work (PC sampling, RGP) are listed in build-and-run.md, Profiling.

## NPU compute tile (AIE2P) (2026-10-07)

Measured on low-asm leaves compiled by `loom-compile` (static bundle counts of straight-line code; the tile has no interlocks, so bundles are cycles) and checked on the NPU:

- A locked leaf (`schedule(locked)`) issues one op per bundle unless compiled with `LOOM_EXP_LOCKED_PACK=1`; then adjacent independent ops share a bundle. `schedule(phased)` is rejected for XDNA.
- A dependent accumulate costs ~9 cycles for both the elementwise bf16 MAC (`mma.bf16bf16.m8n8k1`, 64 lanes) and the bfp16 MMA (`mma.bfp16ebs8.m8n8k8`), whatever their descriptors' bypass stages say. With 5 accumulator registers (x4) that caps dependent MAC chains at ~0.5 per cycle.
- `vst.push.bfp16ebs8[.from.fp32]` pushes 576 bits into a 1024-bit store FIFO that writes 512-bit lines: after 8 pushes a whole line is pending and the next push overflows, losing lines silently. Drain it with `vst.flush.512` after every 8th push.
- `vst.push.bfp16ebs8.from.fp32` converts 64 f32 lanes like the GEMM encoder (E = exponent of the block max) except that E goes up by one only when a rounded mantissa leaves int8 (-128 stays), and subnormal inputs flush to 0.
- Budgets: 24 vec256 units, 5 accumulators of 2048 bits, 8 pointer registers. Pointer adds take immediates in multiples of 64 (others through a modifier register), scalar loads / stores -32..28 bytes, 512-bit loads -512..448. A `concat` of separately loaded halves often costs register moves.
- `vshuffle` modes 0-55 are permutations (56 and up are not); among them byte, 16-bit and 32-bit (de)interleaves and transposes, e.g. mode 35 is an 8 x 8 byte transpose. There is no data-dependent lane permute; `vldb.4x{16,32,64}` gathers four 64-bit windows from four pointers held in a vector.

## NPU data movement (AIE2P) (2026-10-08)

Measured with small Loom array programs on the NPU (yah-scratch/npu/waitp):

- A memory tile connects a north or south stream input only to the north or south output of the same channel (the AIE-ML switch rule); a mismatch (south 2 -> north 5) lost one word per DMA transfer, or stalled the flow under a lock. Loom's router keeps the channel through a memory tile (patch 0016).
- NPU DMA does not snoop CPU caches: CPU stores to registered host pages reach the NPU only after `clflush`. GPU stores with system scope (and GPU fills plus a release copy) reach it.
- A shim MM2S runs ahead of a slow consumer: ~10 transfers (22 of 16 bytes) sit in flight, so a polled value is that many polls old. A shim lock acquired by the reading BD and released by a tick's BD makes each read wait for a request: one fresh read per poll, a host write seen within one record.
- A queued shim task repeats its descriptor (or chain) at most 256 times; the BD iteration dimension adds an address offset per execution and does not multiply the count. A chain of identical BDs repeated 256 times gives chain x 256 transfers per task.
- Core stream reads (`mov.ss`) and writes (`mov.ms`) work in locked leaves. Do not use `mov.ss.nb` with `mov.ss.status`: a count of status-3 reads went wrong and the job hung (TDR).
- Private storage of a core is not cleared between images: a new image starts on the old image's tile memory.
- An image's setup writes only the stream-switch routes, locks, descriptors and channels it uses; nothing resets the array between images of one context (the shim DMA channel control has no reset field: only the context's creation resets a column). An image of another layout therefore inherits the previous image's state: a master still selecting a slave the new image routes broadcasts into a sink nobody drains (`LOOM_EXP_ROUTE_RESET` disables the data masters, clears the shims' packet rules and resets every memory-tile channel first), and a tile that ran a fill (decoder) worker in one image and a GEMM worker of a cascade column in the next stalled that column mid-call even with the reset: its head starved while every producer had finished. Keep each tile's role across a set's images (column pairs keep the fill column at 7). Standalone checks start on a fresh context and cannot show any of this. (2026-10-09)
- The firmware reports each column's tile status (DMA channel states and current BDs, core status, locks) for live contexts: `tools/npu_aie_status.py` (AIE2P Core_Status: bit 14 cascade-input stall, bit 16 debug halt). A stalled run read next to the same job parked healthy at its gate shows the column that differs. (2026-10-09)
- NPU command memory is small (~64 MB per process, shared with the images' storage): ~25 cached commands of ~4 MB failed in `memory_create` (errno 11). (2026-10-09)
- An image instance's storage is ~0.6-0.75 MB, ~0.5 MB of it the first command (the array setup inline: its bytes hold every binding address, so a copy of them taken under other bindings replays that setup and call). The FFN block's set loads ~25 images: one instance each plus the decoder swaps and 3 x 8 MB command arenas filled ~78 storages / 64.8 MB once setups had their own instances. (2026-10-09)
- A setup transaction (libamdf 0.1: op 0 write32 24 B, 1 block write 16 + 4 n B, 3 mask write 28 B, 128 DMA wait): of ~0.5 MB, ~325 KB are core program loads (tile offset 0x20000), ~115 KB data memory (one block per section: read-only tables, all-zero blocks for NOBITS storages), the rest registers. Images that follow each other in a layer (dcol, ffnsw, ffndn4, ffndn3) share 70% of their program words; swiglu and down images differ the most (~210 KB). (2026-10-09)
- Hardware contexts share the array in time only: with firmware that advertises AIE2_TEMPORAL_ONLY (this one) amdxdna requests all 8 columns from column 0 for every context, so two contexts never run side by side (two 1-column contexts of a decoder image: 3.27 s alone, 6.26 s concurrently; `xrt-smi examine -r aie-partitions` lists both in partition 0, columns 0-7). Roles on separate columns need one image. (2026-10-09)
- A shim descriptor's step and wrap fields address at most 1023 words per dimension and ~4 MB per step; a chain of descriptors (one per index of an axis, any byte offset each) and the BD iteration (at most 64 steps) add two axes. (2026-10-09)
