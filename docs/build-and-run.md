# Build, run, verify, profile

How to build the drivers, emit HAL sets, run prefill and decode, check correctness, profile, and measure without hanging the GPU or fooling yourself. Read the GPU safety section before you write or launch a new kernel.

## Prerequisites

- HRX / Loom: the revision pinned in `engine/hrx/PIN` plus the patches in `engine/hrx/patches/`. `engine/hrx/bootstrap.sh` clones it into `external/hrx` (gitignored; `YAH_HRX` overrides, and a symlink to an existing checkout works), checks out the pin, applies the patches and builds what the engine uses: `libhrx` and its HIP binding, `iree-run-loom`, `loom-compile`, `iree-profile`. `engine/hrx/bootstrap.sh --check` verifies the checkout; `engine/build_hrx.sh` refuses anything else.
- The official ROCm Core SDK 10.0.0 runtime packages, extracted (no system install) by `engine/hrx-env.sh --fetch` into `external/rocm10` (`YAH_ROCM` overrides). Do not use a TheRock nightly: HRX needs `hsa_amd_queue_create`, which only the 10.0.0 release exports. `engine/hrx-env.sh --check` verifies the install.
- `source engine/hrx-env.sh` in every shell that runs a GPU job. It only sets `LD_LIBRARY_PATH` and `IREE_HAL_AMDGPU_LIBHSA_PATH`; `/opt/rocm` stays usable. The Python emitters find the same paths through `engine/gpu/loom/hrx_paths.py`.
- The model: `~/Downloads/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf`.
- Python 3 with numpy for the emitters. The checkers also need llama.cpp's `gguf-py` on `PYTHONPATH`; use an OpenBLAS numpy (`/home/q/yah-scratch/venv/bin/python`), the system numpy is ~100x slower for the gate tools.

### HRX patches

Each patch says at its top what it is for. Patches stack (a later one may edit an earlier one's lines): the bootstrap builds the pin with every patch applied in order and brings the checkout's patched files to it, and `--check` compares them; a patch that no longer fits the pin stops it. Raise the pin deliberately: bootstrap, rebuild, then check the emits and the GPU references.

| patch | effect |
|---|---|
| `0001-stream-dispatch-no-ordering-barrier` | `HRX_DISPATCH_FLAG_NO_ORDERING_BARRIER`: decode overlaps independent dispatches (`LoomDevice::NoBarrierNext`) |
| `0002-stream-profile-metadata` | `HRX_PROFILE_MODE=dispatch` records stream dispatches |
| `0003-profile-counters-mode` | `HRX_PROFILE_MODE=counters` with `HRX_PROFILE_COUNTERS`: per-dispatch PMC such as `SQ_BUSY_CYCLES`, plus the gfx11 GL2C counters |
| `0004-loom-profile-function-filter` | `LOOM_PROFILE_FUNCTION=<glob>` for executable traces in the Loom HAL benchmark tool |
| `0005-gfx1151-f16-wmma-operand-placement` | the f16 WMMA spreads its operands over the register banks (upstream does it for bf16 only): prefill GEMM cycles -2.6% |
| `0006-dispatch-timestamps-callback` | dispatch device timestamps to an in-process callback (`hrx_device_profile_dispatches_*`), used by the shelved prefill calibration |
| `0007-amdgpu-u24-address-multiplies` | address multiplies use full-rate `v_mul_u32_u24` when value facts prove 24-bit operands: prefill cycles -0.26%, bit-identical |
| `0008-loom-concat-reservation-unavoidable-tier` | the allocator keeps a concat's result reservation when its residency-tier crossing is unavoidable, instead of placing one half alone and copying both into the tuple later (8 `v_mov` + a load wait per k phase in afrag GEMMs); bit-identical, enables the IQ3_S sign chain |
| `0009-loom-omit-dead-fma-mix-seeds` | after allocation, drops the `v_mov_b32 0` that seeds an f16 pair built by `v_fma_mixlo` + `v_fma_mixhi` (no bit of it is ever read), keeping the schedule and registers; afrag kstore cycles -1.0%, bit-identical |
| `0013-xdna-npu-cascade-gemm-compiler` | Loom AIE2P / XDNA pieces of the NPU cascade GEMM: memory-tile staging, replay and multicast, shared panels, cascade link timing, `worker.accumulate`, leaf-synchronized channels (`constrain.leaf_sync`), replay rings re-armed by the control program, `LOOM_EXP_*` placement and packing knobs (`LOOM_EXP_LS_SEND_PITCH`: leaf-synchronized send-ring slots n records apart, each sending its leading record); NPU-only code paths (GPU emits byte-identical) |
| `0014-xdna-continuation-invocation` | `iree_hal_amd_xdna_executable_query_continuation`: the control-only repeat invocation (no tile setup), ~545 -> ~150 us fixed cost per NPU call |
| `0015-graph-atomic-store-node` | `hrx_graph_add_atomic_store_node`: a 4 / 8-byte store recorded in a graph's command buffer (no partition break); with release + system scope the command processor makes every earlier write visible to the host and other devices first. The NPU prefill's "inputs ready" flags (one graph per chunk) |
| `0016-loom-xdna-npu-side-gate` | Loom AIE2P: an NPU job waits for a flag in memory itself (no host relay): core stream channels (`constrain.core_stream`, the core's own stream port), request-driven polls through a shim lock (`constrain.request`), a gate the control program waits for before moving any data and a done record queued after the outputs (`constrain.gate`, `constrain.signal`; streamed plans get `[gate \| done]` invocations), worker storage zeroed at setup; router fix: through a memory tile a route keeps its channel. Ungated images byte-identical |
| `0017-loom-xdna-gated-call-invocations` | gated plans that are not streamed (one replay group: the NPU GEMM with final outputs) publish [call \| gate \| done] as separate invocations, so a job of calls runs as [gate][call]...[call][done] with one gate; existing images byte-identical |

These are upstream candidates; do not push them to HRX without the owner's agreement.

## Build

```
engine/build_hrx.sh
```

It builds `libyah_core`, the core tools (`yah-tokenize`, `yah-dump`, `yah-weights`), the drivers `loom_forward_pp`, `loom_decode`, `hal_run`, `hal_bench`, the GPU / NPU split harness `npu_split_run` and the server `yah_server` into `engine/build/`. It compiles with clang (`CXX=` overrides), `-O3 -march=native -ffp-contract=off`. Always use this script: `cmake --build engine/build` does not build the drivers and prints nothing, and a stale driver with a new HAL set launches the wrong grid. `gpu_run.sh` refuses a driver binary older than its source.

`engine/build_loomhip.sh` builds `engine/build/loomhip` (needs hipcc), which runs one Loom hsaco through HIP for rocprofv3.

## Autotune

```
engine/tune/tune.py <model.gguf>        # ~7 min; writes engine/tune/tables/<model>.json
```

It tunes every tile GEMM the driver runs (42 for this model) at five token counts (64, 128, 256, 512, 2048): it enumerates the legal tiles (token tile, waves along tokens, KSUB, decode-ahead), compiles them in parallel through the emitter (footprint gate included, spilling tiles dropped), measures clock-free cycles on the model's real weights with `engine/build/gemm_bench` (rotating through each shape's tensors so weights stream from DRAM as in the pipeline), then re-measures the best few (successive halving). Every candidate's output must hash the same as the default tile's. The table holds the full-chunk tile per GEMM, up to two narrower variants and the measured best variant per token bucket. The emitters use the model's table automatically (`YAH_TILES=<file>` picks another, `YAH_TILES=` none); `LoomPrefill::PickGemm` follows its per-bucket picks. Tuning changes no result bits: re-emit, then check the GPU references.

Calibration while serving is built but shelved (architecture.md, Prefill calibration while serving).

## Emit HAL sets

Emitters run on the CPU and take a few minutes. Paths below are relative to `engine/gpu/loom`.

| set | command |
|---|---|
| prefill, one pass of B tokens | `python3 tools/emit_prefill_pp.py <gguf> <dir> 2048` (B must be a multiple of 256) |
| prefill, chunked, context T | `YAH_CTX=32768 python3 tools/emit_prefill_pp.py <gguf> <dir> 2048` (chunk 2048, pools for 32768 tokens; T a multiple of B) |
| prefill with kv8a16 / kv4a16 | add `YAH_KV=kv8` or `YAH_KV=kv4` (or mixed `k8v4`, `k4v8`; `k8` / `v4` alone quantize one side, prefill only) |
| prefill with the NPU column split | add `YAH_NPU_SPLIT=qkv=4480,gate=2560,q=5120,out=2560,down=2560,ffn=7680` (NPU rows per site, multiples of 640, q of 2560; any subset; see `emit_prefill_pp.npu_split`); run `loom_forward_pp` with `YAH_NPU=1` (full 2048-token chunks; AC power and `power_dpm_force_performance_level=high`, see results.md) |
| prefill with the NPU FFN block | `YAH_NPU_DCOL=1 YAH_NPU_FFNBLK=7168` (with `YAH_NPU_SPLIT` sites other than ffn; architecture.md) |
| prefill with fused NPU decode | `YAH_NPU_FUSE=1` with `YAH_NPU_SPLIT` (multiples of 640): 8 GEMM columns, each core decodes its own raw IQ4_XS rows; other formats stay on the GPU for now; add `YAH_NPU_FFNBLK=<F>` (F a multiple of 1024) for the NPU FFN block on layers whose gate, up and down share a format, IQ4_XS or IQ3_XXS |
| decode | `python3 tools/emit_decode.py <gguf> <dir> <max_context>` (multiple of 256, default 4096) |
| decode after a prefill | same, with `max_context` = the prefill set's context and the same `YAH_KV` |

`engine/build_hrx.sh <gguf> [dir]` also emits a decode set (default `engine/hal`), which the end-to-end gates use.

Big sets belong in `~/yah-scratch` or `/home/q/yah-hal-*`, not in the repo.

## Run

Every GPU job goes through `engine/run/gpu_run.sh <tag> -- <command>` (see GPU safety). Source `engine/hrx-env.sh` first.

Prefill:

```
engine/run/gpu_run.sh pp -- engine/build/loom_forward_pp <gguf> <set> <out-prefix> 2048 <ids-file>
```

- `<ids-file>`: whitespace-separated token ids. A short file is repeated to the token count. Make one with `engine/build/yah-tokenize <gguf> --stdin`.
- The token count must equal the set's B, or for a chunked set be a multiple of the chunk up to T.
- Prints `layers_ms=` (the 64-layer loop, wall clock) and `argmax=`. Writes `<prefix>.logits` (last token, f32) and `<prefix>.hidden` (final hidden, f32, token-major).

Prefill then decode:

```
YAH_GEN=64 YAH_DECODE_HAL=<decode set> engine/run/gpu_run.sh gen -- engine/build/loom_forward_pp <gguf> <chunked set> <out-prefix> 8192 <ids-file>
```

- The prefill set must be paged (the default) and leave room: prompt + generated tokens <= T.
- Prints `generated_ids=` and `decode_ms=` / `decode_tok_s=` (mean over the generated steps).

Standalone decode (the prompt goes through the decode path one token at a time):

```
engine/run/gpu_run.sh dec -- engine/build/loom_decode <gguf> <decode set> --ids "760 6511 314 9338 369" --gen 64
```

`--logits FILE` appends every step's logits (f32) for an external KL check.

## Serving

`engine/build/yah_server` serves the model over the OpenAI Responses API. It runs one generation at a time; other requests wait. The engine behind it (`engine/model/engine.hpp`) needs a chunked, paged prefill set and a decode set with the same context and `YAH_KV`. The default is 32K context with kv8a16 KV:

```
cd engine/gpu/loom
YAH_CTX=32768 YAH_KV=kv8 python3 tools/emit_prefill_pp.py <gguf> /home/q/yah-hal-serve 2048
YAH_KV=kv8 python3 tools/emit_decode.py <gguf> /home/q/yah-hal-serve-dec 32768
cd -
source engine/hrx-env.sh
engine/run/gpu_run.sh serve -- engine/build/yah_server --model <gguf> --prefill /home/q/yah-hal-serve --decode /home/q/yah-hal-serve-dec [--host 127.0.0.1] [--port 8080]
engine/build/yah_server --model <gguf> --fake    # canned replies, CPU only: for clients and API tests
```

With `--npu` the NPU computes part of every full prefill chunk's large GEMMs, as `YAH_NPU=1` does for `loom_forward_pp`: emit the prefill set with `YAH_NPU_SPLIT` as well (Emit HAL sets above). It needs the GPU performance level pinned high (`echo high | doas tee /sys/class/drm/card1/device/power_dpm_force_performance_level`; fabric / memory clock switches corrupt NPU outputs, results.md) and refuses to start otherwise. Partial chunks and decode run on the GPU alone.

A prompt runs as prefill chunks of 2048 tokens. When it ends inside a chunk, the last chunk is a partial one: its GEMMs run only the 256-token tiles that hold real tokens, so it costs about 0.25 s plus 0.36 s per 256 tokens (an 18-token prompt: 0.63 s to the first token). Tails of a few tokens go through decode steps instead (about 62 ms per token). Decode runs two steps ahead of the host: each step picks its token on the GPU (argmax, or for `temperature` > 0 `yah_dec_sample`: softmax at T cut to the top-p nucleus, an exact draw by Gumbel-max, ~1 ms) into the token stream the next step reads, and the host only reads the tokens (a host-mapped copy) to stream them and stop. A given seed draws other tokens than the earlier host sampler did; the distribution is the same. `loom_forward_pp` takes any token count the same way.

For quick tests, `engine/build/yah_chat` is a terminal chat client: it streams the reply (reasoning dimmed), keeps the conversation, and prints token counts, time to first token and tok/s after each reply. Commands: `/reset`, `/effort E`, `/temp T`, `/max N`, `/system TEXT`, `/quit`; Ctrl-C stops a reply.

```
engine/build/yah_chat --url http://127.0.0.1:8080 [--effort none|low|medium|high] [--temperature T] [--max N] [--system TEXT]
```

It logs one line per request to stderr: id, prompt and output tokens, prefill ms, decode tok/s, finish reason (`cancelled` when the client left).

```
curl -s localhost:8080/v1/responses -H 'Content-Type: application/json' \
  -d '{"model": "qwen", "input": "Why is the sky blue?", "reasoning": {"effort": "low"}, "max_output_tokens": 2048}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
r = client.responses.create(model="qwen", instructions="Be brief.", input="Why is the sky blue?")
print(r.output_text)
for event in client.responses.create(model="qwen", input="Hi", stream=True):
    if event.type == "response.output_text.delta":
        print(event.delta, end="", flush=True)
```

| | |
|---|---|
| endpoints | `POST /v1/responses` (plain and `stream: true` SSE), `GET /v1/models`, `GET /health` |
| input | a string, or a list of `system` / `developer` / `user` / `assistant` messages with string or text-part content; `reasoning` items from earlier output; `instructions` |
| reasoning | `reasoning.effort`: `none` / `minimal` (thinking off), `low`, `medium`, `high` (default, the template's `xhigh`). The thinking text comes back as a `reasoning` output item with `reasoning_text` content |
| sampling | `temperature` (default 0.6), `top_p` (default 0.95), `max_output_tokens` (default: the rest of the context; prompt + max must fit) |
| accepted and ignored | `store`, `metadata` (echoed), `user`, `service_tier`, `parallel_tool_calls`, and default values of `tool_choice`, `text`, `truncation` |
| not supported (400) | tools, `previous_response_id` and stored responses, images and files, structured output, logprobs |

All system and developer messages merge into one system message at the start. The chat template is a C++ port of the GGUF's Jinja template for this subset.

Tests (no GPU; source `engine/hrx-env.sh` first, `yah_server` links libhrx):

```
engine/build/yah_serve_test <gguf> [yah_server]   # chat template golden, then yah_server --fake over HTTP
```

It checks the C++ chat template byte for byte against `engine/serve/testdata/chat_template_cases.json`, then starts `yah_server --fake` on a free port and checks the response object and stream event shapes (the required fields of the openai SDK models), event order, deltas, `max_output_tokens`, the 400 error objects and client disconnect. The fixture was frozen from jinja2 on 2026-10-02 against the GGUF's template; regenerating it is not supported in-tree.

## Correctness

| tool | what it checks | when |
|---|---|---|
| hidden md5 of `<prefix>.hidden` | bit identity of the whole forward. References (default set, `ids2048` / `ids8192` prompts): pp2048 `ac36332b6b5092a4`, pp8192 `963b7396625e2333` | every exact rewrite |
| `cmp` of every emitted HAL | byte identity of an emit | every refactor or cleanup of a generator |
| `tools/gemv_check.py <gguf> <work> [kind ...]` | every decode GEMV (format, K) against a float64 gguf-py oracle through `hal_run` | GEMV changes |
| `tools/dattn_check.py <gguf> <work> [T] [pos]` | fp16 decode attention against numpy, scrambled page table | decode attention changes |
| `tools/sample_check.py <gguf> <work> [draws]` | the decode sampler against a numpy model of it (token for token), the model against the exact top-p distribution | sampler changes |
| `tools/dattn_q_check.py <gguf> <work> 8\|4\|kb,vb [T] [pos]` | quantized decode attention and the appends against numpy models of the formats | quantized KV changes |
| `engine/run/gate/` (`gate_run.sh`, `accgate2.py`) | tiered accuracy gate on wikitext windows, all-position logits (`YAH_LOGITS_FROM`): T1 rounding level, T2 quantization level. See `engine/run/gate/README.md` | any numerics change |
| `engine/run/kvq/` (`run_rowstats.sh`, `gate2.py`, `needle_score.py`) | KV format quality at 32K: long-document KL / dPPL with bootstrap CIs and multi-key retrieval, from `YAH_ROWSTATS` row stats of an fp16-KV reference set and a `YAH_KV` candidate set. See `engine/run/kvq/README.md` | KV format work |
| `engine/tests/m0_gate.sh <gguf> [set]` | greedy next token on 3 fixed prompts | after any change, cheap |
| `engine/tests/generate_gate.sh <gguf> [set]` | 20 greedy tokens equal the reference | after any change, cheap |

`hal_run` (`engine/build/hal_run`) dispatches one HAL once with bindings from GGUF tensors or files and writes outputs to files. It refuses any binding smaller than the size you declare. The checkers use it; use it for any new kernel oracle.

Gate notes:

- T1's `kl_mean` saturates at ~4e-7 for any attention change that is not bit-exact (a single rounding change gives the same). Judge such changes on `kl_p999`, flips and PPL.
- The `ids2048` / `ids8192` timing prompts repeat one sentence (10 distinct tokens). They are fine for md5 and timing, useless for accuracy.
- Run a standalone kernel check on inputs from several layers and documents, not only layer 0: an alpha underflow in DeltaNet showed up only at 8K / 32K deep in the model.

## Profiling

| question | tool |
|---|---|
| device time per dispatch in the real pipeline | `HRX_PROFILE_FILE=p.irpf HRX_PROFILE_MODE=dispatch <driver ...>`, then `external/hrx/build/cmake/runtime/src/iree/tools/iree-profile/iree-profile dispatch --dispatch_events --format=jsonl p.irpf`. Costs ~1%. |
| cycles per dispatch (clock-free) | `HRX_PROFILE_MODE=counters HRX_PROFILE_COUNTERS=SQ_BUSY_CYCLES`, then `iree-profile counter --format=jsonl --counter_samples`. Needs TheRock's `libhsa-amd-aqlprofile64` (`/var/lib/lemonade/.cache/lemonade/bin/therock/gfx1151-7.13.0/lib`) first on `LD_LIBRARY_PATH`. Inflates dispatch gaps ~4x. |
| registers, spills, residency, schedule | `loom-compile k.loom --root=@k --target=amdgpu:gfx1151 --format=amdgpu-hsaco --output=k.hsaco --compile-report=details --compile-report-output=k.json`, then `PYTHONPATH=external/hrx/loom/py python3 -m loom.tools.compile_report show\|suggest\|diff k.json`. Run `suggest --include-experimental` before hand-tuning. `allocation.materialized_spill_*` shows real spills even when `spill_count` is 0. |
| instruction counts | `llvm-objdump -d --mcpu=gfx1151` on the hsaco (a `.hal` embeds the ELF from the first `\x7fELF`). Run it with `env -u LD_LIBRARY_PATH`. |
| PMC, occupancy, per-wave ATT | rocprofv3 cannot see HRX dispatches. Run the kernel through `engine/build/loomhip <hsaco> <kernel> <gx> <gy> <block_x> <iters> <buf>...` under `rocprofv3 --pmc "A B C"` (one space-separated list) or `--att --att-library-path <TheRock lib>`. ~400 gfx1151 counters (`SQ_INST_CYCLES_VALU`, `SQ_WAIT_BARRIER`, `SPI_RA_*`) need `ROCPROFILER_METRICS_PATH` pointed at a copy of rocprof-compute's `sdk_config.yaml` renamed `config.yaml`. |
| clocks, temperature, throttling | gpu_metrics v3 (`/sys/class/drm/card1/device/gpu_metrics`), the ryzen_smu PM table, `z13ctl status` for the APU temperature. See hardware.md. |

- loomhip buffer lists must match the kernel's launch signature in order and count; loomhip refuses a count that disagrees with the kernel argument size, but cannot check order. Run it through `gpu_run.sh`.
- ATT stretches the tile GEMM ~1.5x. Use it for shares and attribution, never for absolute idle time; check resource claims with PMC.
- After an ablation, check that the ablated kernel's ISA still contains the work: the compiler deletes loads whose data is never used.
- PC sampling and RGP for compute do not work on this box.

## GPU safety

A bad dispatch on this box does not fault. A read past an allocation reaches unmapped VA, the shader hangs with no page fault, the gfx ring times out, MES fails to reset, and the machine reboots.

Rules:

1. Launch exactly the grid the kernel was compiled for. Loom assumes `workgroup.id` < the launch-config grid and drops index clamps it proves redundant, so a larger grid reads out of bounds. The drivers take grids from `dispatch.txt` / `decode.txt` and refuse others; `gen_gemv.rows_per_wg()` is the single source for GEMV grids. Never compute a grid by hand in a harness.
2. LDS data written by other threads needs a workgroup barrier before the first read, prologues included. (A page table in LDS read before its barrier hung the GPU.)
3. Clamp or validate every index read from memory before it addresses a buffer. The host validates page tables; cache writers clamp page indices into the pool.
4. Clamp a vector load by its own width: `min(addr, N - width)`. A clamp copied from a 4-wide load onto a 2-wide load shifts the last element.
5. Never bind a buffer smaller than the kernel's declared footprint. The prefill emitter runs `tools/footprint_gate.py`; `hal_run` checks declared sizes; `tools/safe_bench.py` and `tools/loom_preflight.py` check the compile report's footprint against the bindings.
6. Rebuild drivers with `engine/build_hrx.sh` after every source change, and do not edit `engine/run/*.cc` during a measuring round (`gpu_run.sh` will then refuse the remaining runs).
7. Run every GPU job through `engine/run/gpu_run.sh`. It snapshots and follows the kernel log into `~/yah-scratch/gpu-<tag>-<time>.dmesg.log` and flags timeout / reset lines.
8. A launched chunk graph waits on NPU done words (~2 s bound each). Never leave it waiting on jobs that will not run: when queueing them fails, release the waits (`NpuSplit::Release`); a failed wait sets a sticky status that ends every later wait at once. A graph of ~250 unreleased waits (before the sticky status) spun for minutes, starved the display ring and needed a reset. Test new NPU host paths CPU-driven first (no GPU kernels), then end to end.

9. New NPU work goes through the safe tools first: the kernels' oracle checks (`split_gemm_check.py`, `npu_rem_check.py` with `REM_LEAD`, `npu_unpack_check.py` with `UNP_DUP`, footprint gates), then the NPU host path CPU-driven: `YAH_NPU_CPU=1 loom_forward_pp ...` records the chunk, launches no GPU work, stores each job's ready word from the host and reports the first job that stalls or fails. Only then end to end. For a stall: `YAH_NPU_CPU_JOBS=list` prints the chunk's jobs, `YAH_NPU_CPU_JOBS=a-b,c-d` runs only those (keep whole family runs: the gate values count per run) to bisect image transitions; `YAH_NPU_CPU_STALL_CMD="python3 tools/npu_aie_status.py > stall.txt"` dumps every column's DMA / core / lock state while the stalled context lives, `YAH_NPU_CPU_HOLD=<job>` runs the command with that job parked at its gate (the healthy reference) and `YAH_NPU_CPU_DONE_CMD` after the last job.
10. Never free memory the GPU or the NPU may still touch. `~LoomNpu` waits for the GPU, then destroys the NPU context, then frees the shared pages; a failed `Enqueue` waits for the launched graph before it throws. With the IOMMU in passthrough an NPU write to a freed page lands in whatever the kernel reused it for; a GPU read of a freed page hangs the shader and MES.

After a hang and reboot, read the previous boot's kernel log:

```
doas journalctl -k -b -1 --no-pager | grep -iE 'amdgpu|ring .*timeout|reset|MES'
```

`/tmp` is wiped by the reboot; keep logs elsewhere.

## Measuring

- One round per candidate. No interleaving, no ABBA, no repeats, no clock pinning.
- Before each timed run, cool the APU to <= 55 C (`z13ctl status`), then wait 1 s (single kernels), 15 s (pp2048) or 30 s (pp8192 and longer).
- For pp8192 and longer, report `SQ_BUSY_CYCLES` next to ms. A run can take a mid-run clock step worth ~7% of wall time; the first pp8192 run after an idle period reads fast. Cycles per dispatch and per device tick show both.
- Compare per-kernel rows, not only totals: run-to-run noise on a single kernel row is up to ~5% in cycles.
- Use real weights and real activations for standalone kernels: constant fills run the same cycles at a higher clock.
- Benchmark long-context decode attention at 32K: smaller standalone cases fit in the 32 MB MALL and read flattered bandwidth.
- A kernel change must be bit-identical (hidden md5) or pass its oracle check before it is timed. A refactor must keep every emitted HAL byte-identical.
- Before an A/B, `cmp` the two variants' HALs: a script that loses its env overrides measures the same kernel twice.
- The tctl limit set with `ryzenadj` resets on reboot. Re-apply it before timing and compare only runs with the same setting.
- When an optimization loses, find out why before dropping it, and record one line in results.md.
- A timing ablation that skips a kernel changes the data the rest of the run computes on, and GEMM speed depends on the data (power under the thermal cap): skipping the qkv / gate unpacks ran every later GEMM ~4.6% faster, pp2048 -48 ms of which ~11 ms was the unpacks. Measure a kernel's cost with the output unchanged (run it a second time into a dummy buffer) and require the reference md5; a skip that changes the md5 is an upper bound.

## Further reading

- HRX demos: [DEVELOPMENT.md](https://github.com/ROCm/hrx-demos/blob/main/DEVELOPMENT.md), [AGENTS.md](https://github.com/ROCm/hrx-demos/blob/main/AGENTS.md); Loom docs in `external/hrx/loom/docs/`.
- [gfx950 Gluon tutorials](https://github.com/ROCm/gfx950-gluon-tutorials), [HIP performance guidelines](https://rocmdocs.amd.com/projects/HIP/en/develop/how-to/performance_guidelines.html).
