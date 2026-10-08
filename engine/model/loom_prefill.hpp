// LoomPrefill: the 64-layer Loom prefill on HRX, one chunk of B tokens at a time.
//
// HAL set: tools/emit_prefill_pp.py; <dir>/dispatch.txt gives the launch geometry of each HAL and marker rows for the
// set's features. Every kernel carries the token dimension in its grid:
//   GEMM family    grid (m_tiles / rowgrp, token_tiles, 1)
//   norm/conv/...  grid (tiles, B) or (B, 1, 1)
//   attention      grid (query-token tiles, head groups), causal over keys 0..token
// A chunked set ("ctx" row) runs every kernel at the chunk size B and sizes the KV pools for the context T.
// Reset() zeroes the recurrent state (conv ring, DeltaNet state); later chunks carry it forward.
// DecoderState() hands the KV pools, page table and recurrent state to a LoomDecoder.
#ifndef YAH_MODEL_LOOM_PREFILL_HPP_
#define YAH_MODEL_LOOM_PREFILL_HPP_

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "core/config.hpp"
#include "core/gguf.hpp"
#include "model/loom_decoder.hpp"
#include "model/loom_runtime.hpp"
#include "model/npu_split.hpp"
#include "model/prefill_calib.hpp"

namespace yah::model {

class LoomPrefill {
 public:
  static constexpr std::uint32_t kHidden = 5120, kFfn = 17408, kAttn = 6144, kQProj = 12288, kKv = 1024;
  static constexpr std::uint32_t kInner = 6144, kQkv = 10240, kTs = 48, kKh = 16, kState = 128;
  static constexpr std::uint32_t kHeads = 24, kKvHeads = 4, kHeadDim = 256, kVocab = 248320;
  static constexpr std::uint32_t kKvRow = kKvHeads * kHeadDim;  // one KV cache row: all KV heads of one token

  // Called after RoPE of attention layer ai in chunk ci, with the layer's f16 K and V offsets in the KV scratch.
  using KvHook = std::function<void(std::uint32_t ai, std::uint32_t ci, std::size_t koff, std::size_t voff)>;

  // tokens: the token count of a one-pass (unchunked) set; ignored for a chunked set, which fixes B.
  LoomPrefill(LoomDevice& gpu, const core::Gguf& gguf, const core::Qwen35Config& cfg, std::string dir,
              hrx_buffer_t weights, std::size_t delta, std::uint32_t tokens = 0)
      : gpu_(gpu), gguf_(gguf), cfg_(cfg), dir_(std::move(dir)), weights_(weights), delta_(delta) {
    LoadDispatch();
    B_ = tokens;
    T_ = tokens;
    if (const auto it = geom_.find("ctx"); it != geom_.end()) {
      B_ = it->second.tokens;
      T_ = it->second.tt;
    }
    if (B_ == 0) throw LoomError("prefill: a one-pass set needs the token count");
    for (std::uint32_t l = 0; l < cfg_.main_block_count(); ++l)
      if (cfg_.IsFullAttention(l)) ++full_;
    LoadTables();
    AllocateBuffers();
    LoadExecutables();
  }

  [[nodiscard]] std::uint32_t chunk() const { return B_; }
  [[nodiscard]] std::uint32_t context() const { return T_; }
  [[nodiscard]] bool paged() const { return kv_paged_; }
  // KV bits of this set: (16, 16) fp16, else 8 or 4 per side.
  [[nodiscard]] std::pair<std::uint32_t, std::uint32_t> kv_bits() const {
    return {attn_kq4_ ? 4u : attn_kq8_ ? 8u : 16u, attn_vq4_ ? 4u : attn_vq8_ ? 8u : 16u};
  }
  LoomBuffer& hidden() { return *hidden_; }

  // Buffer sizes of the NPU column split this set carries (dispatch.txt "npusplit_<site>", "npubytes_<K>"); zero: none.
  // A job is the NPU work behind one cut: the DeltaNet qkv + gate, or one other site; the buffers fit the largest
  // (W twice).
  [[nodiscard]] NpuPlan npu_plan() const {
    NpuPlan p;
    const auto need = [&](const std::string& site) {
      const auto it = geom_.find("npusplit_" + site);
      NpuPlan q;
      if (it == geom_.end()) return q;
      const std::size_t calls = it->second.tokens / NpuCallRows(), mats = site == "ffn" ? 2 : 1;
      for (const auto& [k0, K] : NpuChunks(site)) {
        const NpuBytes nb = NpuK(K);
        q.a_bytes += nb.a, q.w_bytes += mats * calls * nb.w, q.c_bytes += mats * calls * nb.c;
      }
      return q;
    };
    const NpuPlan qkv = need("qkv"), gate = need("gate");
    for (const NpuPlan& q : {NpuPlan{std::max(qkv.a_bytes, gate.a_bytes), qkv.w_bytes + gate.w_bytes,
                                     qkv.c_bytes + gate.c_bytes},
                             need("q"), need("out"), need("down"), need("ffn")})
      p.a_bytes = std::max(p.a_bytes, q.a_bytes), p.w_bytes = std::max(p.w_bytes, q.w_bytes),
      p.c_bytes = std::max(p.c_bytes, q.c_bytes);
    p.w_bytes *= 2;   // two slots: a job's weights are decoded while the previous job runs (NpuDecode)
    if (const auto it = geom_.find("npugate"); it != geom_.end())
      p.gate_calls = it->second.rowgrp, p.gate_record = it->second.tt;
    return p;
  }
  // Split the set's NPU sites of full chunks with the NPU from now on: it computes their trailing npusplit rows.
  // npu (built from npu_plan()) outlives this object.
  void EnableNpu(NpuSplit* npu) {
    for (const auto& [name, g] : geom_)
      if (name.rfind("npusplit_", 0) == 0) npu_rows_[name.substr(9)] = g.tokens;
    if (!npu || npu_rows_.empty()) throw LoomError("prefill: the set has no NPU split");
    if (!geom_.count("npu_flag_wait.hal") || !geom_.count("npugate"))
      throw LoomError("prefill: the set has no gated NPU handoffs (npu_flag_wait.hal, npugate; re-emit it)");
    npu_ = npu;
    std::size_t rem = 0;
    for (const auto& [site, rows] : npu_rows_)
      if (NpuRem(site)) rem = std::max<std::size_t>(rem, std::size_t{B_} * rows * 4);
    if (rem) npurem_ = &Alloc(rem);
    // Bind every NPU call of a full chunk now (a pass of the layers that dispatches nothing): a first bind loads the
    // image's storage and patches it, milliseconds per call.
    LoomBuffer *h = hidden_, *h2 = hidden2_;
    const std::uint32_t n = n_;
    n_ = B_, npu_on_ = true, npu_planning_ = true;
    planned_dq_.clear();
    npu_job_ = 0;
    NewGraph();
    RecordLayers(0, {});
    chunk_graph_.reset();
    graph_ = nullptr, n_ = n, npu_on_ = false, npu_planning_ = false;
    hidden_ = h, hidden2_ = h2;
  }

  // Zero the recurrent state before a new sequence. KV rows need no reset: attention reads only keys <= the query.
  void Reset() {
    Zero(*conv_state_);
    Zero(*state_);
  }

  // Stage ids[0..n) for the next RunLayers, whose first kernel (yah_embed) writes their token_embd rows into hidden().
  // n < B pads the chunk with the last token.
  // The GEMMs (~92% of the time) run only the token tiles that hold real tokens; every other kernel runs
  // the whole chunk, so padding rows next to the real ones keep their padded-path values. Shrinking the norms as well
  // changed the last token's logits by ~1e-4 when it sat alone in its 16-token group (cause not found yet). Padding tokens leave the recurrent state
  // unchanged and attention is causal, so the real tokens' results and the state carried forward are exact.
  // The ids and the token count go to the GPU in stream order (stream updates): the host never waits here.
  void Embed(const std::uint32_t* ids, std::uint32_t n) {
    if (n == 0 || n > B_) throw LoomError("prefill: chunk token count out of range");
    for (std::uint32_t t = 0; t < B_; ++t) {
      host_ids_[t] = ids[std::min(t, n - 1)];
      if (host_ids_[t] >= kVocab)
        throw LoomError("token id " + std::to_string(host_ids_[t]) + " is outside the vocabulary");
    }
    gpu_.Update(*ids_, host_ids_.data(), host_ids_.size() * 4);
    const std::int32_t valid = static_cast<std::int32_t>(n);
    gpu_.Update(*valid_, &valid, 4);
    n_ = n;
  }

  // RunLayers' keep: which rows of hidden() the caller reads after the last layer (Head). kAllRows: every row; kNoRows:
  // none, so the last layer's tail (attention / o-proj or postnorm / ssm_out, and the FFN) is skipped (its K / V and
  // recurrent state are still written); a row index: the tail's GEMMs run only the token tile that holds it.
  static constexpr std::int64_t kAllRows = -1, kNoRows = -2;

  // Enqueue the 64 layers for chunk ci (absolute positions ci * B ..), on the hidden() rows from Embed().
  void RunLayers(std::uint32_t ci, const KvHook& hook = {}, std::int64_t keep = kAllRows) {
    keep_rows_ = keep;
    if (std::size_t{ci + 1} * B_ > T_) throw LoomError("prefill: chunk past the emitted context");
    if (n_ == 0) throw LoomError("prefill: RunLayers before Embed");
    // The layers go into one graph, so kernels with no data between them (the input projections of a layer, the DeltaNet
    // gate projection and the conv / DeltaNet chain) can run at the same time.
    if (DumpOn()) {   // YAH_DUMP_ACT: no graph, so RunNorm can synchronize and read its output back
      if (!df_planned_) PlanDecodeFree(ci);
      nodes_.clear();
      df_on_ = false;
      RecordLayers(ci, hook);
      gpu_.Synchronize();
      return;
    }
    if (!df_planned_) PlanDecodeFree(ci);
    // NPU split: full chunks only, its handoffs graph nodes on flag words (NpuEnqueue / NpuJoin)
    npu_on_ = npu_ && n_ == B_;
    flagged_.clear();
    predecoded_.clear();
    npu_job_ = 0;
    NewGraph();
    nodes_.clear();
    const bool calibrate = !npu_on_ && calib_ && calib_->BeginChunk(n_);
    df_on_ = !npu_on_ && !df_plan_.empty() && n_ > kDfMinTokens;
    df_k_ = 0;
    if (df_on_) DispatchDequant(0);
    RecordLayers(ci, hook);
    graph_ = nullptr;
    if (calibrate && !gpu_.profiling()) {
      try {
        gpu_.ProfileBegin([this](const hrx_profile_dispatch_t* e, std::size_t n, std::uint64_t) {
          session_.insert(session_.end(), e, e + n);
        });
      } catch (const LoomError& e) {  // e.g. HRX_PROFILE_FILE holds the device's profiler
        std::fprintf(stderr, "prefill calibration off: %s\n", e.what());
        calib_.reset();
      }
    }
    // every graph of the session, to match them in order (one graph per chunk; NPU chunks are not matched)
    if (gpu_.profiling() && flagged_.empty()) session_chunks_.push_back(nodes_);
    chunk_graph_->Launch();
    std::vector<NpuSplit::Queued> jobs;
    jobs.swap(flagged_);
    if (!jobs.empty()) npu_->Enqueue(jobs);   // on a failure it releases the graph's waits for the jobs left
  }

  // The last RunLayers graph's dispatches in node order (= the order of their dispatch timestamps' command indices).
  struct Node {
    std::array<std::uint32_t, 3> grid;
    std::string name;
    int tag = -1;  // the calibration's tag (PrefillCalib::Choose)
  };
  [[nodiscard]] const std::vector<Node>& nodes() const { return nodes_; }

  // The 64 layers of chunk ci, in dispatch order.
  void RecordLayers(std::uint32_t ci, const KvHook& hook) {
    Dispatch(Exe("embed.hal"), "yah_embed", GeomOf("embed.hal").tt, 1, 1, kHidden / 16, 1, 1,
             {TRef(*Find("token_embd.weight")), Ref(*ids_), Ref(*hidden_)}, 4);
    for (std::uint32_t l = 0; l < cfg_.main_block_count(); ++l) {
      const std::string pre = "blk." + std::to_string(l) + ".";
      tail_ = l + 1 == cfg_.main_block_count() ? keep_rows_ : kAllRows;
      if (cfg_.IsFullAttention(l)) {
        const bool q_af = Af("gemm_kqg", pre + "attn_q.weight");
        const bool k_af = Af("gemm_kstore", pre + "attn_k.weight");
        const bool v_af = Af("gemm_kstore", pre + "attn_v.weight");
        const bool q_npu = NpuRows("q") && !NpuHal(KqgHal(pre + "attn_q.weight", q_af)).empty();
        RunNorm(pre + "attn_norm.weight",
                q_af && k_af && v_af   ? NormOut::kTiled
                : q_af || k_af || v_af ? NormOut::kBoth
                                       : NormOut::kRow,
                q_npu ? "q" : nullptr);
        RunAttention(l, ci, pre, hook, q_af, k_af, v_af);
      } else {
        // afrag qkv / gate read the fragment-major copy; alpha / beta keep the row-major one
        const bool qkv_af = Af("gemm_kstore", pre + "attn_qkv.weight");
        const bool gate_af = Af("gemm_kstore", pre + "attn_gate.weight");
        const bool qkv_npu = NpuRows("qkv") && NpuRows("gate") &&
                             !NpuHal(KstoreHal(pre + "attn_qkv.weight", qkv_af)).empty() &&
                             !NpuHal(KstoreHal(pre + "attn_gate.weight", gate_af)).empty();
        RunNorm(pre + "attn_norm.weight", qkv_af || gate_af ? NormOut::kBoth : NormOut::kRow, qkv_npu ? "qkv" : nullptr);
        RunDeltaNet(l, pre, qkv_af, gate_af);
      }
      if (tail_ == kNoRows) continue;
      trim_row_ = tail_;
      RunFfn(pre);
      trim_row_ = kAllRows;
    }
    tail_ = kAllRows;
  }


  // Calibrate the GEMM variants while serving (model/prefill_calib.hpp), state in path. The caller runs Collect() after
  // the chunks of a prompt completed (Synchronize).
  void EnableCalibration(const std::string& path) {
    std::map<std::string, std::uint32_t> hals;
    for (const auto& [name, g] : geom_) hals[name] = g.tokens;
    calib_ = std::make_unique<PrefillCalib>(hals, path);
  }
  // Ends the profile session, if any, and hands each chunk's dispatch timestamps to the calibration.
  void Collect() {
    if (!gpu_.profiling()) return;
    gpu_.ProfileEnd();
    // Each command buffer of the session in launch order; a chunk's graph is the next one with its node count (stream
    // dispatches, e.g. the output head, come in command buffers of their own).
    std::map<std::uint64_t, std::vector<hrx_profile_dispatch_t>> buffers;
    for (const auto& e : session_) buffers[e.command_buffer_id].push_back(e);
    const auto before = calib_->Progress();
    auto chunk = session_chunks_.begin();
    for (auto& [id, ev] : buffers) {
      if (chunk == session_chunks_.end()) break;
      if (ev.size() != chunk->size()) continue;
      std::sort(ev.begin(), ev.end(), [](const auto& a, const auto& b) { return a.command_index < b.command_index; });
      std::vector<int> tags;
      std::vector<std::array<std::uint32_t, 3>> grids;
      for (const Node& nd : *chunk++) tags.push_back(nd.tag), grids.push_back(nd.grid);
      calib_->EndChunk(tags, grids, ev);
    }
    calib_->EndSession();
    const auto after = calib_->Progress();
    if (after != before)
      std::fprintf(stderr, "prefill calibration: %d GEMM buckets settled, %d open\n", after.first, after.second);
    session_.clear();
    session_chunks_.clear();
  }
  [[nodiscard]] const PrefillCalib* calibration() const { return calib_.get(); }

  // Final norm + output head of hidden() row `row` (this chunk) into dst (kVocab f32).
  void Head(std::uint32_t row, const hrx_buffer_ref_t& dst) {
    const auto* onw = Find("output_norm.weight");
    const auto* ow = Find("output.weight");
    Dispatch(Exe("rmsnorm.hal"), "yah_rmsnorm", 1, 1, 1, 32, 1, 1,
             {{hidden_->handle, std::size_t{row} * kHidden * 4, std::size_t{kHidden} * 4}, TRef(*onw), Ref(*normed_)});
    Dispatch(Exe("gemv.hal"), "yah_gemv_q6k", kVocab, 1, 1, 32, 1, 1, {TRef(*ow), Ref(*normed_), dst});
  }
  // Argmax of kVocab f32 logits into dst (one u32).
  void Argmax(const hrx_buffer_ref_t& logits, const hrx_buffer_ref_t& dst) {
    Dispatch(Exe("argmax.hal"), "yah_argmax", 1, 1, 1, 32, 1, 1, {logits, dst});
  }

  // Quantized V, from a KvHook of the last chunk: copy rows r0..r0+cnt of the layer's f16 V (voff in the KV scratch) into a
  // decoder's open tile dst. params: device i32 (r0, cnt).
  void SeedOpenTile(std::size_t voff, const hrx_buffer_ref_t& params, const hrx_buffer_ref_t& dst) {
    Dispatch(Exe("vseed.hal"), "yah_vseed", 4, 1, 1, 256, 1, 1,
             {{kv16_->handle, voff, std::size_t{B_} * kKvRow * 2}, params, dst});
  }

  // The KV pools, page table and recurrent state, for a LoomDecoder with the same context and KV format.
  [[nodiscard]] LoomDecoderState DecoderState() const {
    if (!kv_paged_) throw LoomError("prefill: decode needs the paged KV layout (the default set)");
    LoomDecoderState st;
    for (std::uint32_t ai = 0; ai < full_; ++ai) {
      if (attn_kq8_) {
        st.kq.push_back({kq8buf_->handle, std::size_t{ai} * kq_bytes_, kq_bytes_});
        st.ks.push_back({ksbuf_->handle, std::size_t{ai} * ks_bytes_, ks_bytes_});
        st.km.push_back({kmbuf_->handle, std::size_t{ai} * 4096, 4096});
      }
      if (attn_vqt_) {
        st.vq.push_back({vqbuf_->handle, std::size_t{ai} * vq_bytes_, vq_bytes_});
        st.vs.push_back({vqsbuf_->handle, std::size_t{ai} * vqs_bytes_, vqs_bytes_});
      }
      if (!attn_kq8_) st.kpool.push_back({kpool_->handle, std::size_t{ai} * pool_bytes_, pool_bytes_});
      if (!attn_vqt_) st.vtpool.push_back({vtpool_->handle, std::size_t{ai} * pool_bytes_, pool_bytes_});
    }
    st.ptab = ptab_ref_;
    st.convstate = conv_state_->handle;
    st.dstate = state_->handle;
    return st;
  }
  [[nodiscard]] std::uint32_t pool_rows() const { return pages_ * 256; }

  // A slot: the sequence state after `chunks` chunks (KV pools, page table, conv rings, DeltaNet states), raw. It loads
  // only into a prefill of the same set (context, chunk size, KV format); LoadState returns the chunks it covers, so the
  // caller continues at that chunk index (marginal prefill at depth without re-running the prefix).
  void SaveState(const std::string& path, std::uint32_t chunks) {
    gpu_.Synchronize();
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw LoomError("prefill: cannot write the slot " + path);
    const auto bufs = StateBuffers();
    const std::uint64_t head[5] = {kSlotMagic, T_, B_, chunks, bufs.size()};
    bool ok = std::fwrite(head, sizeof head, 1, f) == 1;
    std::vector<std::uint8_t> host(std::size_t{64} << 20);
    for (const LoomBuffer* b : bufs) {
      const std::uint64_t n = b->size;
      ok = ok && std::fwrite(&n, 8, 1, f) == 1;
      for (std::size_t o = 0; ok && o < b->size; o += host.size()) {
        const std::size_t k = std::min(host.size(), b->size - o);
        gpu_.D2H(*b, host.data(), k, o);
        ok = std::fwrite(host.data(), 1, k, f) == k;
      }
    }
    if (std::fclose(f) != 0 || !ok) throw LoomError("prefill: short write to the slot " + path);
  }
  std::uint32_t LoadState(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw LoomError("prefill: cannot open the slot " + path);
    const auto bufs = StateBuffers();
    std::uint64_t head[5] = {};
    bool ok = std::fread(head, sizeof head, 1, f) == 1 && head[0] == kSlotMagic && head[1] == T_ && head[2] == B_ &&
              head[4] == bufs.size();
    std::vector<std::uint8_t> host(std::size_t{64} << 20);
    for (const LoomBuffer* b : bufs) {
      std::uint64_t n = 0;
      ok = ok && std::fread(&n, 8, 1, f) == 1 && n == b->size;
      for (std::size_t o = 0; ok && o < b->size; o += host.size()) {
        const std::size_t k = std::min(host.size(), b->size - o);
        ok = std::fread(host.data(), 1, k, f) == k;
        if (ok) gpu_.H2D(*b, host.data(), k, o);
      }
    }
    std::fclose(f);
    if (!ok) throw LoomError("prefill: the slot " + path + " does not match this set (context, chunk, KV format)");
    gpu_.Synchronize();
    return static_cast<std::uint32_t>(head[3]);
  }

 private:
  static constexpr std::uint64_t kSlotMagic = 0x31544f4c53484159ull;  // "YAHSLOT1"
  // Paged sets keep the whole sequence state in these (the f16 K/V buffer is per-chunk scratch there).
  [[nodiscard]] std::vector<const LoomBuffer*> StateBuffers() const {
    if (!kv_paged_ || !kv16_scratch_) throw LoomError("prefill: slots need the paged KV layout (the default set)");
    return {ptab_, kq8buf_, ksbuf_, kmbuf_, vqbuf_, vqsbuf_, kpool_, vtpool_, conv_state_, state_};
  }
  struct Fmt {
    const char* name;
    std::uint32_t qk;
  };
  static bool FmtOf(std::uint32_t type, Fmt* out) {
    switch (type) {
      case 12: *out = {"q4k", 256}; return true;
      case 13: *out = {"q5k", 256}; return true;
      case 14: *out = {"q6k", 256}; return true;
      case 11: *out = {"q3k", 256}; return true;
      case 23: *out = {"iq4xs", 256}; return true;
      case 21: *out = {"iq3s", 256}; return true;
      case 18: *out = {"iq3xxs", 256}; return true;
      case 20: *out = {"iq4nl", 32}; return true;
      case 17: *out = {"iq2xs", 256}; return true;
      case 8: *out = {"q8_0", 32}; return true;
      case 16: *out = {"iq2xxs", 256}; return true;
      case 10: *out = {"q2k", 256}; return true;
      default: return false;
    }
  }
  // dispatch.txt: "<hal> <tokens per workgroup> <16-row tiles per workgroup> <token tiles>" per HAL.
  // Read grids from here; never mirror the emitter's arithmetic: a wrong grid silently skips rows.
  struct Geom {
    std::uint32_t tokens;
    std::uint32_t rowgrp;
    std::uint32_t tt;
  };

  void LoadDispatch() {
    std::ifstream f(dir_ + "/dispatch.txt");
    if (!f) throw LoomError("cannot open " + dir_ + "/dispatch.txt");
    std::string name;
    Geom g{};
    while (f >> name >> g.tokens >> g.rowgrp >> g.tt) geom_[name] = g;
  }
  // Refuses a HAL emitted for another token count: it would compute a silent subset.
  Geom GeomOf(const std::string& hal) const {
    const auto it = geom_.find(hal);
    if (it == geom_.end()) throw LoomError("dispatch.txt has no row for " + hal);
    const Geom g = it->second;
    // A token tile need not divide the chunk: the last one is masked in the kernel.
    if (g.tokens == 0 || g.rowgrp == 0 || (B_ + g.tokens - 1) / g.tokens != g.tt)
      throw LoomError(hal + ": emitted for another token count");
    return g;
  }

  static void ReadFile(const std::string& path, void* dst, std::size_t bytes) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw LoomError("cannot open " + path);
    const bool ok = std::fread(dst, 1, bytes, f) == bytes;
    std::fclose(f);
    if (!ok) throw LoomError("short read: " + path);
  }
  static hrx_buffer_ref_t Ref(const LoomBuffer& b) { return {b.handle, 0, b.size}; }
  hrx_buffer_ref_t TRef(const core::TensorInfo& t) const {
    return {weights_, delta_ + static_cast<std::size_t>(t.offset), static_cast<std::size_t>(t.bytes)};
  }
  const core::TensorInfo* Find(const std::string& name) const {
    const auto* t = gguf_.Find(name);
    if (!t) throw LoomError("tensor not found: " + name);
    return t;
  }
  // Zero-filled: a partial chunk leaves rows past its last token tile unwritten, and the full-chunk kernels (conv,
  // DeltaNet) read them. They must be finite: the DeltaNet masks a padding token by multiplying it with 0.
  LoomBuffer& Alloc(std::size_t bytes) {
    keep_.push_back(gpu_.Allocate(bytes));
    Zero(keep_.back());
    return keep_.back();
  }
  // Waits: the fill is stream work, while uploads (tables, Embed) are synchronous copies that would otherwise land first
  // and be zeroed.
  void Zero(LoomBuffer& b) {
    gpu_.Fill(b, 0);
    gpu_.Synchronize();
  }
  // GEMM token tiles covering this chunk's real tokens.
  std::uint32_t TokenTiles(const Geom& g) const { return (n_ + g.tokens - 1) / g.tokens; }
  // The last layer's tail on one row (trim_row_ >= 0): a GEMM dispatches only the token tile that holds the row, its
  // token-major bindings b[first..] moved to that tile (row_bytes: bytes per token row, 0 for the others). Each output row
  // depends only on its own input row, so the kept row is the same; the other rows are left unwritten. The
  // fragment-major layout groups 16 rows per tile row, so a tile start (a multiple of 16) has the same byte offset.
  // Returns the token tiles to launch.
  std::uint32_t Trim(std::vector<hrx_buffer_ref_t>& b, std::size_t first, std::initializer_list<std::size_t> row_bytes,
                     const Geom& g) const {
    if (trim_row_ < 0) return TokenTiles(g);
    const std::size_t t0 = static_cast<std::size_t>(trim_row_) / g.tokens * g.tokens;
    std::size_t i = first;
    for (const std::size_t rb : row_bytes) {
      if (rb) b[i].offset += t0 * rb, b[i].length -= t0 * rb;
      ++i;
    }
    return 1;
  }
  LoomExecutable& Exe(const std::string& hal) {
    auto it = exes_.find(hal);
    if (it == exes_.end()) it = exes_.emplace(hal, gpu_.Load(dir_ + "/" + hal)).first;
    return it->second;
  }
  // The export's own workgroup size wins; sx is only the fallback for metadata without one.
  // writes: the bindings the kernel may write (bit i = binding i), for the graph's dependencies; by default all of them.
  // also_writes: ranges the dispatch stands for as written though it does not bind them (LoomGraph::Dispatch).
  void Dispatch(const LoomExecutable& exe, const char* name, std::uint32_t gx, std::uint32_t gy, std::uint32_t gz,
                std::uint32_t sx, std::uint32_t sy, std::uint32_t sz, const std::vector<hrx_buffer_ref_t>& b,
                std::uint64_t writes = ~std::uint64_t{0}, const std::vector<hrx_buffer_ref_t>* also_writes = nullptr) {
    if (df_planning_ || npu_planning_) return;
    const std::uint32_t ordinal = exe.OrdinalOrZero(name);
    const std::uint32_t ws = exe.WorkgroupSize(ordinal);
    const hrx_dispatch_config_t config = LoomDevice::Config(gx, gy, gz, ws ? ws : sx, sy, sz);
    if (graph_) {
      graph_->Dispatch(exe, ordinal, config, b.data(), b.size(), writes, df_after_.empty() ? nullptr : &df_after_,
                       also_writes);
      nodes_.push_back({{gx, gy, gz}, name, pending_tag_});
      pending_tag_ = -1;
    } else
      gpu_.Dispatch(exe, ordinal, config, nullptr, 0, b.data(), b.size());
  }
  // A GEMM writes only its outputs; the hand-written Q2_K GEMM also stages its accumulators in ostage.
  std::uint64_t GemmWrites(const std::vector<hrx_buffer_ref_t>& b, const Fmt& f,
                           std::initializer_list<const LoomBuffer*> outs) const {
    std::uint64_t m = 0;
    for (std::size_t i = 0; i < b.size(); ++i) {
      for (const LoomBuffer* o : outs)
        if (b[i].buffer == o->handle) m |= std::uint64_t{1} << i;
      if (std::string(f.name) == "q2k" && b[i].buffer == ostage_->handle) m |= std::uint64_t{1} << i;
    }
    return m;
  }
  // A per-chunk HAL: chunk 0 is "<stem>.hal", chunk c "<stem>_c<c>.hal" (start_pos is compiled in).
  LoomExecutable& ChunkExe(const std::string& stem, std::uint32_t c) {
    return Exe(stem + (c ? "_c" + std::to_string(c) : std::string()) + ".hal");
  }

  void LoadTables() {
    const auto table = [&](const char* file, std::size_t bytes) -> LoomBuffer& {
      LoomBuffer& b = Alloc(bytes);
      std::vector<std::uint8_t> v(bytes);
      ReadFile(dir_ + "/" + file, v.data(), bytes);
      gpu_.H2D(b, v.data(), bytes);
      return b;
    };
    grid_iq3s_ = &table("grid_iq3s.bin", 512 * 4);
    grid_iq3xxs_ = &table("grid_iq3xxs.bin", 256 * 4);
    grid_iq2xxs_ = &table("grid_iq2xxs.bin", 512 * 4);
    grid_iq2xs_ = &table("grid_iq2xs.bin", 1024 * 4);
    ksigns_ = &table("ksigns_iq2xxs.bin", 128);
  }

  void AllocateBuffers() {
    const std::size_t B = B_;
    // Activations are token major, [token][row], with row stride = the K extent of the GEMM that reads them.
    kv_cache_ = std::size_t{T_} * kKvRow;
    hidden_ = &Alloc(B * kHidden * 4);
    reszero_ = &Alloc(B * kHidden * 4);
    sumout_ = &Alloc(B * kHidden * 4);
    scratch_ = &Alloc(B * kFfn * 2);
    qkv_ = &Alloc(B * kQProj * 4);
    gate_ = &Alloc(B * kInner * 4);
    alpha_ = &Alloc(B * kTs * 4);
    beta_ = &Alloc(B * kTs * 4);
    q_ = &Alloc(B * kAttn * 4);
    // q16 sets: RoPE writes the attention's f16(q / 16) here instead of the f32 query in place
    q16_ = geom_.count("q16") ? &Alloc(B * kAttn * 2) : q_;
    // kv4 sets at depth (dispatch.txt "qrot <first chunk>"): yah_qrot writes the H256-rotated f16 Q here for the attention
    if (const auto qr = geom_.find("qrot"); qr != geom_.end()) qr16_ = &Alloc(B * kAttn * 2), qrot_first_ = qr->second.tokens;
    kbuf_ = &Alloc(B * kKv * 4);
    vbuf_ = &Alloc(B * kKv * 4);
    raw_ = &Alloc(B * kInner * 4);
    // DeltaNet as two head ranges (dispatch.txt "dnsplit": heads [0, a) and [a, a + b)); the second writes raw2_
    if (const auto ds = geom_.find("dnsplit"); ds != geom_.end()) {
      dnsplit_a_ = ds->second.rowgrp, dnsplit_b_ = ds->second.tt;
      if (dnsplit_a_ + dnsplit_b_ != kTs || (dnsplit_a_ * B) % 8 || (dnsplit_b_ * B) % 8)
        throw LoomError("dispatch.txt dnsplit does not cover the DeltaNet heads");
      raw2_ = &Alloc(B * kInner * 4);
    }
    conv_out_ = &Alloc(B * kQkv * 4);
    kqbuf_ = &Alloc(B * kKh * 3 * 4);
    ab_ = &Alloc(B * kTs * 2 * 4);
    conv_state_ = &Alloc(std::size_t{48} * kQkv * 4 * 4);
    state_ = &Alloc(std::size_t{48} * kTs * kState * kState * 4);
    // f16 K/V cache: [K | V] x attention layers x context. With "kv16_scratch" (paged or quantized KV) it holds one
    // layer's current chunk only: RoPE writes it and the paged writers or quantizers read it before the next layer.
    kv16_scratch_ = geom_.count("kv16_scratch") != 0;
    kv16_layer_ = kv16_scratch_ ? B * kKvRow * 2 : kv_cache_ * 2;
    kv16_ = &Alloc(kv16_scratch_ ? 2 * kv16_layer_ : std::size_t{2} * full_ * kv_cache_ * 2);
    kc32_ = &Alloc(kv_cache_ * 4);
    vc32_ = &Alloc(kv_cache_ * 4);
    lse_ = &Alloc(B * kHeads * 4);
    eps_ = &Alloc(4);
    ffnup_ = &Alloc(B * kFfn * 2);
    // the FFN norm's fragment-major copy for the afrag GEMMs (norm_t.hal), only in sets that have them
    bool any_af = false;
    for (const auto& [name, g] : geom_) any_af = any_af || name.size() > 7 && name.compare(name.size() - 7, 7, ".af.hal") == 0;
    if (any_af) normt_ = &Alloc(B * kHidden * 2);
    gateffn_ = &Alloc(B * kFfn * 4);
    // Per-workgroup weight staging (ABI only: the tile GEMMs stage in LDS) and epilogue scratch sized for the widest
    // GEMM, which the compiler declares over the whole [m_rows][tokens] tile.
    uwstage_ = &Alloc(std::size_t{kFfn} * 16 * 2);
    wstage_ = &Alloc(std::size_t{kFfn} * 16 * 2);
    ostage_ = &Alloc(std::size_t{kFfn} * B * 4);
    partial_ = &Alloc(B * kHidden * 4);
    hidden2_ = &Alloc(B * kHidden * 4);
    normed_ = &Alloc(std::size_t{kHidden} * 4);
    valid_ = &Alloc(4);
    ids_ = &Alloc(B * 4);
    host_ids_.resize(B);
    const float epsv = 1.0e-6f;
    gpu_.H2D(*eps_, &epsv, 4);
    Zero(*conv_state_);
    Zero(*state_);
    Zero(*kv16_);
    Zero(*reszero_);

    // "kv_paged": 256-token pages, one page table per sequence (logical -> physical page, shared by all layers).
    kv_paged_ = geom_.count("kv_paged") != 0;
    pages_ = (T_ + 255) / 256;
    ptab_ = &Alloc(std::size_t{kv_paged_ ? pages_ : 1} * 4);
    if (kv_paged_) {
      std::vector<std::int32_t> pages(pages_);
      for (std::uint32_t i = 0; i < pages_; ++i) pages[i] = static_cast<std::int32_t>(i);
      // Attention does not clamp page table entries: validate them here.
      for (std::int32_t pg : pages)
        if (pg < 0 || pg >= static_cast<std::int32_t>(pages_)) throw LoomError("page table entry out of range");
      gpu_.H2D(*ptab_, pages.data(), pages.size() * 4);
    }
    ptab_ref_ = {ptab_->handle, 0, std::size_t{kv_paged_ ? pages_ : 1} * 4};
    // Unpaged set with a "vtrans.hal" row: attention reads V as [kv head][16-key tile][dim][16] f16.
    vtrans_ = !kv_paged_ && geom_.count("vtrans.hal");
    vt_bytes_ = std::size_t{(T_ + 15) / 16 * 16} * kKvRow * 2;
    vt16_ = &Alloc(vtrans_ ? vt_bytes_ : 4);
    // Quantized K: "attn_kq8" int8 (kv8a16), "attn_kq4" H256 + asymmetric int4 (kv4a16; kernel yah_kq4 in kq8.hal).
    // Per attention layer: codes [T][1024 or 512 B], scales [T][8 or 32] dwords, the channel mean of the first chunk.
    attn_kq4_ = geom_.count("attn_kq4") != 0;
    attn_kq8_ = geom_.count("attn_kq8") != 0 || attn_kq4_;
    ks_bytes_ = std::size_t{T_} * (attn_kq4_ ? 32 : 8) * 4;
    kq_bytes_ = attn_kq4_ ? kv_cache_ / 2 : kv_cache_;
    kq8buf_ = &Alloc(attn_kq8_ ? std::size_t{full_} * kq_bytes_ : 4);
    ksbuf_ = &Alloc(attn_kq8_ ? std::size_t{full_} * ks_bytes_ : 4);
    kmbuf_ = &Alloc(attn_kq8_ ? std::size_t{full_} * 4096 : 4);
    // Quantized V^T: "attn_vq8" bytes / "attn_vq4" nibbles per channel per 16-key tile, plus (S, C') per tile.
    attn_vq4_ = geom_.count("attn_vq4") != 0;
    attn_vq8_ = geom_.count("attn_vq8") != 0;
    attn_vqt_ = attn_vq4_ || attn_vq8_;
    vq_bytes_ = attn_vq8_ ? vt_bytes_ / 2 : vt_bytes_ / 4;
    vqs_bytes_ = vt_bytes_ / 8;
    vqbuf_ = &Alloc(attn_vqt_ ? std::size_t{full_} * vq_bytes_ : 4);
    vqsbuf_ = &Alloc(attn_vqt_ ? std::size_t{full_} * vqs_bytes_ : 4);
    // Paged fp16 pools: per layer, K rows / V^T tiles of the whole context.
    paged_f16k_ = kv_paged_ && !attn_kq8_;
    paged_f16v_ = kv_paged_ && !attn_vqt_;
    // "rope_kpaged": RoPE writes the fp16 K rows straight into the paged pool.
    rope_kpaged_ = geom_.count("rope_kpaged") != 0;
    if (paged_f16k_ && !rope_kpaged_) throw LoomError("paged fp16 K needs a rope_kpaged HAL set (re-emit)");
    pool_bytes_ = std::size_t{pages_} * 256 * kKvRow * 2;
    kpool_ = &Alloc(paged_f16k_ ? std::size_t{full_} * pool_bytes_ : 4);
    vtpool_ = &Alloc(paged_f16v_ ? std::size_t{full_} * pool_bytes_ : 4);
    // kv4 / kv8 sets (dispatch.txt "kvdeq"): yah_kdeq / yah_vdeq decode the layer's quantized pools into one f16 pool pair (reused
    // layer to layer) and the attention runs the fp16 kernel on it, as llama.cpp's prefill flash attention does
    if (geom_.count("kvdeq")) {
      if (!attn_kq8_ || !attn_vqt_ || !kv_paged_) throw LoomError("dispatch.txt kvdeq needs the paged kv4 pools");
      kdeq_ = &Alloc(pool_bytes_);
      vdeq_ = &Alloc(pool_bytes_);
    }
  }

  void LoadExecutables() {
    // The attention HAL stores f16 straight into the o-projection input ("attn_f16out").
    if (!geom_.count("attn_f16out")) throw LoomError("dispatch.txt lacks attn_f16out (re-emit the set)");
    const auto dn = geom_.find("rowsplit.hal");
    if (dn == geom_.end() || !dn->second.rowgrp)
      throw LoomError("dispatch.txt has no rowsplit.hal row group (re-emit the set)");
    dn_rowgrp_ = dn->second.rowgrp;
    const auto at = geom_.find("wmma.hal");
    if (at == geom_.end() || !at->second.rowgrp || !at->second.tokens)
      throw LoomError("dispatch.txt has no wmma.hal geometry (re-emit the set)");
    attn_hpw_ = at->second.rowgrp;
    attn_tpw_ = at->second.tokens;
    if (kHeads % attn_hpw_) throw LoomError("wmma.hal heads per workgroup does not divide the heads");
    // Loom drops bounds clamps it proves from the launch contract: extra workgroups read unmapped VA and hang.
    if (at->second.tt && (B_ + attn_tpw_ - 1) / attn_tpw_ != at->second.tt)
      throw LoomError("wmma.hal: grid x does not match the emitted token tiles");
    for (const char* hal : {"norm.hal", "convkq.hal", "prepab.hal", "rowsplit.hal", "postnorm.hal", "unpack.hal",
                            "gemv.hal", "rmsnorm.hal", "argmax.hal", "accum.hal", "embed.hal"})
      Exe(hal);
    GeomOf("embed.hal");   // one token per workgroup: refuses a set emitted for another chunk size
    if (static_cast<std::uint32_t>(Find("token_embd.weight")->type) != 23)
      throw LoomError("prefill: token_embd: only IQ4_XS is wired (embed.hal)");
  }

  // Bindings shared by every GEMM: weights, then the IQ grid / sign tables the format needs.
  std::vector<hrx_buffer_ref_t> GemmWeights(const core::TensorInfo& t, const Fmt& f) const {
    std::vector<hrx_buffer_ref_t> b = {TRef(t)};
    const std::string n = f.name;
    if (n == "iq3s") b.push_back(Ref(*grid_iq3s_));
    if (n == "iq3xxs") b.push_back(Ref(*grid_iq3xxs_));
    if (n == "iq2xxs") b.push_back(Ref(*grid_iq2xxs_));
    if (n == "iq2xs") b.push_back(Ref(*grid_iq2xs_));
    // The IQ3_XXS and IQ2 sign tables are the same 128 bytes.
    if (n == "iq3xxs" || n == "iq2xxs" || n == "iq2xs") b.push_back(Ref(*ksigns_));
    return b;
  }
  // HAL name "<kind>_<fmt>_<m_tiles>_<k_blocks>.hal" of the GEMM on tensor t.
  std::string GemmHal(const char* kind, const core::TensorInfo& t, Fmt* f) const {
    if (!FmtOf(static_cast<std::uint32_t>(t.type), f)) throw LoomError(std::string("no GEMM format for ") + kind);
    const std::uint32_t mt = static_cast<std::uint32_t>(t.dims[1] / 16);
    const std::uint32_t kb = static_cast<std::uint32_t>(t.dims[0] / f->qk);
    return std::string(kind) + "_" + f->name + "_" + std::to_string(mt) + "_" + std::to_string(kb) + ".hal";
  }
  std::uint32_t MTiles(const core::TensorInfo& t) const { return static_cast<std::uint32_t>(t.dims[1] / 16); }
  // The GEMM variant for this chunk's real tokens: the set may carry "<hal>.t128.hal" / ".t64.hal" with narrower token
  // tiles. Pick the least (padded token rows x cost per row); a narrow tile pads less but costs more per row (measured:
  // x1.12 at 128 tokens, x1.6 at 64). Every variant computes the same values.
  // A tuned set (engine/tune) carries the measured choice instead: "pick:<hal>:<max tokens>" rows map a token bucket to
  // the token tile to use; chunks above the largest bucket take the full tile.
  std::string PickGemm(const std::string& hal) {
    if (calib_ && graph_) {
      int tag = -1;
      std::string v = calib_->Choose(hal, &tag);
      pending_tag_ = tag;
      if (!v.empty()) return v;
    }
    const std::string stem = hal.substr(0, hal.size() - 4);
    const std::string prefix = "pick:" + hal + ":";
    bool tuned = false;
    std::uint32_t bucket = ~0u, bn = 0;
    for (auto it = geom_.lower_bound(prefix); it != geom_.end() && it->first.compare(0, prefix.size(), prefix) == 0;
         ++it) {
      tuned = true;
      if (it->second.tokens >= n_ && it->second.tokens < bucket) bucket = it->second.tokens, bn = it->second.rowgrp;
    }
    if (tuned) {
      const auto full = geom_.find(hal);
      if (bn == 0 || (full != geom_.end() && bn == full->second.tokens)) return hal;
      const std::string v = stem + ".t" + std::to_string(bn) + ".hal";
      return geom_.count(v) ? v : hal;
    }
    std::string best = hal;
    double best_cost = 1e30;
    for (const std::string& h : {hal, stem + ".t128.hal", stem + ".t64.hal"}) {
      const auto it = geom_.find(h);
      if (it == geom_.end()) continue;
      const std::uint32_t bn = it->second.tokens;
      const double weight = bn >= 256 ? 1.0 : bn >= 128 ? 1.12 : 1.6;
      const double cost = static_cast<double>((n_ + bn - 1) / bn * bn) * weight;
      if (cost < best_cost) best = h, best_cost = cost;
    }
    return best;
  }

  // The afrag form of a GEMM HAL ("<hal>.af.hal" / ".af.to.hal", emit_prefill_pp.py afrag_variants), or "".
  std::string AfHal(const std::string& hal, const char* suffix = ".af.hal") const {
    const std::string v = hal.substr(0, hal.size() - 4) + suffix;
    return geom_.count(v) ? v : "";
  }
  // The FFN: norm, gate, swiglu (up), down. A GEMM with an afrag form reads its input fragment-major: the norm writes
  // that layout into normt_ for gate / up (and the plain one into scratch_ if one of them has no afrag form), the swiglu
  // writes it into ffnup_ for down when both have afrag forms.
  // afrag GEMMs take 512-token tiles: a short chunk pads more with them than with the narrow tiles PickGemm would take
  // (300 tokens: 384 rows at x1.12, +2.4% with afrag). They run when their padded rows at ~0.93 the cost per row (their
  // clock-free gain) cost no more than the best of the full and the set's narrow tiles, PickGemm's untuned weights.
  bool AfChunk() const {
    const auto padded = [&](std::uint32_t bn) { return static_cast<double>((n_ + bn - 1) / bn * bn); };
    double best = padded(256);
    if (std::any_of(geom_.begin(), geom_.end(),
                    [](const auto& g) { return g.first.find(".t128.hal") != std::string::npos; }))
      best = std::min({best, padded(128) * 1.12, padded(64) * 1.6});
    return padded(512) * 0.93 <= best;
  }
  // Whether the GEMM of this kind on tensor wname runs its afrag form in this chunk.
  bool Af(const char* kind, const std::string& wname) const {
    Fmt f{};
    return AfChunk() && !AfHal(GemmHal(kind, *Find(wname), &f)).empty();
  }
  void RunFfn(const std::string& pre) {
    if (RunFfnFused(pre)) return;
    const bool gate_af = Af("gemm_kstore", pre + "ffn_gate.weight");
    const bool up_af = Af("gemm_swiglu", pre + "ffn_up.weight");
    const bool down_af = up_af && Af("gemm_kres", pre + "ffn_down.weight");
    const std::uint32_t nn = trim_row_ == kAllRows ? NpuRows("ffn") : 0;
    const std::string gs = nn ? NpuHal(KstoreHal(pre + "ffn_gate.weight", gate_af)) : "";
    const std::string us = nn ? NpuHal(SwigluHal(pre + "ffn_up.weight", up_af, down_af)) : "";
    RunNorm(pre + "post_attention_norm.weight",
            gate_af && up_af ? NormOut::kTiled : gate_af || up_af ? NormOut::kBoth : NormOut::kRow,
            !gs.empty() && !us.empty() ? "ffn" : nullptr);
    if (!gs.empty() && !us.empty()) {
      // the NPU's rows of gate and up; the unpack applies silu(gate) * up
      const auto a = NpuEncode("ffn", gate_af && up_af ? *normt_ : *scratch_, gate_af && up_af);
      std::size_t w_off = 0, c_off = 0;
      const auto wg = NpuDecode("ffn", pre + "ffn_gate.weight", nn, w_off);
      const auto wu = NpuDecode("ffn", pre + "ffn_up.weight", nn, w_off);
      std::vector<std::uint32_t> calls;
      const NpuView cg = NpuCalls("ffn", a, wg, c_off, calls), cu = NpuCalls("ffn", a, wu, c_off, calls);
      NpuEnqueue(std::move(calls), "ffn");
      RunKstoreSplit(pre + "ffn_gate.weight", *gateffn_, gate_af, gs, nn);
      RunSwigluSplit(pre + "ffn_up.weight", up_af, us, nn);
      NpuJoin();
      FfnUnpack(down_af, {cg.offset, cg.length + cu.length});
    } else {
      RunKstore(pre + "ffn_gate.weight", *gateffn_, gate_af);
      RunSwiglu(pre + "ffn_up.weight", up_af, down_af);
    }
    RunResidual(pre + "ffn_down.weight", *ffnup_, down_af);
  }

  // kRow: f16 row-major into scratch_; kTiled: fragment-major into normt_ (afrag GEMMs); kBoth: both, one pass.
  enum class NormOut { kRow, kTiled, kBoth };
  // The FFN with ffn_gate and ffn_up in one afrag GEMM ("gemm_ffn_<fmt>_..", emit_prefill_pp.py ffn_fused): both of one
  // format and ffn_up stored right after ffn_gate, so one binding spans them (the kernel reads up row r at m_rows + r).
  // It writes f16(silu(gate) * up) into ffnup_, fragment-major when ffn_down is afrag too. Returns false if not taken.
  bool RunFfnFused(const std::string& pre) {
    if (!AfChunk()) return false;
    const auto* tg = Find(pre + "ffn_gate.weight");
    const auto* tu = Find(pre + "ffn_up.weight");
    if (tg->type != tu->type || tg->dims != tu->dims || tu->offset != tg->offset + tg->bytes) return false;
    Fmt f{};
    const std::string base = GemmHal("gemm_ffn", *tg, &f);
    const bool down_af = Af("gemm_kres", pre + "ffn_down.weight");
    const std::string hal = AfHal(base, down_af ? ".af.to.hal" : ".af.hal");
    if (hal.empty()) return false;
    RunNorm(pre + "post_attention_norm.weight", NormOut::kTiled,
            trim_row_ == kAllRows && NpuRows("ffn") && !NpuHal(hal).empty() ? "ffn" : nullptr);
    const Geom g = GeomOf(hal);
    auto b = GemmWeights(*tg, f);
    b[0].length = static_cast<std::size_t>(tg->bytes + tu->bytes);
    const std::size_t first = b.size();
    for (const LoomBuffer* x : std::initializer_list<const LoomBuffer*>{normt_, wstage_, ostage_, ffnup_}) b.push_back(Ref(*x));
    const std::uint32_t tt = Trim(b, first, {std::size_t(tg->dims[0]) * 2, 0, 0, std::size_t(tg->dims[1]) * 2}, g);
    const std::uint32_t nn = tt == g.tt ? NpuRows("ffn") : 0;
    const std::string sh = nn ? NpuHal(hal) : "";
    if (!sh.empty()) {
      // the NPU's rows of gate and up (the weights' last rows of each tensor); the unpack applies silu(gate) * up
      const auto a = NpuEncode("ffn", *normt_, true);
      std::size_t w_off = 0, c_off = 0;
      const auto wg = NpuDecode("ffn", pre + "ffn_gate.weight", nn, w_off);
      const auto wu = NpuDecode("ffn", pre + "ffn_up.weight", nn, w_off);
      std::vector<std::uint32_t> calls;
      const NpuView cg = NpuCalls("ffn", a, wg, c_off, calls), cu = NpuCalls("ffn", a, wu, c_off, calls);
      NpuEnqueue(std::move(calls), "ffn");
      Dispatch(Exe(sh), ("yah_ffn_gemm_" + std::string(f.name) + "_ffn").c_str(),
               (MTiles(*tg) - nn / 16) / GeomOf(sh).rowgrp, tt, 1, 32, 1, 1, b, GemmWrites(b, f, {ffnup_}));
      NpuJoin();
      FfnUnpack(down_af, {cg.offset, cg.length + cu.length});
    } else {
      Dispatch(Exe(hal), ("yah_ffn_gemm_" + std::string(f.name) + "_ffn").c_str(), MTiles(*tg) / g.rowgrp, tt, 1, 32, 1,
               1, b, GemmWrites(b, f, {ffnup_}));
    }
    RunResidual(pre + "ffn_down.weight", *ffnup_, down_af);
    return true;
  }

  // npu_site: the NPU split of a K = 5120 site reads this norm's output; with "npunormbfp" in the set the norm also
  // writes its BFP16 input into A ("<norm>_bfp.hal") and NpuEncode(npu_site) skips the encoder.
  void RunNorm(const std::string& wname, NormOut mode = NormOut::kRow, const char* npu_site = nullptr) {
    // A row takes `split` waves (dispatch.txt norm_split, else 1); a workgroup of w waves takes w / split rows.
    const bool bfp = npu_site && geom_.count("npunormbfp");
    norm_bfp_ = bfp ? npu_site : "";
    const std::string hal = mode == NormOut::kTiled ? "norm_t" : mode == NormOut::kBoth ? "norm_rt" : "norm";
    const LoomExecutable& exe = Exe(hal + (bfp ? "_bfp.hal" : ".hal"));
    const std::uint32_t ws = exe.WorkgroupSize(exe.OrdinalOrZero("yah_half_norm"));
    const auto ns = geom_.find("norm_split");
    const std::uint32_t split = ns != geom_.end() && ns->second.rowgrp ? ns->second.rowgrp : 1;
    const std::uint32_t rows_per_wg = ws ? ws / 32 / split : 1;
    if (!rows_per_wg || (ws && ws % (32 * split))) throw LoomError("norm.hal: workgroup size does not match norm_split");
    if (B_ % rows_per_wg) throw LoomError("norm.hal: rows per workgroup must divide the chunk");
    DumpNorm(wname);
    std::vector<hrx_buffer_ref_t> b =
        mode == NormOut::kBoth
            ? std::vector<hrx_buffer_ref_t>{Ref(*hidden_), Ref(*reszero_), TRef(*Find(wname)), Ref(*sumout_),
                                            Ref(*scratch_), Ref(*normt_)}
            : std::vector<hrx_buffer_ref_t>{Ref(*hidden_), Ref(*reszero_), TRef(*Find(wname)), Ref(*sumout_),
                                            Ref(mode == NormOut::kTiled ? *normt_ : *scratch_)};
    if (bfp) b.push_back({npu_->A().handle, 0, NpuK(5120).a});
    Dispatch(exe, "yah_half_norm", B_ / rows_per_wg, 1, 1, 32, 1, 1, b);
  }
  // Debug (YAH_DUMP_ACT=<prefix>, YAH_DUMP_LAYERS=l0,l1,..): the row-major f16 output of each RMSNorm that feeds a GEMM
  // in those layers (n_ rows x 5120) to <prefix>_<norm weight name>.f16, from an extra norm.hal pass into its own buffer.
  bool DumpOn() const { return std::getenv("YAH_DUMP_ACT") != nullptr; }
  void DumpNorm(const std::string& wname) {
    const char* prefix = std::getenv("YAH_DUMP_ACT");
    if (!prefix || wname.rfind("blk.", 0) != 0) return;
    const std::string layer = wname.substr(4, wname.find('.', 4) - 4);
    const std::string want = std::string(",") + (std::getenv("YAH_DUMP_LAYERS") ? std::getenv("YAH_DUMP_LAYERS") : "") + ",";
    if (want.find("," + layer + ",") == std::string::npos) return;
    if (!dumpbuf_) dumpbuf_ = &Alloc(std::size_t{B_} * kHidden * 2);
    const LoomExecutable& exe = Exe("norm.hal");
    const std::uint32_t ws = exe.WorkgroupSize(exe.OrdinalOrZero("yah_half_norm"));
    const auto ns = geom_.find("norm_split");
    const std::uint32_t split = ns != geom_.end() && ns->second.rowgrp ? ns->second.rowgrp : 1;
    const std::uint32_t rows_per_wg = ws ? ws / 32 / split : 1;
    Dispatch(exe, "yah_half_norm", B_ / rows_per_wg, 1, 1, 32, 1, 1,
             {Ref(*hidden_), Ref(*reszero_), TRef(*Find(wname)), Ref(*sumout_), Ref(*dumpbuf_)});
    gpu_.Synchronize();
    std::vector<std::uint16_t> host(std::size_t{n_} * kHidden);
    gpu_.D2H(*dumpbuf_, host.data(), host.size() * 2);
    std::ofstream(std::string(prefix) + "_" + wname + ".f16", std::ios::binary)
        .write(reinterpret_cast<const char*>(host.data()), static_cast<std::streamsize>(host.size() * 2));
  }
  LoomBuffer* dumpbuf_ = nullptr;

  // Decode-free GEMMs (emit_prefill_pp.py decode_free): "dq_<fmt>_<mt>_<kb>.hal" decodes a GEMM's weights to f16 scratch
  // once per chunk with the tile GEMM's own decode (the same f16 values), "gemm_<kind>_f16_<mt>_<kb>.hal" multiplies them
  // without decode. The pass for the next GEMM is recorded right after each GEMM: no edge between them, so they share a
  // graph segment and run together; the barrier before the GEMM's consumer drains both. Three scratch slots rotate, so a
  // slot is rewritten only after the GEMM two before has drained. Chunks of at most kDfMinTokens real tokens keep the
  // fused decode: one token tile decodes each weight once anyway.
  static constexpr std::uint32_t kDfMinTokens = 512;
  struct DfStep {
    std::string dq, gemm;
    const core::TensorInfo* t;
    Fmt f;
  };
  std::vector<DfStep> df_plan_;
  bool df_planned_ = false, df_planning_ = false, df_on_ = false;
  std::size_t df_k_ = 0;
  LoomBuffer* df_w_[3] = {nullptr, nullptr, nullptr};
  std::vector<hrx_buffer_ref_t> df_after_;

  // Records the GEMMs that have a decode-free path, in dispatch order (a dry run: Dispatch records nothing), and sizes
  // the scratch for the largest.
  void PlanDecodeFree(std::uint32_t ci) {
    df_planned_ = true;
    bool any = false;
    for (const auto& [name, g] : geom_) any = any || name.rfind("dq_", 0) == 0;
    if (!any) return;
    LoomBuffer* h = hidden_;
    LoomBuffer* h2 = hidden2_;
    df_planning_ = true;
    RecordLayers(ci, {});
    df_planning_ = false;
    hidden_ = h, hidden2_ = h2;
    std::size_t most = 0;
    for (const DfStep& s : df_plan_) most = std::max<std::size_t>(most, std::size_t{s.t->dims[1]} * s.t->dims[0] * 2);
    for (auto& w : df_w_) w = &Alloc(most);
  }
  // The f16 GEMM to run for this call, or "" for the fused path. Planning: records the call.
  std::string DfGemm(const char* kind, const core::TensorInfo& t, const Fmt& f) {
    const std::uint32_t mt = MTiles(t);
    const std::string dq = "dq_" + std::string(f.name) + "_" + std::to_string(mt) + "_" +
                           std::to_string(t.dims[0] / f.qk) + ".hal";
    const std::string gemm = "gemm_" + std::string(kind) + "_f16_" + std::to_string(mt) + "_" +
                             std::to_string(t.dims[0] / 256) + ".hal";
    if (!geom_.count(dq) || !geom_.count(gemm)) return "";
    if (df_planning_) {
      df_plan_.push_back({dq, gemm, &t, f});
      return "";
    }
    if (!df_on_) return "";
    if (df_k_ >= df_plan_.size() || df_plan_[df_k_].t != &t) throw LoomError("prefill: decode-free plan out of step");
    return gemm;
  }
  // Decode plan step k's weights into its scratch slot. The dequant is persistent: its row's first column is the
  // workgroup count (one per WGP), and its workgroups walk the tensor.
  void DispatchDequant(std::size_t k) {
    const DfStep& s = df_plan_[k];
    const Geom g = geom_.at(s.dq);
    auto b = GemmWeights(*s.t, s.f);
    b.push_back(Ref(*df_w_[k % 3]));
    Dispatch(Exe(s.dq), ("yah_dequant_" + std::string(s.f.name)).c_str(), g.tokens, 1, 1, 256, 1, 1, b,
             std::uint64_t{1} << (b.size() - 1));
  }
  // The weight binding of the current decode-free GEMM.
  std::vector<hrx_buffer_ref_t> DfWeights() const { return {Ref(*df_w_[df_k_ % 3])}; }
  // Before a decode-free GEMM: the next one's weights. Recorded first, so its few workgroups launch before the GEMM's
  // and run beside them (a dispatch's workgroups only launch once the previous dispatch's have all launched).
  // gemm: the GEMM's bindings; the dequant is ordered after its inputs (not its weights) so that the barrier the GEMM
  // needs falls before the dequant and the two share a graph segment.
  void DfAhead(const std::vector<hrx_buffer_ref_t>& gemm) {
    if (df_k_ + 1 >= df_plan_.size()) return;
    df_after_.assign(gemm.begin() + 1, gemm.end());
    DispatchDequant(df_k_ + 1);
    df_after_.clear();
  }
  void DfNext() { ++df_k_; }

  // NPU column split (EnableNpu, emit_prefill_pp.npu_split). A site's NPU rows in this chunk (0: GPU only).
  std::uint32_t NpuRows(const char* site) const {
    const auto it = npu_rows_.find(site);
    return npu_on_ && it != npu_rows_.end() ? it->second : 0;
  }
  // (k offset, K) of a site's NPU calls: one K, or down's 6 + 6 + 5 passes of K = 17408 (its weight panel does not fit
  // a memory tile).
  // The NPU runs one image (K = 5120 per call; an image switch costs ~0.55 ms): a site's K in chunks of 5120, the rest
  // of K (out: 1024, down: 2048) on the GPU ("npurem_<site>_<fmt>.hal"), added by the unpack.
  static std::uint32_t NpuSiteK(const std::string& site) { return site == "down" ? 17408 : site == "out" ? 6144 : 5120; }
  static std::vector<std::pair<std::uint32_t, std::uint32_t>> NpuChunks(const std::string& site) {
    std::vector<std::pair<std::uint32_t, std::uint32_t>> v;
    for (std::uint32_t k = 0; k + 5120 <= NpuSiteK(site); k += 5120) v.push_back({k, 5120});
    return v;
  }
  static std::uint32_t NpuRem(const std::string& site) { return NpuSiteK(site) % 5120; }
  // Chunked sites name their encoders and decoders per chunk ("_c<i>").
  static bool NpuChunked(const std::string& site) { return NpuChunks(site).size() > 1 || NpuRem(site) != 0; }
  // Bytes per NPU call of a K (dispatch.txt "npubytes_<K>": activations of the chunk, a weight panel, a C panel).
  struct NpuBytes {
    std::size_t a, w, c;
  };
  NpuBytes NpuK(std::uint32_t K) const {
    const Geom& g = geom_.at("npubytes_" + std::to_string(K));
    return {g.tokens, g.rowgrp, g.tt};
  }
  // Output rows per NPU call (dispatch.txt "npurows": 8 columns of gen_npu_gemm.TN).
  std::uint32_t NpuCallRows() const { return geom_.at("npurows").tokens; }
  // The GPU's share of GEMM HAL hal ("<hal>.npu.hal": its leading rows at the full stride), or "".
  std::string NpuHal(const std::string& hal) const {
    const std::string v = hal.empty() ? "" : hal.substr(0, hal.size() - 4) + ".npu.hal";
    return geom_.count(v) ? v : "";
  }
  // The kstore HAL a GEMM on wname runs (afrag form with af).
  std::string KstoreHal(const std::string& wname, bool af) const {
    Fmt f{};
    const std::string base = GemmHal("gemm_kstore", *Find(wname), &f);
    return af ? AfHal(base) : base;
  }
  // input (row-major, or fragment-major with tiled) encoded into A for each of the site's K chunks; their A views.
  std::vector<NpuView> NpuEncode(const std::string& site, const LoomBuffer& input, bool tiled) {
    const auto chunks = NpuChunks(site);
    // down after an ffn unpack that wrote its columns (FfnUnpack): only each chunk's columns before them
    const bool from_ffn = site == "down" && ffn_bfp_;
    ffn_bfp_ = false;
    // a K = 5120 site after a norm that wrote its input (RunNorm npu_site)
    const bool from_norm = norm_bfp_ == site;
    norm_bfp_.clear();
    const std::uint32_t ffn_cols = from_ffn ? geom_.at("npuffnbfp").tokens : NpuSiteK(site);
    std::vector<NpuView> views;
    std::size_t off = 0;
    for (std::size_t c = 0; c < chunks.size(); ++c) {
      const std::size_t bytes = NpuK(chunks[c].second).a;
      const auto [k0, K] = chunks[c];
      const std::uint32_t cover = ffn_cols > k0 ? std::min(K, ffn_cols - k0) : 0;
      if (cover && !from_norm) {
        const std::string hal = "npu_enc_" + std::to_string(NpuSiteK(site)) + (NpuChunked(site) ? "_c" + std::to_string(c) : "") +
                                (cover < K ? "p" : "") + (tiled ? "_t" : "") + ".hal";
        const Geom& g = geom_.at(hal);
        Dispatch(Exe(hal), "yah_bfp16_encode_act", g.tokens, 1, 1, g.rowgrp, 1, 1,
                 {Ref(input), {npu_->A().handle, off, bytes}}, 2);
      }
      views.push_back({off, bytes});
      off += bytes;
    }
    return views;
  }
  // The ffn site's unpack: f16(silu(gate) * up) into ffnup_ (fragment-major for an afrag down). With "npuffnbfp" (down
  // split too) it also writes down's BFP16 input for the ffn's NPU columns into A at down's chunk offsets, and
  // NpuEncode("down") encodes only the columns before them.
  void FfnUnpack(bool tiled, NpuView c) {
    std::vector<hrx_buffer_ref_t> rest = {Ref(*ffnup_)};
    std::uint64_t writes = 2;
    ffn_bfp_ = geom_.count("npuffnbfp") && NpuRows("down");
    if (ffn_bfp_) {
      std::size_t bytes = 0;
      for (const auto& [k0, K] : NpuChunks("down")) bytes += NpuK(K).a;
      rest.push_back({npu_->A().handle, 0, bytes});
      writes |= 4;
    }
    NpuUnpack(tiled ? "npu_unpack_ffn_t.hal" : "npu_unpack_ffn.hal", c, rest, writes);
  }
  // The NPU's rows of wname (its last rows) decoded straight into W panels from w_off within this job's W slot
  // ("dqbfp_<fmt>_<mt>_<kb>[_c<i>]", dispatch.txt: <workgroups> <workgroup size>); the W views per chunk and call.
  // The decode reads only weights, so it is recorded into the previous job's GPU segment, where it runs beside that
  // segment's GEMM while the NPU works; the two slots alternate per job.
  std::vector<std::vector<NpuView>> NpuDecode(const std::string& site, const std::string& wname, std::uint32_t rows,
                                              std::size_t& w_off) {
    const std::size_t slot = (npu_job_ % 2) * (npu_->W().size / 2);
    const auto* t = Find(wname);
    Fmt f{};
    const std::string base = GemmHal("gemm_kstore", *t, &f);
    const auto chunks = NpuChunks(site);
    const std::size_t row_bytes = static_cast<std::size_t>(t->bytes) / static_cast<std::size_t>(t->dims[1]);
    std::vector<std::vector<NpuView>> views;
    for (std::size_t c = 0; c < chunks.size(); ++c) {
      const std::string dq = "dqbfp" + base.substr(std::strlen("gemm_kstore"), base.size() - std::strlen("gemm_kstore") - 4) +
                             (NpuChunked(site) ? "_c" + std::to_string(c) : "") + ".hal";
      const Geom& g = geom_.at(dq);
      const std::size_t panel = NpuK(chunks[c].second).w, calls = rows / NpuCallRows();
      auto b = GemmWeights(*t, f);
      b[0].offset += (static_cast<std::size_t>(t->dims[1]) - rows) * row_bytes;
      b[0].length = std::size_t{rows} * row_bytes;
      b.push_back({npu_->W().handle, slot + w_off, calls * panel});
      PlannedDecode d{dq, "yah_dequant_" + std::string(f.name) + "_bfp16", g.tokens, g.rowgrp, b};
      if (npu_planning_) planned_dq_.resize(npu_job_ + 1), planned_dq_[npu_job_].push_back(d);
      // the previous job's NpuJoin dispatched it already, unless the plan differs here (the last layer's trimmed tail)
      if (!Predecoded(d))
        Dispatch(Exe(dq), d.name.c_str(), d.gx, 1, 1, d.wg, 1, 1, d.b, std::uint64_t{1} << (b.size() - 1));
      views.emplace_back();
      for (std::size_t p = 0; p < calls; ++p) views.back().push_back({slot + w_off + p * panel, panel});
      w_off += calls * panel;
    }
    return views;
  }
  // The NPU calls of one matrix: per chunk, per NpuCallRows() rows; C panels from c_off, chunk-major (the unpack sums the chunks).
  // Returns the C view they fill.
  NpuView NpuCalls(const std::string& site, const std::vector<NpuView>& a, const std::vector<std::vector<NpuView>>& w,
                   std::size_t& c_off, std::vector<std::uint32_t>& calls) {
    const auto chunks = NpuChunks(site);
    const std::size_t start = c_off;
    for (std::size_t c = 0; c < chunks.size(); ++c) {
      const std::string image = dir_ + "/npu_gemm_" + std::to_string(chunks[c].second) + ".xdna";
      const std::size_t cp = NpuK(chunks[c].second).c;
      for (const NpuView& wv : w[c]) {
        calls.push_back(npu_->Bind(image, a[c], wv, {c_off, cp}));
        c_off += cp;
      }
    }
    return {start, c_off - start};
  }
  // npu_unpack_<site>: C view c, then bindings (writes: the mask over all bindings, C at bit 0).
  void NpuUnpack(const std::string& hal, NpuView c, std::vector<hrx_buffer_ref_t> rest, std::uint64_t writes) {
    const Geom& g = geom_.at(hal);
    rest.insert(rest.begin(), {npu_->C().handle, c.offset, c.length});
    Dispatch(Exe(hal), "yah_npu_unpack", g.tokens, 1, 1, g.rowgrp, 1, 1, rest, writes);
  }
  // The swiglu HAL a GEMM on wname runs (afrag form with af, fragment-major output with tout).
  std::string SwigluHal(const std::string& wname, bool af, bool tout) const {
    Fmt f{};
    const std::string base = GemmHal("gemm_swiglu", *Find(wname), &f);
    return af ? AfHal(base, tout ? ".af.to.hal" : ".af.hal") : base;
  }
  // The kqg HAL a GEMM on wname runs (afrag form with af), "" if the set has none.
  std::string KqgHal(const std::string& wname, bool af) const {
    Fmt f{};
    if (!FmtOf(static_cast<std::uint32_t>(Find(wname)->type), &f)) return "";
    const std::string base = GemmHal("gemm_kqg", *Find(wname), &f);
    return !geom_.count(base) ? "" : af ? AfHal(base) : base;
  }
  // The GPU's rows of the swiglu GEMM on wname (hal: the split HAL), its gate input and output at the full stride.
  void RunSwigluSplit(const std::string& wname, bool af, const std::string& hal, std::uint32_t npu_rows) {
    const auto* t = Find(wname);
    Fmt f{};
    GemmHal("gemm_swiglu", *t, &f);
    const Geom g = GeomOf(hal);
    auto b = GemmWeights(*t, f);
    const std::size_t first = b.size();
    for (const LoomBuffer* x :
         std::initializer_list<const LoomBuffer*>{af ? normt_ : scratch_, gateffn_, uwstage_, ostage_, ffnup_})
      b.push_back(Ref(*x));
    const std::size_t M = t->dims[1];
    const std::uint32_t tt = Trim(b, first, {std::size_t(t->dims[0]) * 2, M * 4, 0, 0, M * 2}, g);
    Dispatch(Exe(hal), ("yah_ffn_gemm_" + std::string(f.name) + "_swiglu").c_str(), (MTiles(*t) - npu_rows / 16) / g.rowgrp,
             tt, 1, 32, 1, 1, b, GemmWrites(b, f, {ffnup_}));
  }
  // The GPU's heads of the attention q projection (hal: the split kqg HAL) into q_ / gate_ at the full stride.
  void RunKqgSplit(const std::string& wname, bool af, const std::string& hal, std::uint32_t npu_rows) {
    const auto* t = Find(wname);
    Fmt f{};
    GemmHal("gemm_kqg", *t, &f);
    const Geom g = GeomOf(hal);
    auto b = GemmWeights(*t, f);
    for (const LoomBuffer* x : std::initializer_list<const LoomBuffer*>{af ? normt_ : scratch_, wstage_, ostage_, q_, gate_})
      b.push_back(Ref(*x));
    Dispatch(Exe(hal), ("yah_ffn_gemm_" + std::string(f.name) + "_kqg").c_str(), (MTiles(*t) - npu_rows / 16) / g.rowgrp,
             TokenTiles(g), 1, 32, 1, 1, b, GemmWrites(b, f, {q_, gate_}));
  }
  // The GPU's rows [0, rows - npu_rows) of kstore wname (hal: the split HAL); out keeps the full row stride.
  void RunKstoreSplit(const std::string& wname, const LoomBuffer& out, bool af, const std::string& hal,
                      std::uint32_t npu_rows) {
    const auto* t = Find(wname);
    Fmt f{};
    GemmHal("gemm_kstore", *t, &f);
    const Geom g = GeomOf(hal);
    auto b = GemmWeights(*t, f);
    const std::size_t first = b.size();
    for (const LoomBuffer* x : std::initializer_list<const LoomBuffer*>{af ? normt_ : scratch_, wstage_, ostage_})
      b.push_back(Ref(*x));
    b.push_back(Ref(out));
    const std::uint32_t tt = Trim(b, first, {std::size_t(t->dims[0]) * 2, 0, 0, std::size_t(t->dims[1]) * 4}, g);
    Dispatch(Exe(hal), ("yah_ffn_gemm_" + std::string(f.name)).c_str(), (MTiles(*t) - npu_rows / 16) / g.rowgrp, tt, 1,
             32, 1, 1, b, GemmWrites(b, f, {&out}));
  }
  // The chunk's graph: the whole chunk is one graph, the NPU handoffs included (NpuEnqueue / NpuJoin).
  // Replacing a launched graph is safe: the queue retains its command buffers until they complete, and every buffer and
  // executable they use is this object's.
  void NewGraph() {
    chunk_graph_ = std::make_unique<LoomGraph>(gpu_);
    LoomGraph& g = *chunk_graph_;
    g.ReadOnly(weights_);
    for (const LoomBuffer* t : {grid_iq3s_, grid_iq3xxs_, grid_iq2xxs_, grid_iq2xs_, ksigns_, eps_, reszero_})
      g.ReadOnly(t->handle);
    graph_ = &g;
  }

  // NPU handoffs are graph nodes on flag words in host memory (NpuSplit::Flags), so the host never splits the chunk.
  // NpuEnqueue: the graph stores the job's gate value to its ready word once its encoders / decoders are done (release,
  // system scope: A and W reach memory first); the NPU waits for it itself. NpuJoin: yah_npu_flag_wait polls the done
  // word; it stands for the NPU's reads of A / W and writes of C, so the unpacks and the next job's encoders and
  // decoders order after it. The jobs are queued right after the chunk graph launches.
  // A weight decode of the planning pass (EnableNpu), dispatched early in one-graph mode.
  struct PlannedDecode {
    std::string hal, name;
    std::uint32_t gx, wg;
    std::vector<hrx_buffer_ref_t> b;
    bool operator==(const PlannedDecode& o) const {
      if (hal != o.hal || gx != o.gx || wg != o.wg || b.size() != o.b.size()) return false;
      for (std::size_t i = 0; i < b.size(); ++i)
        if (b[i].buffer != o.b[i].buffer || b[i].offset != o.b[i].offset || b[i].length != o.b[i].length) return false;
      return true;
    }
  };
  bool Predecoded(const PlannedDecode& d) const {
    return std::find(predecoded_.begin(), predecoded_.end(), d) != predecoded_.end();
  }
  // Job j's half of W (NpuDecode alternates them): the next job's decode must not wait for this job's NPU work.
  hrx_buffer_ref_t WSlot(std::uint32_t job) const {
    const std::size_t half = npu_->W().size / 2;
    return {npu_->W().handle, (job % 2) * half, half};
  }
  hrx_buffer_ref_t FlagWord(std::uint32_t w) const { return {npu_->Flags()->handle, std::size_t{w} * 4, 4}; }
  // The job's calls; tag names it in the NPU stats.
  void NpuEnqueue(std::vector<std::uint32_t> calls, std::string tag) {
    const std::uint32_t job = npu_job_++;
    if (npu_planning_) return;
    const NpuSplit::Job words = npu_->NewJob(calls);
    graph_->AtomicStore(FlagWord(words.ready), words.gate, HRX_ATOMIC_FLAG_RELEASE | HRX_ATOMIC_FLAG_SYSTEM_SCOPE,
                        {Ref(npu_->A()), WSlot(job)}, {Ref(npu_->C())});
    flagged_.push_back({std::move(calls), std::move(tag), words});
  }
  // Queue the GPU work that runs beside the last enqueued job before this.
  void NpuJoin() {
    if (npu_planning_) return;
    // the next job's weight decodes run beside this job's GPU work
    predecoded_.clear();
    const auto job = static_cast<std::uint32_t>(flagged_.size() - 1);   // this chunk's last job (NpuEnqueue)
    if (job + 1 < planned_dq_.size())
      for (const PlannedDecode& d : planned_dq_[job + 1]) {
        Dispatch(Exe(d.hal), d.name.c_str(), d.gx, 1, 1, d.wg, 1, 1, d.b, std::uint64_t{1} << (d.b.size() - 1));
        predecoded_.push_back(d);
      }
    const NpuSplit::Queued& j = flagged_.back();
    const std::vector<hrx_buffer_ref_t> npu_side{Ref(npu_->A()), WSlot(job), Ref(npu_->C())};
    Dispatch(Exe("npu_flag_wait.hal"), "yah_npu_flag_wait", 1, 1, 1, 32, 1, 1,
             {FlagWord(j.words.done), FlagWord(j.words.ready), FlagWord(NpuSplit::kFlagStatus)}, 4, &npu_side);
  }

  // af: the afrag form, input normt_ (fragment-major)
  void RunKstore(const std::string& wname, const LoomBuffer& out, bool af = false) {
    const auto* t = Find(wname);
    Fmt f{};
    const std::string base = GemmHal("gemm_kstore", *t, &f);
    const std::string df = af ? "" : DfGemm("kstore", *t, f);
    const std::string hal = af ? AfHal(base) : df.empty() ? PickGemm(base) : df;
    const Geom g = GeomOf(hal);
    auto b = df.empty() ? GemmWeights(*t, f) : DfWeights();
    const std::size_t first = b.size();
    for (const LoomBuffer* x : std::initializer_list<const LoomBuffer*>{af ? normt_ : scratch_, wstage_, ostage_})
      b.push_back(Ref(*x));
    b.push_back(Ref(out));
    const std::uint32_t tt = Trim(b, first, {std::size_t(t->dims[0]) * 2, 0, 0, std::size_t(t->dims[1]) * 4}, g);
    if (!df.empty()) DfAhead(b);
    Dispatch(Exe(hal), ("yah_ffn_gemm_" + std::string(df.empty() ? f.name : "f16")).c_str(), MTiles(*t) / g.rowgrp,
             tt, 1, 32, 1, 1, b, GemmWrites(b, f, {&out}));
    if (!df.empty()) DfNext();
  }
  // The attention q projection with the q / gate unpack fused in (rows = heads x [256 q | 256 gate]).
  // Returns false if the set has no such HAL; the caller then runs kstore + yah_unpack_qg.
  // af: the afrag form, input normt_ (fragment-major)
  bool RunKqg(const std::string& wname, bool af = false) {
    const auto* t = Find(wname);
    Fmt f{};
    if (!FmtOf(static_cast<std::uint32_t>(t->type), &f)) return false;
    std::string hal = GemmHal("gemm_kqg", *t, &f);
    if (!geom_.count(hal)) return false;
    const std::string df = af ? "" : DfGemm("kqg", *t, f);
    hal = af ? AfHal(hal) : df.empty() ? PickGemm(hal) : df;
    const Geom g = GeomOf(hal);
    auto b = df.empty() ? GemmWeights(*t, f) : DfWeights();
    for (const LoomBuffer* x :
         std::initializer_list<const LoomBuffer*>{af ? normt_ : scratch_, wstage_, ostage_, q_, gate_})
      b.push_back(Ref(*x));
    if (!df.empty()) DfAhead(b);
    Dispatch(Exe(hal), ("yah_ffn_gemm_" + std::string(df.empty() ? f.name : "f16") + "_kqg").c_str(),
             MTiles(*t) / g.rowgrp, TokenTiles(g), 1, 32, 1, 1, b, GemmWrites(b, f, {q_, gate_}));
    if (!df.empty()) DfNext();
    return true;
  }
  // af: the afrag form, input normt_; tout: its output fragment-major (for an afrag down projection)
  void RunSwiglu(const std::string& wname, bool af = false, bool tout = false) {
    const auto* t = Find(wname);
    Fmt f{};
    const std::string base = GemmHal("gemm_swiglu", *t, &f);
    const std::string df = af ? "" : DfGemm("swiglu", *t, f);
    const std::string hal = af ? AfHal(base, tout ? ".af.to.hal" : ".af.hal") : df.empty() ? PickGemm(base) : df;
    if (hal.empty()) throw LoomError(base + ": the set lacks its afrag form");
    const Geom g = GeomOf(hal);
    auto b = df.empty() ? GemmWeights(*t, f) : DfWeights();
    const std::size_t first = b.size();
    for (const LoomBuffer* x :
         std::initializer_list<const LoomBuffer*>{af ? normt_ : scratch_, gateffn_, uwstage_, ostage_, ffnup_})
      b.push_back(Ref(*x));
    const std::size_t M = t->dims[1];
    const std::uint32_t tt = Trim(b, first, {std::size_t(t->dims[0]) * 2, M * 4, 0, 0, M * 2}, g);
    if (!df.empty()) DfAhead(b);
    Dispatch(Exe(hal), ("yah_ffn_gemm_" + std::string(df.empty() ? f.name : "f16") + "_swiglu").c_str(),
             MTiles(*t) / g.rowgrp, tt, 1, 32, 1, 1, b, GemmWrites(b, f, {ffnup_}));
    if (!df.empty()) DfNext();
  }
  // hidden += W input. The fused kres GEMM writes hidden + W input into hidden2; otherwise kStore writes W input
  // into partial and yah_residual_1d adds it. Either way the two hidden buffers swap.
  // af: the afrag form of the fused kres GEMM (input fragment-major)
  void RunResidual(const std::string& wname, const LoomBuffer& input, bool af = false) {
    const auto* t = Find(wname);
    Fmt f{};
    const std::string fused0 = GemmHal("gemm_kres", *t, &f);
    if (geom_.count(fused0)) {
      const std::string df = af ? "" : DfGemm("kres", *t, f);
      const std::string fused = af ? AfHal(fused0) : df.empty() ? PickGemm(fused0) : df;
      const Geom g = GeomOf(fused);
      auto b = df.empty() ? GemmWeights(*t, f) : DfWeights();
      const std::size_t first = b.size();
      for (const LoomBuffer* x : std::initializer_list<const LoomBuffer*>{&input, hidden_, wstage_, ostage_, hidden2_}) b.push_back(Ref(*x));
      const std::size_t M = t->dims[1];
      const std::uint32_t tt = Trim(b, first, {std::size_t(t->dims[0]) * 2, M * 4, 0, 0, M * 4}, g);
      if (!df.empty()) DfAhead(b);
      // the persistent kres (".af.p.hal": each workgroup runs every token tile of its row block, grid y = 1) unless trimmed
      const std::string pers = af ? AfHal(fused0, ".af.p.hal") : "";
      const bool use_p = !pers.empty() && tt == g.tt;
      // NPU split (sites "out": ssm_out / attn_output, "down": ffn_down) of untrimmed chunks
      const char* site = wname.find("ffn_down") != std::string::npos ? "down" : "out";
      const std::uint32_t nn = df.empty() && tt == g.tt ? NpuRows(site) : 0;
      // The persistent kres runs one workgroup per row block (one per CU unsplit), so the split uses the tiled form.
      const std::string sh = nn ? NpuHal(fused) : "";
      if (!sh.empty()) {
        const auto a = NpuEncode(site, input, af);
        std::size_t w_off = 0, c_off = 0;
        const auto w = NpuDecode(site, wname, nn, w_off);
        std::vector<std::uint32_t> calls;
        const NpuView c = NpuCalls(site, a, w, c_off, calls);
        NpuEnqueue(std::move(calls), site);
        const Geom gs = GeomOf(sh);
        const std::string kname = std::string("yah_ffn_gemm_") + f.name + "_kres";
        const std::uint32_t blocks = (MTiles(*t) - nn / 16) / gs.rowgrp;
        // The persistent kres (one workgroup per row block) over all but the last token tile, the tiled kres on that
        // tile beside it (disjoint token ranges: the graph runs them together on the CUs the row blocks leave idle).
        const std::string p3 = fused0.substr(0, fused0.size() - 4) + ".af.p3.npu.hal";   // NPU split only
        if (af && geom_.count(p3) && tt == g.tt) {
          const std::size_t T = tt - 1;
          const std::size_t tile = B_ / tt;
          // per-token bytes of input, residual and output (bindings first, first + 1, first + 4)
          const std::size_t stride[] = {std::size_t(t->dims[0]) * 2, M * 4, 0, 0, M * 4};
          auto part = [&](std::size_t tok0, std::size_t ntok) {
            auto r = b;
            for (std::size_t i = 0; i < 5; ++i)
              if (stride[i]) r[first + i].offset += tok0 * stride[i], r[first + i].length = ntok * stride[i];
            return r;
          };
          const auto b0 = part(0, T * tile), b1 = part(T * tile, tile);
          Dispatch(Exe(p3), kname.c_str(), blocks, 1, 1, 32, 1, 1, b0, GemmWrites(b0, f, {hidden2_}));
          Dispatch(Exe(sh), kname.c_str(), blocks, 1, 1, 32, 1, 1, b1, GemmWrites(b1, f, {hidden2_}));
        } else {
          Dispatch(Exe(sh), kname.c_str(), blocks, tt, 1, 32, 1, 1, b, GemmWrites(b, f, {hidden2_}));
        }
        // the rest of K for the NPU's rows (its last nn weight rows), beside the GPU's rows: f32 [tokens][nn]
        const std::string rh = "npurem_" + std::string(site) + "_" + f.name + ".hal";
        const Geom& gr = geom_.at(rh);
        const std::size_t row_bytes = static_cast<std::size_t>(t->bytes) / M;
        auto rb = GemmWeights(*t, f);
        rb[0].offset += (M - nn) * row_bytes, rb[0].length = std::size_t{nn} * row_bytes;
        for (const LoomBuffer* x : std::initializer_list<const LoomBuffer*>{&input, wstage_, ostage_}) rb.push_back(Ref(*x));
        rb.push_back({npurem_->handle, 0, std::size_t{B_} * nn * 4});
        Dispatch(Exe(rh), ("yah_ffn_gemm_" + std::string(f.name)).c_str(), nn / 16 / gr.rowgrp, gr.tt, 1, 32, 1, 1, rb,
                 std::uint64_t{1} << (rb.size() - 1));
        NpuJoin();
        NpuUnpack(std::string("npu_unpack_") + site + ".hal", c,
                  {Ref(*hidden_), {npurem_->handle, 0, std::size_t{B_} * nn * 4}, Ref(*hidden2_)}, 8);
        std::swap(hidden_, hidden2_);
        return;
      }
      Dispatch(Exe(use_p ? pers : fused), (std::string("yah_ffn_gemm_") + (df.empty() ? f.name : "f16") + "_kres").c_str(),
               MTiles(*t) / g.rowgrp, use_p ? 1 : tt, 1, 32, 1, 1, b, GemmWrites(b, f, {hidden2_}));
      if (!df.empty()) DfNext();
      std::swap(hidden_, hidden2_);
      return;
    }
    const std::string base = GemmHal("gemm_kstore", *t, &f);
    const std::string df = DfGemm("kstore", *t, f);
    const std::string hal = df.empty() ? PickGemm(base) : df;
    const Geom g = GeomOf(hal);
    auto b = df.empty() ? GemmWeights(*t, f) : DfWeights();
    for (const LoomBuffer* x : std::initializer_list<const LoomBuffer*>{&input, wstage_, ostage_, partial_}) b.push_back(Ref(*x));
    if (!df.empty()) DfAhead(b);
    Dispatch(Exe(hal), ("yah_ffn_gemm_" + std::string(df.empty() ? f.name : "f16")).c_str(), MTiles(*t) / g.rowgrp,
             TokenTiles(g), 1, 32, 1, 1, b, GemmWrites(b, f, {partial_}));
    if (!df.empty()) DfNext();
    const std::size_t n = std::size_t{B_} * kHidden;
    Dispatch(Exe("accum.hal"), "yah_residual_1d", static_cast<std::uint32_t>(n / 256), 1, 1, 256, 1, 1,
             {Ref(*hidden_), {partial_->handle, 0, n * 4}, Ref(*hidden2_)});
    std::swap(hidden_, hidden2_);
  }

  void RunAttention(std::uint32_t l, std::uint32_t ci, const std::string& pre, const KvHook& hook, bool q_af, bool k_af,
                    bool v_af) {
    const std::uint32_t ai = l / cfg_.full_attention_interval;
    if (ai >= full_) throw LoomError("full-attention layer index past the KV slot count");
    // an afrag o-projection reads the attention output fragment-major (wmma_t)
    const bool o_af = Af("gemm_kres", pre + "attn_output.weight");
    const std::uint32_t nq = NpuRows("q");
    const std::string qs = nq ? NpuHal(KqgHal(pre + "attn_q.weight", q_af)) : "";
    bool qg_fused = !qs.empty();
    if (qg_fused) {
      // the NPU's heads of q (whole heads of 512 rows [256 q | 256 gate]) beside the GPU's heads and k / v;
      // the norm wrote the row-major input unless q, k and v are all afrag
      const bool tiled = q_af && k_af && v_af;
      const auto a = NpuEncode("q", tiled ? *normt_ : *scratch_, tiled);
      std::size_t w_off = 0, c_off = 0;
      const auto w = NpuDecode("q", pre + "attn_q.weight", nq, w_off);
      std::vector<std::uint32_t> calls;
      const NpuView c = NpuCalls("q", a, w, c_off, calls);
      NpuEnqueue(std::move(calls), "q");
      RunKqgSplit(pre + "attn_q.weight", q_af, qs, nq);
      RunKstore(pre + "attn_k.weight", *kbuf_, k_af);
      RunKstore(pre + "attn_v.weight", *vbuf_, v_af);
      NpuJoin();
      NpuUnpack("npu_unpack_q.hal", c, {Ref(*q_), Ref(*gate_)}, 6);
    } else {
      qg_fused = RunKqg(pre + "attn_q.weight", q_af);
      if (!qg_fused) RunKstore(pre + "attn_q.weight", *qkv_);
      RunKstore(pre + "attn_k.weight", *kbuf_, k_af);
      RunKstore(pre + "attn_v.weight", *vbuf_, v_af);
    }
    if (!qg_fused)
      Dispatch(Exe("unpack.hal"), "yah_unpack_qg", 24, B_, 1, 256, 1, 1, {Ref(*qkv_), Ref(*q_), Ref(*gate_)});
    const std::size_t koff = kv16_scratch_ ? 0 : std::size_t{ai} * kv_cache_ * 2;
    const std::size_t voff = kv16_scratch_ ? kv16_layer_ : std::size_t{full_} * kv_cache_ * 2 + koff;
    const hrx_buffer_ref_t kpool_l{kpool_->handle, std::size_t{ai} * pool_bytes_, pool_bytes_};
    const hrx_buffer_ref_t vtpool_l{vtpool_->handle, std::size_t{ai} * pool_bytes_, pool_bytes_};
    {
      std::vector<hrx_buffer_ref_t> b = {Ref(*q_),
                                         Ref(*kbuf_),
                                         Ref(*vbuf_),
                                         TRef(*Find(pre + "attn_q_norm.weight")),
                                         TRef(*Find(pre + "attn_k_norm.weight")),
                                         Ref(*q16_),
                                         Ref(*kbuf_),
                                         Ref(*kc32_),
                                         Ref(*vc32_),
                                         rope_kpaged_ ? kpool_l : hrx_buffer_ref_t{kv16_->handle, koff, kv16_layer_},
                                         {kv16_->handle, voff, kv16_layer_},
                                         {eps_->handle, 0, 4}};
      if (rope_kpaged_) b.push_back(ptab_ref_);
      Dispatch(ChunkExe("rope", ci), "yah_fused_qk_rope_batched", 28, B_, 1, 256, 1, 1, b);
    }
    if (hook) hook(ai, ci, koff, voff);
    const std::size_t q8off = std::size_t{ai} * kq_bytes_;
    const std::size_t ksoff = std::size_t{ai} * ks_bytes_;
    const std::size_t kmoff = std::size_t{ai} * 4096;
    const std::size_t first = std::size_t{ci} * B_;              // this chunk's first cache row
    const std::size_t f16rows = std::size_t{B_} * kKvRow * 2;    // the chunk's f16 K or V rows
    const std::size_t f16first = kv16_scratch_ ? 0 : first * kKvRow * 2;
    if (attn_kq8_) {
      const std::size_t qrow = kq_bytes_ / T_, srow = ks_bytes_ / T_;
      if (ci == 0)  // channel mean of the first chunk, kept for the later ones
        Dispatch(Exe("kmean.hal"), "yah_kmean", 4, 1, 1, 256, 1, 1,
                 {{kv16_->handle, koff, f16rows}, {kmbuf_->handle, kmoff, 4096}, Ref(*valid_)});
      std::vector<hrx_buffer_ref_t> b = {{kv16_->handle, koff + f16first, f16rows},
                                         {kmbuf_->handle, kmoff, 4096},
                                         {kq8buf_->handle, q8off + first * qrow, B_ * qrow},
                                         {ksbuf_->handle, ksoff + first * srow, B_ * srow}};
      if (kv_paged_) {  // whole-layer pools, rows placed through the page table
        b[2] = {kq8buf_->handle, q8off, kq_bytes_};
        b[3] = {ksbuf_->handle, ksoff, ks_bytes_};
        b.push_back(ptab_ref_);
      }
      Dispatch(kv_paged_ ? ChunkExe("kq8", ci) : Exe("kq8.hal"), attn_kq4_ ? "yah_kq4" : "yah_kq8", (B_ + 1) / 2, 1, 1,
               256, 1, 1, b);
    }
    const std::size_t vqoff = std::size_t{ai} * vq_bytes_, vqsoff = std::size_t{ai} * vqs_bytes_;
    if (attn_vqt_) {
      std::vector<hrx_buffer_ref_t> b = {{kv16_->handle, voff + f16first, f16rows},
                                         {vqbuf_->handle, vqoff, vq_bytes_},
                                         {vqsbuf_->handle, vqsoff, vqs_bytes_}};
      if (kv_paged_) b.push_back(ptab_ref_);
      b.push_back(Ref(*valid_));
      Dispatch(ChunkExe(attn_vq8_ ? "vq8" : "vq4", ci), attn_vq8_ ? "yah_vq8" : "yah_vq4", 4, (B_ + 15) / 16, 1, 256,
               1, 1, b);
    } else if (paged_f16v_) {
      Dispatch(ChunkExe("vtpage", ci), "yah_vtpage", 32, (B_ + 31) / 32, 1, 256, 1, 1,
               {{kv16_->handle, voff, f16rows}, vtpool_l, ptab_ref_});
    } else if (vtrans_) {
      Dispatch(Exe("vtrans.hal"), "yah_transpose_v16", 32, (T_ + 31) / 32, 1, 256, 1, 1,
               {{kv16_->handle, voff, kv_cache_ * 2}, {vt16_->handle, 0, vt_bytes_}});
    }
    if (tail_ == kNoRows) return;  // K / V are written; nothing reads this layer's output
    if (kdeq_) {  // the context so far, this chunk included, as f16 pools for the fp16 attention
      const std::uint32_t rows = (ci + 1) * B_;
      const char* kd = attn_kq4_ ? "kdeq4" : "kdeq8";
      const char* vd = attn_kq4_ ? "vdeq4" : "vdeq8";
      Dispatch(ChunkExe(kd, ci), attn_kq4_ ? "yah_kdeq4" : "yah_kdeq8", (rows * 64 + 255) / 256, 1, 1, 256, 1, 1,
               {{kq8buf_->handle, q8off, kq_bytes_}, {ksbuf_->handle, ksoff, ks_bytes_}, ptab_ref_, Ref(*kdeq_)});
      Dispatch(ChunkExe(vd, ci), attn_kq4_ ? "yah_vdeq4" : "yah_vdeq8", 4 * ((rows + 15) / 16), 1, 1, 256, 1, 1,
               {{vqbuf_->handle, vqoff, vq_bytes_}, {vqsbuf_->handle, vqsoff, vqs_bytes_}, ptab_ref_, Ref(*vdeq_)});
    }
    const bool qrot = qr16_ && ci >= qrot_first_;
    if (qrot)
      Dispatch(Exe("qrot.hal"), "yah_qrot", (B_ * 96 + 255) / 256, 1, 1, 256, 1, 1, {Ref(*q16_), Ref(*qr16_)});
    {
      std::vector<hrx_buffer_ref_t> b = {
          Ref(qrot ? *qr16_ : *q16_),
          Ref(*gate_),
          attn_kq8_     ? hrx_buffer_ref_t{kq8buf_->handle, q8off, kq_bytes_}
          : paged_f16k_ ? kpool_l
                        : hrx_buffer_ref_t{kv16_->handle, koff, kv_cache_ * 2},
          attn_vqt_     ? hrx_buffer_ref_t{vqbuf_->handle, vqoff, vq_bytes_}
          : paged_f16v_ ? vtpool_l
          : vtrans_     ? hrx_buffer_ref_t{vt16_->handle, 0, vt_bytes_}
                        : hrx_buffer_ref_t{kv16_->handle, voff, kv_cache_ * 2},
          Ref(*scratch_),
          Ref(*lse_)};
      if (kdeq_) b = {b[0], b[1], Ref(*kdeq_), Ref(*vdeq_), b[4], b[5]};  // the fp16 kernel's bindings
      if (attn_kq8_ && !kdeq_) b.push_back({ksbuf_->handle, ksoff, ks_bytes_});
      if (attn_vqt_ && !kdeq_) b.push_back({vqsbuf_->handle, vqsoff, vqs_bytes_});
      if (kv_paged_) b.push_back(ptab_ref_);
      Dispatch(ChunkExe(o_af ? "wmma_t" : "wmma", ci), "yah_attn_wmma", (B_ + attn_tpw_ - 1) / attn_tpw_, kHeads / attn_hpw_, 1, 256, 1,
               1, b);
    }
    trim_row_ = tail_;
    RunResidual(pre + "attn_output.weight", *scratch_, o_af);
    trim_row_ = kAllRows;
  }

  void RunDeltaNet(std::uint32_t l, const std::string& pre, bool qkv_af, bool gate_af) {
    const std::uint32_t si = l - l / cfg_.full_attention_interval;
    // alpha / beta first: they read the norm's row-major copy (21 MB), which the qkv / gate GEMMs would evict from the
    // last-level cache (0.18 vs ~0.39 M cycles each)
    const std::uint32_t nq = NpuRows("qkv"), ng = NpuRows("gate");
    const std::string qkv_split = nq ? NpuHal(KstoreHal(pre + "attn_qkv.weight", qkv_af)) : "";
    const std::string gate_split = ng ? NpuHal(KstoreHal(pre + "attn_gate.weight", gate_af)) : "";
    // conv_c: the conv reads the NPU's qkv columns straight from C ("convkq_c.hal"); the unpack writes only the conv ring
    NpuView conv_c{};
    if (!qkv_split.empty() && !gate_split.empty()) {
      // the NPU's rows of qkv and gate beside alpha / beta and the GPU's rows (one job: the same activations)
      const auto a = NpuEncode("qkv", *scratch_, false);
      std::size_t w_off = 0, c_off = 0;
      const auto wq = NpuDecode("qkv", pre + "attn_qkv.weight", nq, w_off);
      const auto wg = NpuDecode("gate", pre + "attn_gate.weight", ng, w_off);
      std::vector<std::uint32_t> calls;
      const NpuView cq = NpuCalls("qkv", a, wq, c_off, calls), cg = NpuCalls("gate", a, wg, c_off, calls);
      NpuEnqueue(std::move(calls), "qkv");
      RunKstore(pre + "ssm_alpha.weight", *alpha_);
      RunKstore(pre + "ssm_beta.weight", *beta_);
      RunKstoreSplit(pre + "attn_qkv.weight", *qkv_, qkv_af, qkv_split, nq);
      RunKstoreSplit(pre + "attn_gate.weight", *gate_, gate_af, gate_split, ng);
      NpuJoin();
      if (geom_.count("convkq_c.hal")) {
        conv_c = cq;
        NpuUnpack("npu_unpack_qkv_tail.hal", cq, {Ref(*qkv_)}, 2);
      } else {
        NpuUnpack("npu_unpack_qkv.hal", cq, {Ref(*qkv_)}, 2);
      }
      NpuUnpack("npu_unpack_gate.hal", cg, {Ref(*gate_)}, 2);
    } else {
      RunKstore(pre + "ssm_alpha.weight", *alpha_);
      RunKstore(pre + "ssm_beta.weight", *beta_);
      RunKstore(pre + "attn_qkv.weight", *qkv_, qkv_af);
      RunKstore(pre + "attn_gate.weight", *gate_, gate_af);
    }
    const hrx_buffer_ref_t cs{conv_state_->handle, std::size_t{si} * kQkv * 4 * 4, std::size_t{kQkv} * 4 * 4};
    const hrx_buffer_ref_t st{state_->handle, std::size_t{si} * kTs * kState * kState * 4,
                              std::size_t{kTs} * kState * kState * 4};
    // The conv with the q / k L2 norm (prep_kq) fused in.
    // dispatch.txt "convtb <tokens>": a workgroup per tokens (gen_conv_kq), else per token
    const auto ctb = geom_.find("convtb");
    const std::uint32_t conv_tb = ctb != geom_.end() ? ctb->second.tokens : 1;
    if (conv_c.length) {
      const Geom& gg = geom_.at("convkq_g.hal");
      const Geom& gc = geom_.at("convkq_c.hal");
      Dispatch(Exe("convkq_g.hal"), "yah_ssm_conv_kq", gg.tokens, B_ / conv_tb, 1, 256, 1, 1,
               {Ref(*qkv_), TRef(*Find(pre + "ssm_conv1d.weight")), cs, Ref(*conv_out_), Ref(*kqbuf_)});
      Dispatch(Exe("convkq_c.hal"), "yah_ssm_conv_kq_c", gc.tokens, gc.tt, 1, gc.rowgrp, 1, 1,
               {{npu_->C().handle, conv_c.offset, conv_c.length},
                TRef(*Find(pre + "ssm_conv1d.weight")),
                cs,
                Ref(*conv_out_)},
               8);
    } else {
      Dispatch(Exe("convkq.hal"), "yah_ssm_conv_kq", 40, B_ / conv_tb, 1, 256, 1, 1,
               {Ref(*qkv_), TRef(*Find(pre + "ssm_conv1d.weight")), cs, Ref(*conv_out_), Ref(*kqbuf_)});
    }
    // One lane index does two jobs: advance the conv ring past the real tokens (i < qkv_size; the next chunk and the
    // decoder read it) and alpha / beta (i < B * heads). So the grid covers max() of the two.
    const std::uint32_t prepab_tiles = (std::max<std::uint32_t>(B_ * kTs, kQkv) + 255u) / 256u;
    Dispatch(Exe("prepab.hal"), "yah_deltanet_prep_ab", prepab_tiles, 1, 1, 256, 1, 1,
             {Ref(*alpha_), Ref(*beta_), TRef(*Find(pre + "ssm_a")), TRef(*Find(pre + "ssm_dt.bias")), Ref(*qkv_), cs,
              Ref(*ab_), Ref(*valid_)});
    // DeltaNet grid: (blocks per head, heads) x 256, blocks per head from the "rowsplit.hal" row group.
    // Split (dnsplit): heads [0, a) into raw_, then [a, kTs) into raw2_. The first part's postnorm depends only on raw_,
    // so it runs beside the second part (whose few workgroups leave most slots free).
    if (dnsplit_a_) {
      Dispatch(Exe("rowsplit_a.hal"), "yah_deltanet", dn_rowgrp_, dnsplit_a_, 1, 256, 1, 1,
               {Ref(*conv_out_), Ref(*kqbuf_), Ref(*ab_), st, Ref(*raw_)});
      Dispatch(Exe("rowsplit_b.hal"), "yah_deltanet", dn_rowgrp_, dnsplit_b_, 1, 256, 1, 1,
               {Ref(*conv_out_), Ref(*kqbuf_), Ref(*ab_), st, Ref(*raw2_)});
    } else {
      Dispatch(Exe("rowsplit.hal"), "yah_deltanet", dn_rowgrp_, kTs, 1, 256, 1, 1,
               {Ref(*conv_out_), Ref(*kqbuf_), Ref(*ab_), st, Ref(*raw_)});
    }
    if (tail_ == kNoRows) return;  // the recurrent state is written; nothing reads this layer's output
    // an afrag ssm_out reads the postnorm output fragment-major (postnorm_t)
    const bool out_af = Af("gemm_kres", pre + "ssm_out.weight");
    const hrx_buffer_ref_t gate{gate_->handle, 0, std::size_t{B_} * kInner * 4};
    if (dnsplit_a_) {
      const std::string pn = out_af ? "postnorm_t" : "postnorm";
      // heads [0, a) are ssm_out's NPU K chunk: with "npupostnormbfp" this part also writes the NPU's input into A and
      // NpuEncode("out") skips the encoder (RunResidual splits only full chunks; an unused A costs nothing)
      const bool bfp = geom_.count("npupostnormbfp") && NpuRows("out") && trim_row_ == kAllRows && tail_ == kAllRows;
      norm_bfp_ = bfp ? "out" : "";
      std::vector<hrx_buffer_ref_t> b{Ref(*raw_), TRef(*Find(pre + "ssm_norm.weight")), gate, Ref(*scratch_)};
      if (bfp) b.push_back({npu_->A().handle, 0, NpuK(5120).a});
      Dispatch(Exe(pn + (bfp ? "_a_bfp.hal" : "_a.hal")), "yah_ssm_postnorm_fp16", dnsplit_a_ * B_ / 8, 1, 1, 256, 1, 1, b);
      Dispatch(Exe(pn + "_b.hal"), "yah_ssm_postnorm_fp16", dnsplit_b_ * B_ / 8, 1, 1, 256, 1, 1,
               {Ref(*raw2_), TRef(*Find(pre + "ssm_norm.weight")), gate, Ref(*scratch_)});
    } else {
      Dispatch(Exe(out_af ? "postnorm_t.hal" : "postnorm.hal"), "yah_ssm_postnorm_fp16", 6 * B_, 1, 1, 256, 1, 1,
               {Ref(*raw_), TRef(*Find(pre + "ssm_norm.weight")), gate, Ref(*scratch_)});
    }
    trim_row_ = tail_;
    RunResidual(pre + "ssm_out.weight", *scratch_, out_af);
    trim_row_ = kAllRows;
  }

  LoomDevice& gpu_;
  const core::Gguf& gguf_;
  const core::Qwen35Config& cfg_;
  std::string dir_;
  hrx_buffer_t weights_;
  std::size_t delta_;
  std::map<std::string, Geom> geom_;
  std::map<std::string, LoomExecutable> exes_;
  std::deque<LoomBuffer> keep_;  // stable addresses
  std::uint32_t B_ = 0, T_ = 0, full_ = 0, pages_ = 0, dn_rowgrp_ = 0, attn_hpw_ = 0, attn_tpw_ = 0;
  std::uint32_t dnsplit_a_ = 0, dnsplit_b_ = 0;  // DeltaNet head split (dispatch.txt "dnsplit"; 0: one dispatch)
  std::uint32_t qrot_first_ = 0;  // first chunk whose attention reads yah_qrot's rotated Q (dispatch.txt "qrot")
  std::uint32_t n_ = 0;  // real tokens of the chunk from the last Embed
  LoomGraph* graph_ = nullptr;  // open while RunLayers records
  std::unique_ptr<LoomGraph> chunk_graph_;
  NpuSplit* npu_ = nullptr;        // EnableNpu
  bool npu_on_ = false;            // this chunk splits with the NPU
  bool npu_planning_ = false;      // EnableNpu's binding pass: Dispatch records nothing
  LoomBuffer* npurem_ = nullptr;   // the GPU's K remainder of the NPU's out / down rows (f32 [tokens][rows])
  bool ffn_bfp_ = false;           // the last ffn unpack wrote down's BFP16 input for its columns (FfnUnpack)
  std::string norm_bfp_;           // the site whose NPU input the last norm wrote (RunNorm npu_site)
  std::uint32_t npu_job_ = 0;      // NPU jobs of this chunk so far (W slot = job % 2)
  std::vector<NpuSplit::Queued> flagged_;  // this chunk's NPU jobs (index = job), queued after the launch
  std::vector<std::vector<PlannedDecode>> planned_dq_;  // per job: its weight decodes (planning pass)
  std::vector<PlannedDecode> predecoded_;               // the next job's, dispatched at the last Join
  std::map<std::string, std::uint32_t> npu_rows_;  // NPU rows per site (dispatch.txt "npusplit_<site>")
  // The last RunLayers graph's dispatches in node order.
  std::vector<Node> nodes_;
  std::unique_ptr<PrefillCalib> calib_;
  int pending_tag_ = -1;
  std::vector<hrx_profile_dispatch_t> session_;   // dispatch timestamps of the open profile session
  std::vector<std::vector<Node>> session_chunks_;  // the graphs launched in it
  bool kv16_scratch_ = false, kv_paged_ = false, vtrans_ = false, rope_kpaged_ = false;
  bool attn_kq4_ = false, attn_kq8_ = false, attn_vq4_ = false, attn_vq8_ = false, attn_vqt_ = false;
  bool paged_f16k_ = false, paged_f16v_ = false;
  std::size_t kv_cache_ = 0, kv16_layer_ = 0, vt_bytes_ = 0, ks_bytes_ = 0, kq_bytes_ = 0, vq_bytes_ = 0,
              vqs_bytes_ = 0, pool_bytes_ = 0;
  hrx_buffer_ref_t ptab_ref_{};
  std::vector<std::uint32_t> host_ids_;  // Embed's staging
  LoomBuffer *grid_iq3s_ = nullptr, *grid_iq3xxs_ = nullptr, *grid_iq2xxs_ = nullptr, *grid_iq2xs_ = nullptr,
             *ksigns_ = nullptr;
  LoomBuffer *hidden_ = nullptr, *reszero_ = nullptr, *sumout_ = nullptr, *scratch_ = nullptr, *qkv_ = nullptr,
             *gate_ = nullptr, *alpha_ = nullptr, *beta_ = nullptr, *q_ = nullptr, *q16_ = nullptr, *kbuf_ = nullptr,
             *vbuf_ = nullptr, *raw_ = nullptr, *raw2_ = nullptr, *conv_out_ = nullptr, *kqbuf_ = nullptr, *ab_ = nullptr,
             *conv_state_ = nullptr, *state_ = nullptr, *kv16_ = nullptr, *kc32_ = nullptr, *vc32_ = nullptr,
             *lse_ = nullptr, *eps_ = nullptr, *ffnup_ = nullptr, *gateffn_ = nullptr, *uwstage_ = nullptr,
             *wstage_ = nullptr, *ostage_ = nullptr, *partial_ = nullptr, *hidden2_ = nullptr, *normed_ = nullptr,
             *ptab_ = nullptr, *vt16_ = nullptr, *kq8buf_ = nullptr, *ksbuf_ = nullptr, *kmbuf_ = nullptr,
             *vqbuf_ = nullptr, *vqsbuf_ = nullptr, *kpool_ = nullptr, *vtpool_ = nullptr, *valid_ = nullptr, *qr16_ = nullptr,
             *kdeq_ = nullptr, *vdeq_ = nullptr, *ids_ = nullptr,
             *normt_ = nullptr;
  std::int64_t keep_rows_ = kAllRows, tail_ = kAllRows, trim_row_ = kAllRows;
};

}  // namespace yah::model

#endif  // YAH_MODEL_LOOM_PREFILL_HPP_
