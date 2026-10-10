// Engine: the GPU text generator. Loads the model, a chunked prefill HAL set and a decode HAL set once, then serves
// Generate() calls one at a time: prefill the prompt in chunks, then decode.
// A prompt that ends inside a chunk finishes either with a padded (masked) chunk or with decode steps, whichever the
// measured costs say is faster for its tail.
#ifndef YAH_MODEL_ENGINE_HPP_
#define YAH_MODEL_ENGINE_HPP_

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/config.hpp"
#include "core/gguf.hpp"
#include "core/tokenizer.hpp"
#include "model/generator.hpp"
#include "model/loom_decoder.hpp"
#include "model/loom_npu.hpp"
#include "model/loom_prefill.hpp"
#include "model/loom_runtime.hpp"

namespace yah::model {

class Engine : public TextGenerator {
 public:
  struct Options {
    std::string model;        // GGUF path
    std::string prefill_hal;  // emit_prefill_pp.py set (chunked, paged)
    std::string decode_hal;   // emit_decode.py set with the prefill's context and YAH_KV
    bool npu = false;         // the NPU computes the trailing rows of the prefill set's split GEMMs (YAH_NPU_SPLIT)
  };
  // How the tokens after the last whole chunk are run.
  enum class Tail { kAuto, kChunk, kDecode };
  void set_tail(Tail t) { tail_ = t; }

  explicit Engine(const Options& o)
      : gguf_(core::Gguf::OpenResident(o.model)),
        cfg_(core::Qwen35Config::FromGguf(gguf_)),
        tokenizer_(core::Tokenizer::FromGguf(gguf_, core::TokenizerConfig::FromGguf(gguf_))),
        weights_(gpu_, gguf_.tensor_data_base(), gguf_.tensor_data_size()),
        prefill_(gpu_, gguf_, cfg_, o.prefill_hal, weights_.handle(), weights_.delta()),
        decoder_(gpu_, gguf_, cfg_, o.decode_hal, weights_.handle(), weights_.delta()) {
    if (decoder_.context() != prefill_.pool_rows())
      throw LoomError("engine: decode set context " + std::to_string(decoder_.context()) + " != prefill pool rows " +
                      std::to_string(prefill_.pool_rows()));
    if (decoder_.kv_bits() != prefill_.kv_bits())
      throw LoomError("engine: the decode set's KV format differs from the prefill's (emit with the same YAH_KV)");
    gpu_.SetSleepSync(200);
    if (o.npu) {
      const NpuPlan plan = prefill_.npu_plan();
      if (plan.a_bytes == 0) throw LoomError("engine: --npu needs a prefill set emitted with YAH_NPU_SPLIT");
      npu_ = std::make_unique<LoomNpuSplit>(gpu_, plan);
      prefill_.EnableNpu(npu_.get());
    }
    // Prefill calibration while serving (model/prefill_calib.hpp) is shelved: it converged too slowly to pay off yet.
    // To resume: prefill_.EnableCalibration(CalibrationPath(o.prefill_hal)) and emit sets with the calibration menu.
    logits_ = gpu_.Allocate(std::size_t{LoomPrefill::kVocab} * 4);
    seed_ = gpu_.Allocate(8);
    const void* th = nullptr;
    tokens_mirror_ = gpu_.AllocateHost((std::size_t{decoder_.context()} + 1) * 4, &th);
    tokens_host_ = static_cast<const std::uint32_t*>(th);
    const core::MetadataValue* name = gguf_.Meta("general.name");
    name_ = name && !name->s.empty() ? name->s : "qwen3.8-27b";
  }

  [[nodiscard]] const core::Tokenizer& tokenizer() const override { return tokenizer_; }
  [[nodiscard]] std::uint32_t context() const override { return prefill_.context(); }
  [[nodiscard]] std::string model_name() const override { return name_; }

  GenerateResult Generate(const std::vector<core::TokenId>& prompt, const GenerateParams& params,
                          const std::function<bool(core::TokenId)>& on_token) override {
    const auto t0 = std::chrono::steady_clock::now();
    const std::uint32_t n = static_cast<std::uint32_t>(prompt.size());
    if (n == 0) throw LoomError("engine: empty prompt");
    for (core::TokenId id : prompt)
      if (id >= LoomPrefill::kVocab) throw LoomError("engine: token id outside the vocabulary");
    if (std::uint64_t{n} + params.max_tokens > context()) throw LoomError("engine: prompt + max_tokens exceeds context");
    GenerateResult r;
    r.prompt_tokens = n;
    const std::uint64_t seed =
        params.sampling.seed ? params.sampling.seed : static_cast<std::uint64_t>(t0.time_since_epoch().count());
    const bool greedy = params.sampling.temperature <= 0.0f;

    // Prompt: whole chunks through the prefill; the tail as one more partial chunk or through decode steps, whichever
    // the measured costs say is cheaper (a partial chunk runs only the GEMM token tiles that hold its tokens).
    prefill_.Reset();
    decoder_.Bind(prefill_.DecoderState());
    const std::uint32_t B = prefill_.chunk(), full = n / B * B, tail = n - full;
    const bool tail_chunk =
        tail && (tail_ == Tail::kChunk || (tail_ == Tail::kAuto && tail * step_ms_ > chunk_ms_ * ChunkShare(tail)));
    const std::uint32_t chunks = full / B + (tail_chunk ? 1 : 0);
    // The host queues every chunk without waiting: one chunk costs the whole prefill's time over its chunks' shares.
    const auto tp = std::chrono::steady_clock::now();
    double shares = 0.0;
    for (std::uint32_t c = 0; c < chunks; ++c) {
      const std::uint32_t valid = std::min(B, n - c * B);
      shares += ChunkShare(valid);
      prefill_.Embed(prompt.data() + std::size_t{c} * B, valid);
      LoomPrefill::KvHook hook;
      if (valid % 16 && decoder_.kv_bits().second != 16) {
        // Quantized V and the prompt ends mid-tile: seed the decoder's open tile with that tile's real V rows.
        const std::int32_t rc[2] = {static_cast<std::int32_t>(valid - valid % 16), static_cast<std::int32_t>(valid % 16)};
        gpu_.Update(seed_, rc, 8);
        hook = [&](std::uint32_t ai, std::uint32_t, std::size_t, std::size_t voff) {
          prefill_.SeedOpenTile(voff, Ref(seed_), decoder_.OpenTile(ai));
        };
      }
      // only the last chunk's last row is read (Head), and only when the prompt ends in it
      const bool head = c + 1 == chunks && (tail_chunk || tail == 0);
      prefill_.RunLayers(c, hook, head ? std::int64_t{valid} - 1 : LoomPrefill::kNoRows);
      if (c + 1 == chunks && (tail_chunk || tail == 0)) prefill_.Head(valid - 1, Ref(logits_));
      if (c + 1 == chunks) {
        gpu_.Synchronize();
        Track(chunk_ms_, (std::chrono::steady_clock::now() - tp) / shares, true);
      }
    }
    prefill_.Collect();
    // Every token, greedy or sampled (LoomDecoder::SetSampling), is picked on the GPU into the decoder's token stream,
    // which the next step reads, so the steps run kAhead ahead of the token the host reads; made[i] marks the copy of
    // the host's next token + i into tokens_host_. At most kAhead steps past a stop run for nothing.
    decoder_.SetSampling(params.sampling.temperature, params.sampling.top_p, seed);
    const bool tail_steps = tail && !tail_chunk;
    const auto ts = std::chrono::steady_clock::now();
    if (tail_steps) {
      decoder_.SetTokens(prompt.data() + full, tail, full);
      for (std::uint32_t pos = full; pos < n; ++pos) decoder_.Step(pos, n);   // the last one picks token n
    } else {
      decoder_.ResumeAt(n);
      if (greedy)
        prefill_.Argmax(Ref(logits_), decoder_.TokenRef(n));
      else
        decoder_.Sample(Ref(logits_), n - 1, decoder_.TokenRef(n));
    }
    std::deque<LoomEvent> made;
    made.push_back(MarkToken(n));
    auto t1 = std::chrono::steady_clock::now();

    // Decode: report each token, then run the step that consumes it.
    r.finish_reason = "length";
    std::uint32_t queued = n;  // the next step to queue
    for (std::uint32_t pos = n;; ++pos) {
      for (; queued < pos + kAhead && queued + 1 < n + params.max_tokens && queued < decoder_.context(); ++queued) {
        decoder_.Step(queued, n);
        made.push_back(MarkToken(queued + 1));
      }
      if (made.empty()) throw LoomError("decoder: position past the set's context");
      gpu_.Wait(made.front());
      made.pop_front();
      const core::TokenId tok = __atomic_load_n(&tokens_host_[pos], __ATOMIC_ACQUIRE);
      if (tok >= LoomDecoder::kVocab) throw LoomError("decoder: token id out of range: " + std::to_string(tok));
      if (pos == n) {   // the first token: the prompt's time ends here
        if (tail_steps) Track(step_ms_, (std::chrono::steady_clock::now() - ts) / tail, true);
        t1 = std::chrono::steady_clock::now();
        r.prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      }
      if (std::find(params.stop_ids.begin(), params.stop_ids.end(), tok) != params.stop_ids.end()) {
        r.finish_reason = "stop";
        break;
      }
      ++r.generated_tokens;
      if (!on_token(tok)) {
        r.finish_reason = "stop";
        break;
      }
      if (r.generated_tokens >= params.max_tokens) break;
    }
    r.decode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
    if (r.generated_tokens > 1) step_ms_ = 0.8 * step_ms_ + 0.2 * r.decode_ms / (r.generated_tokens - 1);
    return r;
  }

 private:
  static hrx_buffer_ref_t Ref(const LoomBuffer& b) { return {b.handle, 0, b.size}; }
  // Decode steps queued ahead of the token the host reads: the next step is queued while the current one runs.
  static constexpr std::uint32_t kAhead = 2;
  // Copies token pos of the decoder's stream to tokens_host_ and marks it.
  LoomEvent MarkToken(std::uint32_t pos) {
    gpu_.Copy(decoder_.TokenRef(pos), {tokens_mirror_.handle, std::size_t{pos} * 4, 4});
    return gpu_.Mark();
  }
  // A chunk of n real tokens costs about this share of a full one: the GEMMs (~92% of a full chunk) run whole 256-token
  // tiles, everything else runs the whole chunk.
  // The prefill calibration's state file: $XDG_CACHE_HOME/yah (else ~/.cache/yah), one per emitted set (its
  // dispatch.txt contents and time; a re-emit starts over). "" when there is no cache directory.
  static std::string CalibrationPath(const std::string& set) {
    namespace fs = std::filesystem;
    const char* xdg = std::getenv("XDG_CACHE_HOME");
    const char* home = std::getenv("HOME");
    if (!(xdg && *xdg) && !(home && *home)) return "";
    const fs::path dir = (xdg && *xdg) ? fs::path(xdg) / "yah" : fs::path(home) / ".cache" / "yah";
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) return "";
    std::ifstream f(set + "/dispatch.txt", std::ios::binary);
    std::string text((std::istreambuf_iterator<char>(f)), {});
    text += std::to_string(fs::last_write_time(set + "/dispatch.txt", ec).time_since_epoch().count());
    std::uint64_t h = 1469598103934665603ull;
    for (unsigned char c : text) h = (h ^ c) * 1099511628211ull;
    char name[40];
    std::snprintf(name, sizeof name, "calib-%016llx.txt", static_cast<unsigned long long>(h));
    return (dir / name).string();
  }

  double ChunkShare(std::uint32_t n) const {
    const std::uint32_t B = prefill_.chunk(), tiles = (n + 255) / 256 * 256;
    return 0.08 + 0.92 * std::min(1.0, double(tiles) / B);
  }
  // Running estimate of a cost in ms; only timings that include the GPU work (synced) count.
  template <class D>
  static void Track(double& est, D elapsed, bool synced) {
    if (synced) est = 0.8 * est + 0.2 * std::chrono::duration<double, std::milli>(elapsed).count();
  }

  core::Gguf gguf_;
  core::Qwen35Config cfg_;
  core::Tokenizer tokenizer_;
  LoomDevice gpu_;
  LoomWeights weights_;
  LoomPrefill prefill_;
  LoomDecoder decoder_;
  std::unique_ptr<LoomNpuSplit> npu_;  // Options::npu; the prefill keeps a pointer
  LoomBuffer logits_, seed_;
  LoomBuffer tokens_mirror_;                    // host-local copy of the decoder's token stream (MarkToken)
  const std::uint32_t* tokens_host_ = nullptr;  // its mapping
  Tail tail_ = Tail::kAuto;
  double chunk_ms_ = 3100.0, step_ms_ = 62.0;  // prefill chunk and decode step costs (pp2048, short context)
  std::string name_;
};

}  // namespace yah::model

#endif  // YAH_MODEL_ENGINE_HPP_
