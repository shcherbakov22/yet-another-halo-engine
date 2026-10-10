// loom_decode: single-token decode on Loom through HRX (engine/model/loom_decoder.hpp).
//
// usage: loom_decode <model.gguf> <hal_dir> --ids "1 2 3" [--gen N] [--logits FILE]
//
// The prompt goes through the decode path one token at a time (positions 0..n-1), then N tokens are generated greedily.
// HAL set: tools/emit_decode.py. Steps queue back to back.
// The host waits after the prompt (to time generation alone) and at the end; --logits / YAH_DEC_TRACE wait every step.
// --logits FILE appends every step's 248320 logits (f32) for an external KL gate.
// For decode after a Loom prefill, see loom_forward_pp with YAH_GEN.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include "core/config.hpp"
#include "core/gguf.hpp"
#include "core/tokenizer.hpp"
#include "model/loom_decoder.hpp"
#include "model/loom_runtime.hpp"

using namespace yah::model;  // NOLINT(google-build-using-namespace)

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: loom_decode <model.gguf> <hal_dir> --ids \"1 2\" [--gen N] [--logits FILE]\n");
    return 2;
  }
  const std::string model = argv[1], dir = argv[2];
  std::string ids_arg, logits_path;
  std::uint32_t gen_count = 16;
  for (int i = 3; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--ids" && i + 1 < argc)
      ids_arg = argv[++i];
    else if (a == "--gen" && i + 1 < argc)
      gen_count = static_cast<std::uint32_t>(std::atoi(argv[++i]));
    else if (a == "--logits" && i + 1 < argc)
      logits_path = argv[++i];
    else {
      std::fprintf(stderr, "loom_decode: unknown argument %s\n", a.c_str());
      return 2;
    }
  }
  try {
    auto gguf = yah::core::Gguf::OpenResident(model);
    const auto cfg = yah::core::Qwen35Config::FromGguf(gguf);
    const auto tconfig = yah::core::TokenizerConfig::FromGguf(gguf);
    const auto tokenizer = yah::core::Tokenizer::FromGguf(gguf, tconfig);
    std::vector<std::uint32_t> prompt;
    {
      std::istringstream in(ids_arg);
      std::uint32_t v;
      while (in >> v) prompt.push_back(v);
    }
    if (prompt.empty() || gen_count == 0) throw LoomError("--ids and --gen >= 1 are required");

    LoomDevice gpu;
    LoomBuffer weights;
    std::size_t delta = 0;
    {
      const std::uint8_t* wbase = gguf.tensor_data_base();
      const std::uintptr_t start = reinterpret_cast<std::uintptr_t>(wbase) & ~std::uintptr_t{4095};
      delta = reinterpret_cast<std::uintptr_t>(wbase) - start;
      weights = gpu.Import(reinterpret_cast<void*>(start), gguf.tensor_data_size() + delta);
    }
    LoomDecoder dec(gpu, gguf, cfg, dir, weights.handle, delta);
    const std::uint32_t n = static_cast<std::uint32_t>(prompt.size());
    const std::uint32_t steps = n + gen_count - 1;
    if (steps > dec.context()) throw LoomError("prompt + gen exceeds the set's max context");
    dec.Bind(dec.OwnState());
    dec.SetTokens(prompt.data(), prompt.size(), 0);

    std::FILE* lf = logits_path.empty() ? nullptr : std::fopen(logits_path.c_str(), "wb");
    std::vector<float> host_logits(lf ? LoomDecoder::kVocab : 0);
    const bool sync_steps = lf || std::getenv("YAH_DEC_TRACE");
    std::vector<double> step_ms;
    auto tgen = std::chrono::steady_clock::now();
    for (std::uint32_t pos = 0; pos < steps; ++pos) {
      const auto t0 = std::chrono::steady_clock::now();
      dec.Step(pos, n);
      if (sync_steps) {
        gpu.Synchronize();
        if (lf) {
          dec.CopyLogits(host_logits.data());
          std::fwrite(host_logits.data(), 4, host_logits.size(), lf);
        }
        step_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
      }
      if (pos + 1 == n) {
        gpu.Synchronize();
        tgen = std::chrono::steady_clock::now();
      }
    }
    gpu.Synchronize();
    const double gen_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tgen).count();
    if (lf) std::fclose(lf);
    const std::vector<std::uint32_t> gen = dec.Tokens(n, steps + 1);
    if (!step_ms.empty()) {
      std::fprintf(stderr, "step_ms=");
      for (std::size_t i = 0; i < step_ms.size(); ++i)
        std::fprintf(stderr, "%.1f%s", step_ms[i], i + 1 == step_ms.size() ? "\n" : " ");
    }
    // Decode rate over the generation steps (positions n .. steps - 1).
    const std::uint32_t dn = steps - n;
    if (dn) std::fprintf(stderr, "decode_ms=%.2f decode_tok_s=%.2f\n", gen_ms / dn, 1000.0 * dn / gen_ms);
    std::printf("generated_ids=");
    for (std::size_t i = 0; i < gen.size(); ++i) std::printf("%u%s", gen[i], i + 1 == gen.size() ? "" : " ");
    std::printf("\ngenerated_text=%s\n", tokenizer.Decode(gen).c_str());
  } catch (const std::exception& error) {
    std::fprintf(stderr, "loom_decode: %s\n", error.what());
    return 1;
  }
  return 0;
}
