// loom_forward_pp: the Loom prefill on HRX (model/loom_prefill.hpp) as a command-line tool.
//
// usage: loom_forward_pp <model.gguf> <haldir> <out-prefix> [tokens] [ids-file]
// outputs: <out-prefix>.logits (vocab f32, last token) and <out-prefix>.hidden (B x 5120 f32, token major, last chunk;
//          rows past a partial chunk's tokens are padding)
// tokens: any count up to the set's context; the last chunk may be partial (masked prefill, as the Engine runs it).
// A short ids file is repeated to the token count.
//
// env:
//   YAH_GEN=N, YAH_DECODE_HAL=<emit_decode.py set>  N greedy tokens after the prompt (the prefill's argmax first)
//   YAH_LOGITS_FROM=P   f32 logits of positions P..tokens-1 into <out-prefix>.all_logits (engine/run/gate)
//   YAH_ROWSTATS=<file> compact per-position logit statistics (engine/run/kvq/gate2.py); YAH_ROWSTATS_FROM (1024),
//                       YAH_ROWSTATS_STRIDE (8) or a position list in YAH_ROWSTATS_POS
//   YAH_SLOT_SAVE=<file> [YAH_SLOT_AT=C]  save the sequence state before chunk C (default: after the last)
//   YAH_SLOT_LOAD=<file> restore a slot and run from the chunk it covers (marginal prefill at depth)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/config.hpp"
#include "core/gguf.hpp"
#include "model/loom_decoder.hpp"
#include "model/loom_npu.hpp"
#include "model/loom_prefill.hpp"
#include "model/loom_runtime.hpp"

using namespace yah::model;  // NOLINT(google-build-using-namespace)

namespace {
constexpr std::uint32_t kVocab = LoomPrefill::kVocab;

FILE* OpenOut(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) throw LoomError("cannot write " + path);
  return f;
}

std::vector<std::uint32_t> ParseIds(const char* path) {
  std::ifstream file(path);
  if (!file) throw LoomError(std::string("cannot read ids file ") + path);
  std::vector<std::uint32_t> ids;
  std::uint64_t v;
  while (file >> v) {
    if (v >= kVocab) throw LoomError("token id " + std::to_string(v) + " is outside the vocabulary");
    ids.push_back(static_cast<std::uint32_t>(v));
  }
  if (ids.empty()) throw LoomError("ids file is empty");
  return ids;
}

// One YAH_ROWSTATS record per position (little endian): int32 pos, int32 next-token id (-1 at the end), f32 logsumexp,
// f32 logit of the next token, int32 argmax, f32 top1, f32 top2, then 64 x (int32 id, f32 logit), highest first.
void WriteRowStats(FILE* f, const float* lg, std::uint32_t pos, std::int32_t nxt, std::vector<std::uint32_t>& idx) {
  double mx = lg[0];
  for (std::uint32_t v = 1; v < kVocab; ++v) mx = std::max<double>(mx, lg[v]);
  double se = 0.0;
  for (std::uint32_t v = 0; v < kVocab; ++v) se += std::exp(static_cast<double>(lg[v]) - mx);
  const float lse = static_cast<float>(mx + std::log(se));
  for (std::uint32_t v = 0; v < kVocab; ++v) idx[v] = v;
  std::partial_sort(idx.begin(), idx.begin() + 64, idx.end(),
                    [&](std::uint32_t a, std::uint32_t b) { return lg[a] > lg[b] || (lg[a] == lg[b] && a < b); });
  const std::int32_t ipos = static_cast<std::int32_t>(pos), am = static_cast<std::int32_t>(idx[0]);
  const float lt = nxt >= 0 ? lg[nxt] : 0.0f, t1 = lg[idx[0]], t2 = lg[idx[1]];
  std::fwrite(&ipos, 4, 1, f);
  std::fwrite(&nxt, 4, 1, f);
  std::fwrite(&lse, 4, 1, f);
  std::fwrite(&lt, 4, 1, f);
  std::fwrite(&am, 4, 1, f);
  std::fwrite(&t1, 4, 1, f);
  std::fwrite(&t2, 4, 1, f);
  for (int k = 0; k < 64; ++k) {
    const std::int32_t id = static_cast<std::int32_t>(idx[k]);
    std::fwrite(&id, 4, 1, f);
    std::fwrite(&lg[idx[k]], 4, 1, f);
  }
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: loom_forward_pp <model.gguf> <haldir> <out-prefix> [tokens] [ids-file]\n");
    return 2;
  }
  const std::string prefix = argv[3];
  const std::uint32_t want = argc > 4 ? std::strtoul(argv[4], nullptr, 10) : 2048;
  const char* ids_path = argc > 5 ? argv[5] : "/home/q/yah-scratch/ids2048.txt";
  try {
    auto gguf = yah::core::Gguf::Open(argv[1]);
    const auto cfg = yah::core::Qwen35Config::FromGguf(gguf);
    std::vector<std::uint32_t> ids = ParseIds(ids_path);
    LoomDevice gpu;
    // The layers queue ~950 dispatches and wait once: the runtime wait would busy-poll a host core the whole time.
    gpu.SetSleepSync(200);
    LoomWeights weights(gpu, gguf.tensor_data_base(), gguf.tensor_data_size());
    LoomPrefill pf(gpu, gguf, cfg, argv[2], weights.handle(), weights.delta(), want);
    // YAH_NPU=1: the NPU computes the trailing rows of the set's split GEMMs (a set emitted with YAH_NPU_SPLIT).
    std::unique_ptr<yah::model::LoomNpuSplit> npu;
    if (const char* e = std::getenv("YAH_NPU"); e && std::string(e) == "1") {
      const auto plan = pf.npu_plan();
      if (plan.a_bytes == 0) {
        std::fprintf(stderr, "YAH_NPU=1: the set has no NPU split (emit it with YAH_NPU_SPLIT)\n");
        return 2;
      }
      npu = std::make_unique<yah::model::LoomNpuSplit>(gpu, plan);
      pf.EnableNpu(npu.get());
    }
    const std::uint32_t B = pf.chunk(), T_run = want ? want : B;
    if (T_run == 0 || T_run > pf.context()) {
      std::fprintf(stderr, "tokens=%u must be at most the context %u\n", T_run, pf.context());
      return 2;
    }
    for (std::size_t i = ids.size(), n = ids.size(); i < T_run; ++i) ids.push_back(ids[i % n]);
    const std::uint32_t n_chunks = (T_run + B - 1) / B, last_n = T_run - (n_chunks - 1) * B;
    std::fprintf(stderr, "loom_forward_pp: tokens=%u chunk=%u context=%u\n", T_run, B, pf.context());

    std::uint32_t logits_from = T_run;
    if (const char* lf = std::getenv("YAH_LOGITS_FROM")) {
      logits_from = static_cast<std::uint32_t>(std::atoi(lf));
      if (logits_from >= T_run) throw LoomError("YAH_LOGITS_FROM must be below the token count");
    }
    LoomBuffer every = gpu.Allocate(std::size_t{T_run - logits_from + (logits_from == T_run)} * kVocab * 4);
    FILE* rowstats = nullptr;
    std::vector<char> rs_want(pf.context(), 0);
    if (const char* rf = std::getenv("YAH_ROWSTATS")) {
      rowstats = OpenOut(rf);
      if (const char* pfile = std::getenv("YAH_ROWSTATS_POS")) {
        std::ifstream pfs(pfile);
        std::uint32_t pp;
        while (pfs >> pp)
          if (pp < pf.context()) rs_want[pp] = 1;
      } else {
        const char* rs_from = std::getenv("YAH_ROWSTATS_FROM");
        const char* rs_stride = std::getenv("YAH_ROWSTATS_STRIDE");
        const std::uint32_t from = rs_from ? std::atoi(rs_from) : 1024, stride = rs_stride ? std::atoi(rs_stride) : 8;
        for (std::uint32_t pp = from; pp < pf.context(); pp += stride) rs_want[pp] = 1;
      }
    }
    std::uint32_t rs_max = 1;  // rows per chunk
    for (std::uint32_t c = 0; c < n_chunks; ++c)
      rs_max = std::max<std::uint32_t>(
          rs_max, std::count(rs_want.begin() + std::size_t{c} * B, rs_want.begin() + std::size_t{c + 1} * B, 1));
    LoomBuffer rsbuf = gpu.Allocate(std::size_t{rs_max} * kVocab * 4);

    // Slots (LoomPrefill::SaveState): YAH_SLOT_SAVE=<file> writes the state before chunk YAH_SLOT_AT (default: after the
    // last chunk); YAH_SLOT_LOAD=<file> restores it and runs from the chunk it covers, so layers_ms is the marginal time.
    std::uint32_t ci0 = 0;
    if (const char* sl = std::getenv("YAH_SLOT_LOAD")) {
      ci0 = pf.LoadState(sl);
      if (ci0 >= n_chunks) throw LoomError("the slot covers every chunk of this run");
      std::fprintf(stderr, "loom_forward_pp: slot %s covers %u chunks; running from token %u\n", sl, ci0, ci0 * B);
    }
    const char* slot_save = std::getenv("YAH_SLOT_SAVE");
    const std::uint32_t slot_at = std::getenv("YAH_SLOT_AT") ? std::atoi(std::getenv("YAH_SLOT_AT")) : n_chunks;

    gpu.Synchronize();
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::uint32_t> idx(kVocab);
    for (std::uint32_t ci = ci0; ci < n_chunks; ++ci) {
      if (slot_save && ci == slot_at) pf.SaveState(slot_save, ci);
      pf.Embed(ids.data() + std::size_t{ci} * B, ci + 1 == n_chunks ? last_n : B);
      // the last layer's tail only for the rows read below: every row for logits_from / rowstats, else the last row
      const bool rows = rowstats || logits_from < std::min((ci + 1) * B, T_run);
      pf.RunLayers(ci, {}, rows ? LoomPrefill::kAllRows
                       : ci + 1 == n_chunks ? std::int64_t{last_n} - 1
                                            : LoomPrefill::kNoRows);
      for (std::uint32_t ra = std::max(logits_from, ci * B); ra < std::min((ci + 1) * B, T_run); ++ra)
        pf.Head(ra - ci * B, {every.handle, std::size_t{ra - logits_from} * kVocab * 4, std::size_t{kVocab} * 4});
      if (rowstats) {
        std::vector<std::uint32_t> rows;
        for (std::uint32_t r = 0; r < (ci + 1 == n_chunks ? last_n : B); ++r)
          if (rs_want[std::size_t{ci} * B + r]) rows.push_back(r);
        for (std::size_t i = 0; i < rows.size(); ++i)
          pf.Head(rows[i], {rsbuf.handle, i * kVocab * 4, std::size_t{kVocab} * 4});
        if (!rows.empty()) {
          gpu.Synchronize();
          std::vector<float> host(rows.size() * kVocab);
          gpu.D2H(rsbuf, host.data(), host.size() * 4, 0);
          for (std::size_t i = 0; i < rows.size(); ++i) {
            const std::uint32_t pos = ci * B + rows[i];
            const std::int32_t nxt = pos + 1 < T_run ? static_cast<std::int32_t>(ids[pos + 1]) : -1;
            WriteRowStats(rowstats, host.data() + i * kVocab, pos, nxt, idx);
          }
        }
      }
    }
    gpu.Synchronize();
    if (slot_save && slot_at >= n_chunks) pf.SaveState(slot_save, n_chunks);
    if (rowstats) std::fclose(rowstats);
    const double layer_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("layers_ms=%.1f\n", layer_ms);

    {
      std::vector<float> out(std::size_t{B} * LoomPrefill::kHidden);
      gpu.D2H(pf.hidden(), out.data(), out.size() * 4, 0);
      FILE* fo = OpenOut(prefix + ".hidden");
      std::fwrite(out.data(), 4, out.size(), fo);
      std::fclose(fo);
    }
    LoomBuffer logits = gpu.Allocate(std::size_t{kVocab} * 4);
    LoomBuffer token = gpu.Allocate(4);
    pf.Head(last_n - 1, {logits.handle, 0, logits.size});
    pf.Argmax({logits.handle, 0, logits.size}, {token.handle, 0, 4});
    gpu.Synchronize();
    std::uint32_t tok = 0;
    gpu.D2H(token, &tok, 4, 0);
    std::vector<float> logit_host(kVocab);
    gpu.D2H(logits, logit_host.data(), logit_host.size() * 4, 0);
    FILE* fl = OpenOut(prefix + ".logits");
    std::fwrite(logit_host.data(), 4, logit_host.size(), fl);
    std::fclose(fl);
    std::printf("argmax=%u\n", tok);

    if (const char* gv = std::getenv("YAH_GEN"); gv && std::atoi(gv) > 0) {
      const std::uint32_t ngen = static_cast<std::uint32_t>(std::atoi(gv));
      const char* ddir = std::getenv("YAH_DECODE_HAL");
      if (!ddir) throw LoomError("YAH_GEN needs YAH_DECODE_HAL (tools/emit_decode.py set)");
      if (T_run + ngen - 1 > pf.context()) throw LoomError("YAH_GEN: prompt + gen exceeds the emitted context");
      LoomDecoder dec(gpu, gguf, cfg, ddir, weights.handle(), weights.delta());
      if (dec.context() != pf.pool_rows())
        throw LoomError("YAH_GEN: decode set context " + std::to_string(dec.context()) + " != prefill pool rows " +
                        std::to_string(pf.pool_rows()));
      if (dec.kv_bits() != pf.kv_bits())
        throw LoomError("YAH_GEN: the decode set's KV format differs from the prefill's (emit with the same YAH_KV)");
      dec.Bind(pf.DecoderState());
      dec.SetTokens(&tok, 1, T_run);
      gpu.Synchronize();
      const auto tg = std::chrono::steady_clock::now();
      for (std::uint32_t pos = T_run; pos + 1 < T_run + ngen; ++pos) dec.Step(pos, T_run);
      gpu.Synchronize();
      const double gms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tg).count();
      const std::vector<std::uint32_t> gen = dec.Tokens(T_run, T_run + ngen);
      std::printf("generated_ids=");
      for (std::size_t i = 0; i < gen.size(); ++i) std::printf("%u%s", gen[i], i + 1 == gen.size() ? "" : " ");
      std::printf("\n");
      if (ngen > 1)
        std::printf("decode_ms=%.2f decode_tok_s=%.2f (context %u)\n", gms / (ngen - 1), 1000.0 * (ngen - 1) / gms,
                    T_run);
    }

    if (logits_from < T_run) {
      const std::size_t rows = T_run - logits_from;
      std::vector<float> host(rows * kVocab);
      gpu.D2H(every, host.data(), host.size() * 4, 0);
      FILE* fa = OpenOut(prefix + ".all_logits");
      std::fwrite(host.data(), 4, host.size(), fa);
      std::fclose(fa);
      std::printf("all_logits rows=%zu from=%u\n", rows, logits_from);
    }
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "loom_forward_pp: %s\n", error.what());
    return 1;
  }
}
