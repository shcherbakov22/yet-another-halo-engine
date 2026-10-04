// gemm_bench: runs tile-GEMM HAL variants on real weights, one process, for the autotuner (engine/tune).
//
// usage: gemm_bench <model.gguf> <table dir> <job file>
//
// Each job line: <hal> <export> <tensors> <fmt> <kind> <row groups> <tokens per workgroup> <tokens> <reps> <roles>
//   tensors: comma separated, same shape (or file:<path>:<rows>, raw weight bytes); the dispatches rotate through them like the layers of a forward pass, so the
//   weights stream from DRAM as in the pipeline instead of staying in the last-level cache.
//   roles: the kernel's bindings in order, comma separated (weight, grid, ksigns, input, gate, resid, wstage, ostage,
//   output, gate_out). The table dir holds grid_<fmt>.bin / ksigns_iq2xxs.bin (any prefill HAL set).
// Every binding is at least as large as loom_forward_pp's largest use of it (chunk 2048), so a HAL that passed the
// emitter's footprint gate is in bounds here too. The grid is (m_tiles / row groups, ceil(tokens / tile)): a subset of
// the compiled grid whenever tokens <= the chunk.
// Per job it prints "job <i> wall_ms <ms per rep> hash <h>": h hashes the output of the real token rows only (variants
// pad differently) of one dispatch on the first tensor. Run under HRX_PROFILE_MODE=counters for clock-free cycles: the
// dispatches appear in job order, 2 warmups + 1 hashed + reps per job.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "core/gguf.hpp"
#include "model/loom_runtime.hpp"

namespace {

using yah::model::LoomBuffer;
using yah::model::LoomDevice;
using yah::model::LoomError;
using yah::model::LoomExecutable;

constexpr std::size_t kChunk = 2048, kFfn = 17408;

struct Job {
  std::string hal, exp, tensor, fmt, kind;
  std::uint32_t rowgrp = 0, tile = 0, tokens = 0, reps = 0;
  std::vector<std::string> roles;
};

std::vector<Job> ReadJobs(const std::string& path) {
  std::vector<Job> jobs;
  std::ifstream f(path);
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream s(line);
    Job j;
    std::string roles;
    if (!(s >> j.hal >> j.exp >> j.tensor >> j.fmt >> j.kind >> j.rowgrp >> j.tile >> j.tokens >> j.reps >> roles))
      throw LoomError("bad job line: " + line);
    for (std::size_t a = 0, b; a <= roles.size(); a = b + 1) {
      b = roles.find(',', a);
      if (b == std::string::npos) b = roles.size();
      j.roles.push_back(roles.substr(a, b - a));
    }
    jobs.push_back(std::move(j));
  }
  return jobs;
}

// Small finite values: f16 0.5..1 or f32 -1..1 with a fixed sequence, the same for every variant.
std::vector<std::uint8_t> Pattern(std::size_t bytes, bool f16, std::uint32_t seed) {
  std::vector<std::uint8_t> v(bytes);
  std::uint32_t x = seed;
  if (f16) {
    auto* h = reinterpret_cast<std::uint16_t*>(v.data());
    for (std::size_t i = 0; i < bytes / 2; ++i) {
      x = x * 1664525u + 1013904223u;
      h[i] = static_cast<std::uint16_t>(((x >> 16) & 0x8000u) | 0x3800u | ((x >> 8) & 0x3ffu));
    }
  } else {
    auto* g = reinterpret_cast<float*>(v.data());
    for (std::size_t i = 0; i < bytes / 4; ++i) {
      x = x * 1664525u + 1013904223u;
      g[i] = static_cast<float>(static_cast<std::int32_t>(x >> 8) - (1 << 23)) / static_cast<float>(1 << 23);
    }
  }
  return v;
}

std::uint64_t Hash(const std::vector<std::uint8_t>& v, std::uint64_t h = 1469598103934665603ull) {
  for (std::uint8_t c : v) h = (h ^ c) * 1099511628211ull;
  return h;
}

std::vector<std::uint8_t> ReadFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw LoomError("cannot read " + path);
  return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(f), {});
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: gemm_bench <model.gguf> <table dir> <job file>\n");
    return 2;
  }
  try {
    const auto gguf = yah::core::Gguf::Open(argv[1]);
    const std::string tables = argv[2];
    const std::vector<Job> jobs = ReadJobs(argv[3]);
    LoomDevice gpu;
    // Shared bindings, sized for the largest use in loom_forward_pp at chunk 2048.
    LoomBuffer input = gpu.Allocate(kChunk * kFfn * 2), gate = gpu.Allocate(kChunk * kFfn * 4),
               resid = gpu.Allocate(kChunk * kFfn * 4), wstage = gpu.Allocate(kFfn * 16 * 2),
               ostage = gpu.Allocate(kFfn * kChunk * 4), output = gpu.Allocate(kChunk * kFfn * 4),
               gate_out = gpu.Allocate(kChunk * kFfn * 4), ksigns = gpu.Allocate(128);
    const std::vector<std::uint8_t> input_host = Pattern(input.size, true, 1);
    gpu.H2D(input, input_host.data(), input.size);
    // "input_tiled": the same activations [token][K] laid out fragment-major for Tile.atiled: 16 x 16 tiles
    // (token / 16, k / 16) of 512 contiguous bytes, token-major tiles, k fastest inside; built per K on demand
    LoomBuffer input_tiled = gpu.Allocate(input.size);
    std::uint32_t tiled_k = 0;
    gpu.H2D(gate, Pattern(gate.size, false, 2).data(), gate.size);
    gpu.H2D(resid, Pattern(resid.size, false, 3).data(), resid.size);
    {
      const auto k = ReadFile(tables + "/ksigns_iq2xxs.bin");
      if (k.size() != 128) throw LoomError("ksigns_iq2xxs.bin is not 128 bytes");
      gpu.H2D(ksigns, k.data(), 128);
    }
    std::vector<LoomBuffer> weights;  // the current job's tensors; consecutive jobs of one kernel reuse them
    std::string weight_names;
    for (std::size_t i = 0; i < jobs.size(); ++i) {
      const Job& j = jobs[i];
      // "file:<path>:<rows>": raw weight bytes from a file (e.g. pre-dequantized f16), instead of GGUF tensors
      std::vector<std::uint8_t> file_w;
      std::uint32_t m_rows = 0, tiled_dims0 = 0;
      std::vector<const yah::core::TensorInfo*> ts;
      if (j.tensor.rfind("file:", 0) == 0) {
        const std::size_t c = j.tensor.rfind(':');
        file_w = ReadFile(j.tensor.substr(5, c - 5));
        m_rows = static_cast<std::uint32_t>(std::stoul(j.tensor.substr(c + 1)));
      } else if (j.kind == "ffn" && j.tensor.find('+') != std::string::npos) {
        // "<gate>+<up>": a mixed-format fused gate / up GEMM's one binding, the two tensors back to back
        const std::size_t c = j.tensor.find('+');
        const auto* g = gguf.Find(j.tensor.substr(0, c));
        const auto* u = gguf.Find(j.tensor.substr(c + 1));
        if (!g || !u || g->dims[1] != u->dims[1]) throw LoomError("ffn pair not found or of different rows: " + j.tensor);
        file_w.resize(g->bytes + u->bytes);
        std::memcpy(file_w.data(), gguf.Data(*g), g->bytes);
        std::memcpy(file_w.data() + g->bytes, gguf.Data(*u), u->bytes);
        m_rows = static_cast<std::uint32_t>(g->dims[1]);
        tiled_dims0 = static_cast<std::uint32_t>(g->dims[0]);
      } else {
        for (std::size_t a = 0, b; a <= j.tensor.size(); a = b + 1) {
          b = j.tensor.find(',', a);
          if (b == std::string::npos) b = j.tensor.size();
          const auto* t = gguf.Find(j.tensor.substr(a, b - a));
          if (!t || t->dims.size() < 2) throw LoomError("tensor not found: " + j.tensor.substr(a, b - a));
          ts.push_back(t);
        }
        for (const auto* u : ts)
          if (u->bytes != ts[0]->bytes || u->dims != ts[0]->dims) throw LoomError(j.hal + ": tensors of different shapes");
        m_rows = static_cast<std::uint32_t>(ts[0]->dims[1]);
      }
      if (j.tokens == 0 || j.tokens > kChunk || m_rows % 16 || (m_rows / 16) % j.rowgrp)
        throw LoomError(j.hal + ": bad shape or token count");
      if (weight_names != j.tensor + j.kind) {
        weights.clear();
        for (const auto* u : ts) {
          // ffn (fused gate / up): the binding spans gate then up; here the same tensor twice
          const int copies = j.kind == "ffn" ? 2 : 1;
          weights.push_back(gpu.Allocate(u->bytes * copies));
          for (int c = 0; c < copies; ++c) gpu.H2D(weights.back(), gguf.Data(*u), u->bytes, u->bytes * c);
        }
        if (!file_w.empty()) {
          weights.push_back(gpu.Allocate(file_w.size()));
          gpu.H2D(weights.back(), file_w.data(), file_w.size());
        }
        weight_names = j.tensor + j.kind;
      }
      LoomBuffer grid;
      // a mixed ffn ("<gate>:<up>") binds the grid of whichever of its formats has one (the generator allows only one)
      std::string gfmt = j.fmt;
      for (std::size_t a = 0, b; a <= j.fmt.size(); a = b + 1) {
        b = j.fmt.find(':', a);
        if (b == std::string::npos) b = j.fmt.size();
        const std::string f = j.fmt.substr(a, b - a);
        if (f == "iq3s" || f == "iq3xxs" || f == "iq2xxs" || f == "iq2xs") gfmt = f;
      }
      const bool needs_grid = gfmt == "iq3s" || gfmt == "iq3xxs" || gfmt == "iq2xxs" || gfmt == "iq2xs";
      if (needs_grid) {
        const auto g = ReadFile(tables + "/grid_" + gfmt + ".bin");
        grid = gpu.Allocate(g.size());
        gpu.H2D(grid, g.data(), g.size());
      }
      const std::uint32_t k_dim = ts.empty() ? tiled_dims0 : static_cast<std::uint32_t>(ts[0]->dims[0]);
      if (std::find(j.roles.begin(), j.roles.end(), "input_tiled") != j.roles.end() && tiled_k != k_dim) {
        if (!k_dim || k_dim % 16) throw LoomError(j.hal + ": input_tiled needs a GGUF weight with K % 16 == 0");
        const auto* src = reinterpret_cast<const std::uint16_t*>(input_host.data());
        std::vector<std::uint16_t> t(std::size_t{kChunk} * k_dim);
        for (std::size_t tok = 0; tok < kChunk; ++tok)
          for (std::size_t k = 0; k < k_dim; ++k)
            t[((tok / 16) * (k_dim / 16) + k / 16) * 256 + (tok % 16) * 16 + k % 16] = src[tok * k_dim + k];
        gpu.H2D(input_tiled, t.data(), t.size() * 2);
        tiled_k = k_dim;
      }
      std::vector<hrx_buffer_ref_t> b;
      for (const std::string& r : j.roles) {
        const LoomBuffer* x = r == "weight"    ? &weights[0]
                              : r == "grid"    ? (needs_grid ? &grid : nullptr)
                              : r == "ksigns"  ? &ksigns
                              : r == "input"   ? &input
                              : r == "input_tiled" ? &input_tiled
                              : r == "gate"    ? &gate
                              : r == "resid"   ? &resid
                              : r == "wstage"  ? &wstage
                              : r == "ostage"  ? &ostage
                              : r == "output"  ? &output
                              : r == "gate_out" ? &gate_out
                                                : nullptr;
        if (!x) throw LoomError(j.hal + ": unknown binding " + r);
        b.push_back({x->handle, 0, x->size});
      }
      LoomExecutable exe = gpu.Load(j.hal);
      const std::uint32_t ord = exe.OrdinalOrZero(j.exp);
      const std::uint32_t ws = exe.WorkgroupSize(ord);
      if (!ws) throw LoomError(j.hal + ": no workgroup size in the export metadata");
      const auto cfg = LoomDevice::Config(m_rows / 16 / j.rowgrp, (j.tokens + j.tile - 1) / j.tile, 1, ws, 1, 1);
      // binding 0..: the role "weight" is rebound to each tensor in turn
      std::size_t wslot = 0;
      while (wslot < j.roles.size() && j.roles[wslot] != "weight") ++wslot;
      if (wslot == j.roles.size()) throw LoomError(j.hal + ": no weight binding");
      auto with_weight = [&](std::size_t r) {
        b[wslot] = {weights[r % weights.size()].handle, 0, weights[r % weights.size()].size};
        return b.data();
      };
      for (int w = 0; w < 2; ++w) gpu.Dispatch(exe, ord, cfg, nullptr, 0, with_weight(w + 1), b.size());
      gpu.Fill(output, 0);
      gpu.Fill(gate_out, 0);
      gpu.Dispatch(exe, ord, cfg, nullptr, 0, with_weight(0), b.size());
      gpu.Synchronize();
      // The real token rows of the outputs: [token][rows], f16 for swiglu, q and gate halves for kqg.
      const std::size_t row_bytes = j.kind == "swiglu" || j.kind == "ffn" ? std::size_t{m_rows} * 2
                                    : j.kind == "kqg"  ? std::size_t{m_rows} / 2 * 4
                                                       : std::size_t{m_rows} * 4;
      std::vector<std::uint8_t> host(row_bytes * j.tokens);
      gpu.D2H(output, host.data(), host.size());
      std::uint64_t h = Hash(host);
      if (j.kind == "kqg") {
        gpu.D2H(gate_out, host.data(), host.size());
        h = Hash(host, h);
      }
      const auto t0 = std::chrono::steady_clock::now();
      for (std::uint32_t r = 0; r < j.reps; ++r) gpu.Dispatch(exe, ord, cfg, nullptr, 0, with_weight(r + 1), b.size());
      gpu.Synchronize();
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      std::printf("job %zu wall_ms %.4f hash %016llx\n", i, j.reps ? ms / j.reps : 0.0,
                  static_cast<unsigned long long>(h));
      std::fflush(stdout);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "gemm_bench: %s\n", e.what());
    return 1;
  }
  return 0;
}
