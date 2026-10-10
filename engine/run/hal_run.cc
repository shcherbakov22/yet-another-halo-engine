// hal_run: dispatches one HAL once with bindings from GGUF tensors and files, then writes the output bindings to files.
// A correctness harness for generated kernels (e.g. gen_gemv.py decode GEMVs) against an external numpy oracle.
//
// usage: hal_run <model.gguf> <hal> <gx[,gy]> <wg_size> <min_sizes> <binding>...
//   min_sizes  comma-separated byte sizes, one per binding: the kernel's declared footprint (gen_gemv.footprint).
//              A smaller binding stops the run before dispatch (exit 3): a read past an allocation hangs this GPU.
//   binding    t:<tensor>          a GGUF tensor, bound in place (exact bytes)
//              f:<file>            a device buffer initialized from the file
//              o:<bytes>:<file>    a zeroed device buffer, written to <file> after the run
//              io:<in>:<out>       initialized from <in>, written to <out> after the run
//              z:<bytes>           an uninitialized device buffer, not read back
//   env        HAL_RUN_TIME1=1 times the first dispatch; HAL_RUN_ITERS=N times N more dispatches.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "core/gguf.hpp"
#include "model/loom_runtime.hpp"

using namespace yah::model;  // NOLINT(google-build-using-namespace)

namespace {
std::vector<char> ReadAll(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw LoomError("cannot open " + path);
  return std::vector<char>(std::istreambuf_iterator<char>(in), {});
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 7) {
    std::fprintf(stderr, "usage: hal_run <model.gguf> <hal> <gx> <wg_size> <min_sizes> <binding>...\n");
    return 2;
  }
  try {
    auto gguf = yah::core::Gguf::OpenResident(argv[1]);
    const std::string hal = argv[2];
    const uint32_t gx = static_cast<uint32_t>(std::atoi(argv[3]));  // "gx" or "gx,gy"
    const char* comma = std::strchr(argv[3], ',');
    const uint32_t gy = comma ? static_cast<uint32_t>(std::atoi(comma + 1)) : 1;
    const uint32_t wg = static_cast<uint32_t>(std::atoi(argv[4]));
    std::vector<size_t> mins;
    {
      std::stringstream ss(argv[5]);
      std::string item;
      while (std::getline(ss, item, ',')) mins.push_back(std::strtoull(item.c_str(), nullptr, 10));
    }
    const int nb = argc - 6;
    if (static_cast<int>(mins.size()) != nb) {
      std::fprintf(stderr, "hal_run: %zu min sizes for %d bindings\n", mins.size(), nb);
      return 3;
    }
    LoomDevice gpu;
    LoomBuffer weights;
    size_t delta = 0;
    {
      const std::uint8_t* wbase = gguf.tensor_data_base();
      const std::uintptr_t start = reinterpret_cast<std::uintptr_t>(wbase) & ~std::uintptr_t{4095};
      delta = reinterpret_cast<std::uintptr_t>(wbase) - start;
      weights = gpu.Import(reinterpret_cast<void*>(start), gguf.tensor_data_size() + delta);
    }
    std::vector<LoomBuffer> owned;
    owned.reserve(nb);
    std::vector<hrx_buffer_ref_t> refs;
    std::vector<std::pair<size_t, std::string>> outs;  // (ref index, file)
    for (int i = 0; i < nb; ++i) {
      const std::string spec = argv[6 + i];
      size_t size = 0;
      if (spec.rfind("t:", 0) == 0) {
        const auto* t = gguf.Find(spec.substr(2));
        if (!t) throw LoomError("tensor not found: " + spec.substr(2));
        size = static_cast<size_t>(t->bytes);
        refs.push_back({weights.handle, delta + static_cast<size_t>(t->offset), size});
      } else if (spec.rfind("f:", 0) == 0 || spec.rfind("io:", 0) == 0) {
        const bool io = spec[0] == 'i';
        std::string in = spec.substr(io ? 3 : 2), out;
        if (io) {
          const size_t c = in.find(':');
          out = in.substr(c + 1);
          in = in.substr(0, c);
        }
        const auto data = ReadAll(in);
        size = data.size();
        owned.push_back(gpu.Allocate(size));
        gpu.H2D(owned.back(), data.data(), size);
        refs.push_back({owned.back().handle, 0, size});
        if (io) outs.emplace_back(refs.size() - 1, out);
      } else if (spec.rfind("z:", 0) == 0) {  // scratch: allocated, not initialized or read back
        size = std::strtoull(spec.substr(2).c_str(), nullptr, 10);
        owned.push_back(gpu.Allocate(size));
        refs.push_back({owned.back().handle, 0, size});
      } else if (spec.rfind("o:", 0) == 0) {
        const size_t c = spec.find(':', 2);
        size = std::strtoull(spec.substr(2, c - 2).c_str(), nullptr, 10);
        owned.push_back(gpu.Allocate(size));
        std::vector<char> z(size, 0);
        gpu.H2D(owned.back(), z.data(), size);
        refs.push_back({owned.back().handle, 0, size});
        outs.emplace_back(refs.size() - 1, spec.substr(c + 1));
      } else {
        throw LoomError("bad binding " + spec);
      }
      if (size < mins[i]) {
        std::fprintf(stderr, "hal_run: binding %d (%s) is %zu bytes, kernel footprint %zu: refusing\n", i, spec.c_str(),
                     size, mins[i]);
        return 3;
      }
    }
    LoomExecutable exe = gpu.Load(hal);
    const auto cfg = LoomDevice::Config(gx, gy, 1, wg, 1, 1);
    const auto t_first = std::chrono::steady_clock::now();
    gpu.Dispatch(exe, 0, cfg, nullptr, 0, refs.data(), refs.size());
    gpu.Synchronize();
    if (std::getenv("HAL_RUN_TIME1"))  // the first dispatch alone, for stateful kernels
      std::printf("hal_run: first dispatch %.4f ms\n",
                  std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_first).count());
    // HAL_RUN_ITERS=N: time N more back-to-back dispatches.
    // The outputs then equal the first dispatch's only if the kernel is idempotent (resid is not).
    if (const char* it = std::getenv("HAL_RUN_ITERS")) {
      const int iters = std::atoi(it);
      if (iters <= 0) throw LoomError("HAL_RUN_ITERS must be positive");
      // Host wall clock around the loop and the final wait: HRX event pairs around plain dispatches read near zero.
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < iters; ++i) gpu.Dispatch(exe, 0, cfg, nullptr, 0, refs.data(), refs.size());
      gpu.Synchronize();
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      std::printf("hal_run: %.4f ms per dispatch over %d\n", ms / iters, iters);
    }
    for (const auto& [idx, file] : outs) {
      std::vector<char> host(refs[idx].length);
      // Outputs are always owned buffers: find the one that backs this ref.
      for (auto& b : owned) {
        if (b.handle == refs[idx].buffer) {
          gpu.D2H(b, host.data(), host.size());
          break;
        }
      }
      std::ofstream(file, std::ios::binary).write(host.data(), static_cast<std::streamsize>(host.size()));
    }
    std::printf("hal_run: ok\n");
  } catch (const std::exception& error) {
    std::fprintf(stderr, "hal_run: %s\n", error.what());
    return 1;
  }
  return 0;
}
