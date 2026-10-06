// One kstore GEMM split between the GPU and the NPU, end to end on one stream (tools/npu_split_check.py drives it).
//
// usage: npu_split_run <model.gguf> <plan.txt> <act.f16> <out.f32>
//
// The plan (key=value lines, written by npu_split_check.py) names the tensor, the shapes and the kernels.
// The GPU encodes the input and the NPU's rows to BFP16, then computes its leading rows while the relay runs the NPU calls.
// The stream then waits for the NPU and unpacks C into the trailing columns.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "core/gguf.hpp"
#include "model/loom_npu.hpp"
#include "model/loom_runtime.hpp"

using yah::model::LoomBuffer;
using yah::model::LoomDevice;
using yah::model::LoomError;
using yah::model::LoomExecutable;
using yah::model::LoomNpu;

namespace {

std::vector<char> ReadAll(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw LoomError("cannot open " + path);
  std::vector<char> data(static_cast<size_t>(f.tellg()));
  f.seekg(0);
  f.read(data.data(), static_cast<std::streamsize>(data.size()));
  return data;
}

std::map<std::string, std::string> ReadPlan(const std::string& path) {
  std::map<std::string, std::string> plan;
  std::ifstream f(path);
  std::string line;
  while (std::getline(f, line)) {
    const size_t eq = line.find('=');
    if (eq != std::string::npos) plan[line.substr(0, eq)] = line.substr(eq + 1);
  }
  return plan;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) {
    std::fprintf(stderr, "usage: npu_split_run <model.gguf> <plan.txt> <act.f16> <out.f32>\n");
    return 2;
  }
  try {
    auto gguf = yah::core::Gguf::Open(argv[1]);
    auto plan = ReadPlan(argv[2]);
    auto num = [&](const char* key) {
      if (!plan.count(key)) throw LoomError(std::string("plan: missing ") + key);
      return std::strtoull(plan[key].c_str(), nullptr, 10);
    };
    const size_t tokens = num("tokens"), n = num("n"), k = num("k"), npu_rows = num("npu_rows"), panels = num("panels");
    const auto* t = gguf.Find(plan["tensor"]);
    if (!t) throw LoomError("tensor not found: " + plan["tensor"]);
    const size_t row_bytes = static_cast<size_t>(t->bytes) / n;

    LoomDevice gpu;
    LoomBuffer weights;
    size_t delta = 0;
    {
      const std::uint8_t* wbase = gguf.tensor_data_base();
      const std::uintptr_t start = reinterpret_cast<std::uintptr_t>(wbase) & ~std::uintptr_t{4095};
      delta = reinterpret_cast<std::uintptr_t>(wbase) - start;
      weights = gpu.Import(reinterpret_cast<void*>(start), gguf.tensor_data_size() + delta);
    }
    const size_t w_off = delta + static_cast<size_t>(t->offset);
    std::vector<LoomBuffer> tables;
    if (!plan["tables"].empty()) {
      std::stringstream ss(plan["tables"]);
      std::string path;
      while (std::getline(ss, path, ',')) {
        const auto data = ReadAll(path);
        tables.push_back(gpu.Allocate(data.size()));
        gpu.H2D(tables.back(), data.data(), data.size());
      }
    }
    const auto act = ReadAll(argv[3]);
    if (act.size() != tokens * k * 2) throw LoomError("act.f16 size");
    LoomBuffer x = gpu.Allocate(act.size());
    gpu.H2D(x, act.data(), act.size());
    LoomBuffer out = gpu.Allocate(tokens * n * 4);
    LoomBuffer dq = gpu.Allocate(npu_rows * k * 2);
    LoomBuffer wstage = gpu.Allocate(17408 * 16 * 2), ostage = gpu.Allocate(17408 * tokens * 4);

    LoomNpu npu(gpu, static_cast<std::uint32_t>(num("npu_columns")));
    const size_t a_bytes = num("a_bytes"), w_panel = num("w_panel_bytes"), c_panel = num("c_panel_bytes");
    auto& sa = npu.CreateShared(a_bytes);
    auto& sw = npu.CreateShared(panels * w_panel);
    auto& sc = npu.CreateShared(panels * c_panel);
    std::vector<LoomNpu::Kernel*> kernels;
    for (size_t p = 0; p < panels; ++p)
      kernels.push_back(&npu.Load(plan["npu_xdna"], plan["npu_entry"],
                                  {{&sa, 0, a_bytes}, {&sw, p * w_panel, w_panel}, {&sc, p * c_panel, c_panel}}));

    LoomExecutable enc_act = gpu.Load(plan["enc_act_hal"]), dqx = gpu.Load(plan["dq_hal"]),
                   enc_wgt = gpu.Load(plan["enc_wgt_hal"]), split = gpu.Load(plan["split_hal"]),
                   unpack = gpu.Load(plan["unpack_hal"]);
    auto cfg1 = [](size_t gx, size_t wg) { return LoomDevice::Config(gx, 1, 1, wg, 1, 1); };
    const int iters = static_cast<int>(num("iters"));
    // The last iteration runs under device dispatch timestamps (100 MHz ticks): one line per dispatch.
    std::vector<hrx_profile_dispatch_t> events;
    for (int it = 0; it < iters; ++it) {
      if (it == iters - 1)
        gpu.ProfileBegin([&](const hrx_profile_dispatch_t* e, size_t n, std::uint64_t) { events.insert(events.end(), e, e + n); });
      // Before the last iteration C starts as NaN: stale GPU cache lines of it would show in the checked output.
      if (it < iters - 1) gpu.Fill(sc.gpu, 0x7fc00000u);
      const auto t0 = std::chrono::steady_clock::now();
      {
        hrx_buffer_ref_t b[] = {{x.handle, 0, x.size}, {sa.gpu.handle, 0, a_bytes}};
        gpu.Dispatch(enc_act, 0, cfg1(num("enc_act_wgs"), 256), nullptr, 0, b, 2);
      }
      {
        std::vector<hrx_buffer_ref_t> b = {{weights.handle, w_off + (n - npu_rows) * row_bytes, npu_rows * row_bytes}};
        for (auto& tb : tables) b.push_back({tb.handle, 0, tb.size});
        b.push_back({dq.handle, 0, dq.size});
        gpu.Dispatch(dqx, 0, cfg1(num("dq_wgs"), num("dq_wg")), nullptr, 0, b.data(), b.size());
      }
      {
        hrx_buffer_ref_t b[] = {{dq.handle, 0, dq.size}, {sw.gpu.handle, 0, panels * w_panel}};
        gpu.Dispatch(enc_wgt, 0, cfg1(num("enc_wgt_wgs"), 256), nullptr, 0, b, 2);
      }
      const std::uint64_t ticket = npu.Enqueue(kernels);
      {
        std::vector<hrx_buffer_ref_t> b = {{weights.handle, w_off, static_cast<size_t>(t->bytes)}};
        for (auto& tb : tables) b.push_back({tb.handle, 0, tb.size});
        for (const LoomBuffer* bb : {&x, &wstage, &ostage, &out}) b.push_back({bb->handle, 0, bb->size});
        gpu.Dispatch(split, 0, LoomDevice::Config(num("split_gx"), num("split_gy"), 1, num("split_wg"), 1, 1),
                     nullptr, 0, b.data(), b.size());
      }
      npu.StreamWait(ticket);
      {
        hrx_buffer_ref_t b[] = {{sc.gpu.handle, 0, panels * c_panel}, {out.handle, 0, out.size}};
        gpu.Dispatch(unpack, 0, cfg1(num("unpack_wgs"), 256), nullptr, 0, b, 2);
      }
      gpu.Synchronize();
      npu.CheckHealth();
      std::printf("npu_split_run: iteration %d %.3f ms (NPU %.3f ms)\n", it,
                  std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(),
                  npu.LastJobMs());
    }
    gpu.ProfileEnd();
    static const char* const kStage[] = {"enc_act", "dq", "enc_wgt", "split", "unpack"};
    for (size_t i = 0; i < events.size(); ++i)
      std::printf("npu_split_run: %-8s start %8.1f us  %8.1f us\n", i < 5 ? kStage[i] : "?",
                  (events[i].start_tick - events[0].start_tick) / 100.0,
                  (events[i].end_tick - events[i].start_tick) / 100.0);
    std::vector<char> host(out.size);
    gpu.D2H(out, host.data(), host.size());
    std::ofstream(argv[4], std::ios::binary).write(host.data(), static_cast<std::streamsize>(host.size()));
    std::printf("npu_split_run: ok\n");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "npu_split_run: %s\n", e.what());
    return 1;
  }
  return 0;
}
