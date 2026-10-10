// One kstore GEMM split between the GPU and the NPU, end to end (tools/npu_split_check.py drives it).
//
// usage: npu_split_run <model.gguf> <plan.txt> <act.f16> <out.f32>
//
// The plan (key=value lines, written by npu_split_check.py) names the tensor, the shapes and the kernels.
// One graph per split, like a prefill NPU job (LoomPrefill NpuEnqueue / NpuJoin): the GPU encodes the input to BFP16,
// decodes the NPU's rows straight to BFP16, stores the job's ready word, computes its leading rows while the gated NPU
// image runs the calls, waits for the job's done word (yah_npu_flag_wait) and unpacks C into the trailing columns.
#include <algorithm>
#include <array>
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
using yah::model::LoomGraph;
using yah::model::LoomNpuSplit;
using yah::model::NpuPlan;
using yah::model::NpuSplit;

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

// The GPU's current shader clock in MHz from sysfs (0 if unknown): GPU times only compare at a known clock.
int GpuMhz() {
  for (int card = 0; card < 4; ++card) {
    std::ifstream f("/sys/class/drm/card" + std::to_string(card) + "/device/pp_dpm_sclk");
    std::string line;
    while (std::getline(f, line))
      if (line.find('*') != std::string::npos) return std::atoi(line.substr(line.find(':') + 1).c_str());
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) {
    std::fprintf(stderr, "usage: npu_split_run <model.gguf> <plan.txt> <act.f16> <out.f32>\n");
    return 2;
  }
  try {
    auto gguf = yah::core::Gguf::OpenResident(argv[1]);
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
    LoomBuffer wstage = gpu.Allocate(17408 * 16 * 2), ostage = gpu.Allocate(17408 * tokens * 4);

    const size_t a_bytes = num("a_bytes"), w_panel = num("w_panel_bytes"), c_panel = num("c_panel_bytes");
    NpuPlan np;
    np.a_bytes = a_bytes, np.w_bytes = panels * w_panel, np.c_bytes = panels * c_panel;
    np.gate_calls = static_cast<std::uint32_t>(num("gate_calls"));
    np.gate_record = static_cast<std::uint32_t>(num("gate_record"));
    LoomNpuSplit npu(gpu, np);
    std::vector<std::uint32_t> calls;
    for (size_t p = 0; p < panels; ++p)
      calls.push_back(npu.Bind(plan["npu_xdna"], {0, a_bytes}, {p * w_panel, w_panel}, {p * c_panel, c_panel}));

    LoomExecutable enc_act = gpu.Load(plan["enc_act_hal"]), dqx = gpu.Load(plan["dq_hal"]),
                   split = gpu.Load(plan["split_hal"]), unpack = gpu.Load(plan["unpack_hal"]),
                   flag_wait = gpu.Load(plan["flag_wait_hal"]);
    const hrx_buffer_ref_t sa{npu.A().handle, 0, a_bytes}, sw{npu.W().handle, 0, panels * w_panel},
        sc{npu.C().handle, 0, panels * c_panel};
    const auto flag = [&](std::uint32_t word) {
      return hrx_buffer_ref_t{npu.Flags()->handle, std::size_t{word} * 4, 4};
    };
    auto cfg1 = [](size_t gx, size_t wg) { return LoomDevice::Config(gx, 1, 1, wg, 1, 1); };
    const int iters = static_cast<int>(num("iters"));
    // The GPU clock ramps up over seconds of load: run the GPU share alone for warmup_ms first.
    {
      std::vector<hrx_buffer_ref_t> b = {{weights.handle, w_off, static_cast<size_t>(t->bytes)}};
      for (auto& tb : tables) b.push_back({tb.handle, 0, tb.size});
      for (const LoomBuffer* bb : {&x, &wstage, &ostage, &out}) b.push_back({bb->handle, 0, bb->size});
      const auto w0 = std::chrono::steady_clock::now();
      while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count() < num("warmup_ms")) {
        gpu.Dispatch(split, 0, LoomDevice::Config(num("split_gx"), num("split_gy"), 1, num("split_wg"), 1, 1), nullptr, 0,
                     b.data(), b.size());
        gpu.Synchronize();
      }
    }
    const auto split_bindings = [&] {
      std::vector<hrx_buffer_ref_t> b = {{weights.handle, w_off, static_cast<size_t>(t->bytes)}};
      for (auto& tb : tables) b.push_back({tb.handle, 0, tb.size});
      for (const LoomBuffer* bb : {&x, &wstage, &ostage, &out}) b.push_back({bb->handle, 0, bb->size});
      return b;
    };
    // One split GEMM as one graph: encode A, decode the NPU's rows to BFP16, store the ready word, the GPU's rows
    // beside the NPU, wait for the done word, unpack C.
    // stage_grid[i]: stage i's workgroup count (kStage order), which identifies its timestamps (their command index
    // counts HRX's barriers and stores too).
    std::array<std::uint32_t, 3> stage_grid[4] = {};
    const auto run_split = [&] {
      LoomGraph g(gpu);
      g.ReadOnly(weights.handle);
      for (auto& tb : tables) g.ReadOnly(tb.handle);
      {
        const hrx_buffer_ref_t b[] = {{x.handle, 0, x.size}, sa};
        stage_grid[0] = g.Grid(g.Dispatch(enc_act, 0, cfg1(num("enc_act_wgs"), 256), b, 2, 2));
      }
      {
        std::vector<hrx_buffer_ref_t> b = {{weights.handle, w_off + (n - npu_rows) * row_bytes, npu_rows * row_bytes}};
        for (auto& tb : tables) b.push_back({tb.handle, 0, tb.size});
        b.push_back(sw);
        stage_grid[1] = g.Grid(g.Dispatch(dqx, 0, cfg1(num("dq_wgs"), num("dq_wg")), b.data(), b.size(),
                                   std::uint64_t{1} << (b.size() - 1)));
      }
      const NpuSplit::Job job = npu.NewJob(calls);
      g.AtomicStore(flag(job.ready), job.gate, HRX_ATOMIC_FLAG_RELEASE | HRX_ATOMIC_FLAG_SYSTEM_SCOPE, {sa, sw}, {sc});
      {
        const auto b = split_bindings();
        const auto cfg = LoomDevice::Config(num("split_gx"), num("split_gy"), 1, num("split_wg"), 1, 1);
        const std::uint64_t writes = std::uint64_t{3} << (b.size() - 2);   // ostage, out
        stage_grid[2] = g.Grid(g.Dispatch(split, 0, cfg, b.data(), b.size(), writes));
      }
      {
        const hrx_buffer_ref_t b[] = {flag(job.done), flag(job.ready), flag(NpuSplit::kFlagStatus)};
        const std::vector<hrx_buffer_ref_t> npu_side{sa, sw, sc};
        g.Dispatch(flag_wait, 0, cfg1(1, 32), b, 3, 4, nullptr, &npu_side);
      }
      {
        const hrx_buffer_ref_t b[] = {sc, {out.handle, 0, out.size}};
        stage_grid[3] = g.Grid(g.Dispatch(unpack, 0, cfg1(num("unpack_wgs"), num("unpack_wg")), b, 2, 2));
      }
      for (int i = 0; i < 4; ++i)
        for (int j = 0; j < i; ++j)
          if (stage_grid[i] == stage_grid[j]) throw LoomError("profile: two stages share a grid");
      g.Launch();
      npu.Enqueue({{calls, "split", job}});
      gpu.Synchronize();
      npu.CheckHealthNow();
      if (npu.Word(NpuSplit::kFlagStatus)) throw LoomError("the GPU's wait for the NPU failed");
    };
    // Checked output: C starts as NaN before every iteration but the last, so stale GPU cache lines of it would show.
    for (int it = 0; it < 2; ++it) {
      if (it == 0) gpu.Fill(npu.C(), 0x7fc00000u);
      run_split();
    }
    std::vector<char> host(out.size);
    gpu.D2H(out, host.data(), host.size());
    // Timing: rounds of one split GEMM then the whole GEMM on the GPU, interleaved so clock drift hits both alike.
    // Device timestamps are 100 MHz ticks.
    LoomExecutable full = gpu.Load(plan["full_hal"]);
    const int rounds = static_cast<int>(num("iters"));
    std::vector<hrx_profile_dispatch_t> events;
    std::vector<double> npu_ms;
    gpu.ProfileBegin([&](const hrx_profile_dispatch_t* e, size_t n, std::uint64_t) { events.insert(events.end(), e, e + n); });
    const auto full_bindings = split_bindings();
    for (int r = 0; r < rounds; ++r) {
      run_split();
      npu_ms.push_back(npu.LastCommandMs());
      gpu.Dispatch(full, 0, LoomDevice::Config(num("full_gx"), num("split_gy"), 1, num("split_wg"), 1, 1), nullptr, 0,
                   full_bindings.data(), full_bindings.size());
      gpu.Synchronize();
    }
    gpu.ProfileEnd();
    // Per round: the split graph's five dispatches (stages by node index), then the whole GEMM (it starts last).
    if (events.size() != static_cast<size_t>(rounds) * 6) throw LoomError("profile: dispatch count");
    static const char* const kStage[] = {"enc_act", "dq_bfp", "split", "unpack", "full"};
    const auto median = [](std::vector<double> v) {
      std::sort(v.begin(), v.end());
      return v[v.size() / 2];
    };
    std::vector<double> stage[5], span;
    for (int r = 0; r < rounds; ++r) {
      std::vector<hrx_profile_dispatch_t> e(events.begin() + r * 6, events.begin() + (r + 1) * 6);
      std::sort(e.begin(), e.end(), [](const auto& a, const auto& b) { return a.start_tick < b.start_tick; });
      stage[4].push_back((e[5].end_tick - e[5].start_tick) / 100.0);
      std::uint64_t first = UINT64_MAX, last = 0;
      for (int i = 0; i < 5; ++i) {
        first = std::min(first, e[i].start_tick), last = std::max(last, e[i].end_tick);
        for (int s = 0; s < 4; ++s)
          if (std::equal(e[i].workgroup_count, e[i].workgroup_count + 3, stage_grid[s].begin()))
            stage[s].push_back((e[i].end_tick - e[i].start_tick) / 100.0);
      }
      span.push_back((last - first) / 100.0);
    }
    for (int i = 0; i < 5; ++i) {
      if (stage[i].size() != static_cast<size_t>(rounds)) throw LoomError(std::string("profile: no ") + kStage[i]);
      std::printf("npu_split_run: %-8s %8.1f us (median of %d)\n", kStage[i], median(stage[i]), rounds);
    }
    std::printf("npu_split_run: split GEMM %.1f us vs whole GEMM on the GPU %.1f us; NPU command %.1f us; GPU %d MHz\n",
                median(span), median(stage[4]), 1000 * median(npu_ms), GpuMhz());
    std::ofstream(argv[4], std::ios::binary).write(host.data(), static_cast<std::streamsize>(host.size()));
    std::printf("npu_split_run: ok\n");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "npu_split_run: %s\n", e.what());
    return 1;
  }
  return 0;
}
