// NpuSplit: what LoomPrefill needs from the NPU for the column-split GEMMs (dispatch.txt "npusplit_<site>").
// The implementation (LoomNpuSplit in model/loom_npu.hpp) links libamdf and the HRX xdna loader; this header does not.
#ifndef YAH_MODEL_NPU_SPLIT_HPP_
#define YAH_MODEL_NPU_SPLIT_HPP_

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "model/loom_runtime.hpp"

namespace yah::model {

// Sizes of the GPU buffers shared with the NPU (LoomPrefill::npu_plan).
struct NpuPlan {
  std::size_t a_bytes = 0, w_bytes = 0, c_bytes = 0;  // encoded activations, encoded weight panels, f32 C panels
  std::uint32_t gate = 0;  // polls per job of a gated image (dispatch.txt "npugate"; 0: the relay waits)
};

// A byte range of A, W or C.
struct NpuView {
  std::size_t offset = 0, length = 0;
};

class NpuSplit {
 public:
  virtual ~NpuSplit() = default;
  [[nodiscard]] virtual const LoomBuffer& A() const = 0;
  [[nodiscard]] virtual const LoomBuffer& W() const = 0;
  [[nodiscard]] virtual const LoomBuffer& C() const = 0;
  // One call of the NPU GEMM image at path on these views (cold: loads and binds; the same arguments return the same id).
  virtual std::uint32_t Bind(const std::string& image, NpuView a, NpuView w, NpuView c) = 0;
  // The prefill graph and the relay hand jobs over through 32-bit words in host memory (one graph per chunk).
  // Flags(): the GPU view of the words. Word kFlagStatus is set by a GPU wait that timed out.
  static constexpr std::uint32_t kFlagStatus = 1, kFlagFirstJob = 2;
  [[nodiscard]] virtual const LoomBuffer* Flags() const = 0;
  [[nodiscard]] virtual std::uint32_t FlagWords() const = 0;
  [[nodiscard]] virtual std::uint32_t Word(std::uint32_t word) const = 0;
  // The calls in order once word ready holds epoch (the graph stores it); then the relay stores epoch to word done,
  // also when the NPU failed, so no GPU wait is left spinning. tag names the job in the stats.
  virtual void EnqueueFlagged(const std::vector<std::uint32_t>& calls, const std::string& tag, std::uint32_t ready,
                              std::uint32_t done, std::uint32_t epoch) = 0;
  // A gated image (NpuPlan::gate) waits for its jobs itself: no relay between the GPU and the NPU. A job's ready word
  // is its first call's, its done word its last call's (GateWords: {ready, done}); the graph stores the job's gate value
  // (NextGate: sequence * 64 + calls, jobs counting from 1 in queue order) to ready, the NPU writes it to done (bit 31:
  // the NPU gave up). EnqueueFlagged then takes the gate value as epoch and queues the job at once.
  // Stores value to done word: ends the GPU's wait for a job that will never run (a gated value with bit 31, so the wait
  // reports it). For a chunk graph already launched when queueing its jobs failed.
  virtual void Release(std::uint32_t done, std::uint32_t value) = 0;
  [[nodiscard]] virtual bool Gated() const = 0;
  virtual std::uint32_t NextGate(const std::vector<std::uint32_t>& calls) = 0;
  [[nodiscard]] virtual std::pair<std::uint32_t, std::uint32_t> GateWords(
      const std::vector<std::uint32_t>& calls) const = 0;
};

}  // namespace yah::model

#endif  // YAH_MODEL_NPU_SPLIT_HPP_
