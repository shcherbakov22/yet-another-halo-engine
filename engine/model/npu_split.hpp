// NpuSplit: what LoomPrefill needs from the NPU for the column-split GEMMs (dispatch.txt "npusplit_<site>").
// The implementation (LoomNpuSplit in model/loom_npu.hpp) links libamdf and the HRX xdna loader; this header does not.
#ifndef YAH_MODEL_NPU_SPLIT_HPP_
#define YAH_MODEL_NPU_SPLIT_HPP_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "model/loom_runtime.hpp"

namespace yah::model {

// Sizes of the GPU buffers shared with the NPU (LoomPrefill::npu_plan).
struct NpuPlan {
  std::size_t a_bytes = 0, w_bytes = 0, c_bytes = 0;  // encoded activations, encoded weight panels, f32 C panels
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
  // The calls in order behind the stream's current position; returns the value to Join on.
  virtual std::uint64_t Enqueue(const std::vector<std::uint32_t>& calls) = 0;
  // Orders the stream after that work (the host waits for it first).
  virtual void Join(std::uint64_t value) = 0;
};

}  // namespace yah::model

#endif  // YAH_MODEL_NPU_SPLIT_HPP_
