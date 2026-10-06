// NpuSplit: what LoomPrefill needs from the NPU for the column-split GEMMs (dispatch.txt "npusplit").
// The implementation (LoomNpuSplit in model/loom_npu.hpp) links libamdf and the HRX xdna loader; this header does not.
#ifndef YAH_MODEL_NPU_SPLIT_HPP_
#define YAH_MODEL_NPU_SPLIT_HPP_

#include <cstddef>
#include <cstdint>
#include <string>

#include "model/loom_runtime.hpp"

namespace yah::model {

// The NPU GEMM's stream sizes (dispatch.txt "npubytes") and its image.
struct NpuPlan {
  std::size_t a_bytes = 0, w_panel = 0, c_panel = 0;  // activations (whole chunk); weights and C per 512-row call
  std::uint32_t panels = 0;                            // NPU calls per layer (qkv's, then gate's)
  std::string image;                                   // the .xdna path
};

class NpuSplit {
 public:
  virtual ~NpuSplit() = default;
  // GPU buffers the NPU reads and writes: A (encoded activations), W (panels of encoded weights), C (panels of f32 C).
  virtual const LoomBuffer& A() const = 0;
  virtual const LoomBuffer& W() const = 0;
  virtual const LoomBuffer& C() const = 0;
  // Calls first .. first + count - 1 behind the stream's current position; returns the value to Join on.
  virtual std::uint64_t Enqueue(std::uint32_t first, std::uint32_t count) = 0;
  // Orders the stream after that work (the host waits for it first).
  virtual void Join(std::uint64_t value) = 0;
};

}  // namespace yah::model

#endif  // YAH_MODEL_NPU_SPLIT_HPP_
