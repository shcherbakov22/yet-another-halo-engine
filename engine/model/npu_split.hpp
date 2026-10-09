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

// Sizes of the GPU buffers shared with the NPU and the gate protocol of the set's NPU image (LoomPrefill::npu_plan).
struct NpuPlan {
  std::size_t a_bytes = 0, w_bytes = 0, c_bytes = 0;  // encoded activations, encoded weight panels, f32 C panels
  // dispatch.txt "npugate <supply> <calls> <record>" (gen_npu_gemm GATE_*): a job's gate value is sequence * gate_calls
  // + its calls; done lands gate_record bytes into a call's signal binding.
  std::uint32_t gate_calls = 0, gate_record = 0;
  // dispatch.txt "npudcol": the NPU decodes the weights itself from raw GGUF rows (gen_npu_gemm dcol 2; w_bytes is
  // then one call's unread panel binding).
  bool dcol = false;
};

// A byte range of A, W or C (in_c: an activation view of C, e.g. the FFN block's H, read by its down calls).
struct NpuView {
  std::size_t offset = 0, length = 0;
  bool in_c = false;
};

class NpuSplit {
 public:
  virtual ~NpuSplit() = default;
  [[nodiscard]] virtual const LoomBuffer& A() const = 0;
  [[nodiscard]] virtual const LoomBuffer& W() const = 0;
  [[nodiscard]] virtual const LoomBuffer& C() const = 0;
  // One call of the NPU GEMM image at path on these views (cold: loads and binds; the same arguments return the same id).
  virtual std::uint32_t Bind(const std::string& image, NpuView a, NpuView w, NpuView c) = 0;
  // Decoder-column sets (NpuPlan::dcol): the NPU reads raw weight rows from the model's own mapping.
  // RegisterRaw makes the pages holding [p, p + bytes) readable by the NPU (once per range, before any work).
  // BindRaw is one call of a decoder-column image whose raw rows are [raw, raw + raw_bytes) of a registered range;
  // swap is the image that re-programs only the decoder tiles for its format, dec that format's id. family names the
  // images that differ only in their decoder tiles (swaps switch between them); a job's images are one family, and
  // where a job's family differs from the array's, the command first runs the image's setup ("<image>.setup.xdna").
  virtual void RegisterRaw(const void* p, std::size_t bytes) = 0;
  virtual std::uint32_t BindRaw(const std::string& image, const std::string& swap, int dec, NpuView a,
                                const void* raw, std::size_t raw_bytes, NpuView c,
                                const std::string& family = "dcol") = 0;
  // A column-pair image (gate rows of one format, up rows of another): two raw ranges, and junk_bytes for the gate
  // columns' C records, which nothing reads.
  virtual std::uint32_t BindRawPair(const std::string& image, const std::string& swap, int dec, NpuView a,
                                    const void* gate, std::size_t gate_bytes, const void* up, std::size_t up_bytes,
                                    NpuView c, std::size_t junk_bytes, const std::string& family) = 0;
  // Handoffs are 32-bit words in host memory (Flags(): their GPU view), with no host between the GPU and the NPU.
  // The graph stores a job's gate value to its ready word once the job's inputs are written (release, system scope).
  // The NPU image waits for it (gen_npu_gemm GATE), runs the calls and writes the value to the job's done word.
  // The graph's yah_npu_flag_wait polls done.
  // Failure: done with kGateFailed (the NPU gave up waiting, or the host saw its command fail) sets word kFlagStatus.
  // So does a GPU wait that timed out. It is sticky: every later wait returns at once, and the host reports it.
  // Bounds: the NPU gives up after gen_npu_gemm.GATE_SUPPLY polls (~400 ms, below the driver's 2000 ms command limit,
  // amdxdna tdr_timeout_ms); a GPU wait after gen_npu_unpack.FLAG_WAIT_POLLS (~2 s, far beyond any NPU job).
  static constexpr std::uint32_t kFlagStatus = 1, kGateFailed = 0x80000000u;
  [[nodiscard]] virtual const LoomBuffer* Flags() const = 0;
  [[nodiscard]] virtual std::uint32_t Word(std::uint32_t word) const = 0;
  // A job's flag words (its first call's ready, its last call's done) and gate value.
  struct Job {
    std::uint32_t ready = 0, done = 0, gate = 0;
  };
  // The next job, of these calls (jobs count from 1 in queue order; the NPU counts them too).
  virtual Job NewJob(const std::vector<std::uint32_t>& calls) = 0;
  // Before a chunk's first NewJob: decoder-column sets restart the job count per chunk (and per image family run).
  virtual void BeginChunk() {}
  // A job to queue: its calls, a tag naming it in the stats, its words.
  struct Queued {
    std::vector<std::uint32_t> calls;
    std::string tag;
    Job words;
  };
  // Queues the jobs at once, in NewJob order, several per NPU command: the NPU waits for each ready word itself.
  // If it throws, it has stored gate | kGateFailed to the done words of the jobs it did not queue (a launched graph
  // waits for them).
  virtual void Enqueue(const std::vector<Queued>& jobs) = 0;
  // A flag word from / to the host (CPU-driven tests: the host stands in for the GPU's stores and waits).
  virtual void HostStore(std::uint32_t word, std::uint32_t value) = 0;
  virtual std::uint32_t HostLoad(std::uint32_t word) = 0;
};

}  // namespace yah::model

#endif  // YAH_MODEL_NPU_SPLIT_HPP_
