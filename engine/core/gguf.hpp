#ifndef YAH_CORE_GGUF_HPP_
#define YAH_CORE_GGUF_HPP_

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace yah::core {

// GGML tensor type ids, exactly as stored in a GGUF tensor table.
enum class GgmlType : std::uint32_t {
  kF32 = 0,
  kF16 = 1,
  kQ4_0 = 2,
  kQ4_1 = 3,
  kQ5_0 = 6,
  kQ5_1 = 7,
  kQ8_0 = 8,
  kQ8_1 = 9,
  kQ2_K = 10,
  kQ3_K = 11,
  kQ4_K = 12,
  kQ5_K = 13,
  kQ6_K = 14,
  kQ8_K = 15,
  kIQ2_XXS = 16,
  kIQ2_XS = 17,
  kIQ3_XXS = 18,
  kIQ1_S = 19,
  kIQ4_NL = 20,
  kIQ3_S = 21,
  kIQ2_S = 22,
  kIQ4_XS = 23,
  kI8 = 24,
  kI16 = 25,
  kI32 = 26,
  kI64 = 27,
  kF64 = 28,
  kIQ1_M = 29,
  kBF16 = 30,
};

[[nodiscard]] const char* TypeName(GgmlType type);
[[nodiscard]] std::uint32_t BlockElements(GgmlType type);
[[nodiscard]] std::uint32_t BlockBytes(GgmlType type);

struct TensorInfo {
  std::string name;
  std::vector<std::uint64_t> dims;  // GGUF order: [ne0, ne1, ...]
  GgmlType type{GgmlType::kF32};
  std::uint64_t offset{0};  // from the start of the tensor data region
  std::uint64_t elements{0};
  std::uint64_t bytes{0};
};

// One GGUF metadata value. Arrays carry their element kind in `kind` too.
struct MetadataValue {
  enum class Kind { kUInt, kInt, kFloat, kBool, kString, kArray } kind{Kind::kUInt};
  std::uint64_t u{0};
  std::int64_t i{0};
  double f{0.0};
  bool b{false};
  std::string s;
  std::vector<MetadataValue> array;
};

// A read-only, mmapped GGUF file. Weight bytes are not copied: Data() points into the mapping.
class Gguf {
 public:
  static Gguf Open(const std::string& path);
  // The file read into locked anonymous memory (2 MB pages where the kernel has them) instead of mapped: for a GPU
  // that imports the bytes as user pages. Reclaim scans mapped file pages (their large folios escape mlock) and every
  // scan of an imported page evicts all of the process's GPU queues for 0.03-3 s. The memory is writable, so a
  // device that pins it for writing shares these pages (no copy-on-write copies). Needs RLIMIT_MEMLOCK (a warning
  // when the kernel refuses the lock).
  static Gguf OpenResident(const std::string& path);
  Gguf(Gguf&& other) noexcept;
  Gguf& operator=(Gguf&& other) noexcept;
  Gguf(const Gguf&) = delete;
  Gguf& operator=(const Gguf&) = delete;
  ~Gguf();

  [[nodiscard]] std::uint32_t version() const { return version_; }
  [[nodiscard]] const std::vector<TensorInfo>& tensors() const { return tensors_; }
  [[nodiscard]] const TensorInfo* Find(const std::string& name) const;
  [[nodiscard]] const MetadataValue* Meta(const std::string& key) const;
  [[nodiscard]] std::size_t file_size() const { return size_; }
  [[nodiscard]] std::size_t tensor_data_offset() const { return data_offset_; }
  // The contiguous tensor-data region that the whole tensor table addresses.
  // A device backend copies exactly this region and rebases every TensorRef by the pointer delta.
  [[nodiscard]] const std::uint8_t* tensor_data_base() const { return base_ + data_offset_; }
  [[nodiscard]] std::size_t tensor_data_size() const { return size_ - data_offset_; }
  [[nodiscard]] std::size_t metadata_count() const { return meta_.size(); }
  [[nodiscard]] const std::uint8_t* Data(const TensorInfo& tensor) const {
    return base_ + data_offset_ + tensor.offset;
  }

 private:
  Gguf() = default;
  void Close();

  std::uint8_t* base_{nullptr};
  std::size_t size_{0};
  std::size_t map_bytes_{0};   // the mapping's length (OpenResident rounds it up to 2 MB)
  int fd_{-1};
  std::uint32_t version_{0};
  std::uint64_t data_offset_{0};
  std::vector<TensorInfo> tensors_;
  std::map<std::string, MetadataValue> meta_;
};

}  // namespace yah::core

#endif  // YAH_CORE_GGUF_HPP_
