#include "core/gguf.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace yah::core {
namespace {

constexpr std::uint32_t kGgufMagic = 0x46554747U;  // "GGUF"

struct TypeTraits {
  const char* name;
  std::uint32_t elements;
  std::uint32_t bytes;
};

// Indexed by GgmlType. Indices 4 and 5 are the removed Q4_2/Q4_3 slots.
const TypeTraits kTraits[] = {
    {"F32", 1, 4},      {"F16", 1, 2},        {"Q4_0", 32, 18},    {"Q4_1", 32, 20},     {"unused4", 1, 1},
    {"unused5", 1, 1},  {"Q5_0", 32, 22},     {"Q5_1", 32, 24},    {"Q8_0", 32, 34},     {"Q8_1", 32, 36},
    {"Q2_K", 256, 84},  {"Q3_K", 256, 110},   {"Q4_K", 256, 144},  {"Q5_K", 256, 176},   {"Q6_K", 256, 210},
    {"Q8_K", 256, 292}, {"IQ2_XXS", 256, 66}, {"IQ2_XS", 256, 74}, {"IQ3_XXS", 256, 98}, {"IQ1_S", 256, 50},
    {"IQ4_NL", 32, 18}, {"IQ3_S", 256, 110},  {"IQ2_S", 256, 82},  {"IQ4_XS", 256, 136}, {"I8", 1, 1},
    {"I16", 1, 2},      {"I32", 1, 4},        {"I64", 1, 8},       {"F64", 1, 8},        {"IQ1_M", 256, 56},
    {"BF16", 1, 2},
};
constexpr std::size_t kTypeCount = sizeof(kTraits) / sizeof(kTraits[0]);

struct Cursor {
  const std::uint8_t* p;
  const std::uint8_t* end;

  [[nodiscard]] std::size_t left() const { return static_cast<std::size_t>(end - p); }
  void Need(std::size_t n) const {
    if (left() < n) throw std::runtime_error("gguf: truncated file");
  }
  template <typename T>
  T Read() {
    Need(sizeof(T));
    T value{};
    std::memcpy(&value, p, sizeof(T));
    p += sizeof(T);
    return value;
  }
  std::string ReadString() {
    const auto n = Read<std::uint64_t>();
    Need(static_cast<std::size_t>(n));
    std::string s(reinterpret_cast<const char*>(p), static_cast<std::size_t>(n));
    p += n;
    return s;
  }
};

MetadataValue ReadValue(Cursor& c, std::uint32_t kind) {
  MetadataValue v;
  switch (kind) {
    case 0:
      v.kind = MetadataValue::Kind::kUInt;
      v.u = c.Read<std::uint8_t>();
      break;
    case 1:
      v.kind = MetadataValue::Kind::kInt;
      v.i = c.Read<std::int8_t>();
      break;
    case 2:
      v.kind = MetadataValue::Kind::kUInt;
      v.u = c.Read<std::uint16_t>();
      break;
    case 3:
      v.kind = MetadataValue::Kind::kInt;
      v.i = c.Read<std::int16_t>();
      break;
    case 4:
      v.kind = MetadataValue::Kind::kUInt;
      v.u = c.Read<std::uint32_t>();
      break;
    case 5:
      v.kind = MetadataValue::Kind::kInt;
      v.i = c.Read<std::int32_t>();
      break;
    case 6:
      v.kind = MetadataValue::Kind::kFloat;
      v.f = c.Read<float>();
      break;
    case 7:
      v.kind = MetadataValue::Kind::kBool;
      v.b = c.Read<std::uint8_t>() != 0;
      break;
    case 8:
      v.kind = MetadataValue::Kind::kString;
      v.s = c.ReadString();
      break;
    case 10:
      v.kind = MetadataValue::Kind::kUInt;
      v.u = c.Read<std::uint64_t>();
      break;
    case 11:
      v.kind = MetadataValue::Kind::kInt;
      v.i = c.Read<std::int64_t>();
      break;
    case 12:
      v.kind = MetadataValue::Kind::kFloat;
      v.f = c.Read<double>();
      break;
    case 9: {
      v.kind = MetadataValue::Kind::kArray;
      const auto element = c.Read<std::uint32_t>();
      const auto count = c.Read<std::uint64_t>();
      if (count > (1ULL << 32)) {
        throw std::runtime_error("gguf: metadata array too large");
      }
      v.array.reserve(static_cast<std::size_t>(count));
      for (std::uint64_t i = 0; i < count; ++i) {
        v.array.push_back(ReadValue(c, element));
      }
      break;
    }
    default:
      throw std::runtime_error("gguf: unknown metadata kind");
  }
  return v;
}

}  // namespace

const char* TypeName(GgmlType type) {
  return kTraits[static_cast<std::size_t>(type)].name;
}
std::uint32_t BlockElements(GgmlType type) {
  return kTraits[static_cast<std::size_t>(type)].elements;
}
std::uint32_t BlockBytes(GgmlType type) {
  return kTraits[static_cast<std::size_t>(type)].bytes;
}

Gguf::Gguf(Gguf&& other) noexcept {
  *this = std::move(other);
}

Gguf& Gguf::operator=(Gguf&& other) noexcept {
  if (this != &other) {
    Close();
    base_ = other.base_;
    size_ = other.size_;
    map_bytes_ = other.map_bytes_;
    fd_ = other.fd_;
    version_ = other.version_;
    data_offset_ = other.data_offset_;
    tensors_ = std::move(other.tensors_);
    meta_ = std::move(other.meta_);
    other.base_ = nullptr;
    other.size_ = 0;
    other.map_bytes_ = 0;
    other.fd_ = -1;
  }
  return *this;
}

Gguf::~Gguf() {
  Close();
}

void Gguf::Close() {
  if (base_ != nullptr) {
    ::munmap(base_, map_bytes_);
    base_ = nullptr;
  }
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
  size_ = 0;
}

Gguf Gguf::Open(const std::string& path) {
  Gguf g;
  g.fd_ = ::open(path.c_str(), O_RDONLY);
  if (g.fd_ < 0) throw std::runtime_error("gguf: cannot open " + path);
  struct stat st{};
  if (::fstat(g.fd_, &st) != 0) {
    g.Close();
    throw std::runtime_error("gguf: fstat failed");
  }
  g.size_ = static_cast<std::size_t>(st.st_size);
  void* base = ::mmap(nullptr, g.size_, PROT_READ, MAP_PRIVATE, g.fd_, 0);
  if (base == MAP_FAILED) {
    g.Close();
    throw std::runtime_error("gguf: mmap failed");
  }
  g.base_ = static_cast<std::uint8_t*>(base);
  g.map_bytes_ = g.size_;

  Cursor c{g.base_, g.base_ + g.size_};
  if (c.Read<std::uint32_t>() != kGgufMagic) {
    g.Close();
    throw std::runtime_error("gguf: bad magic");
  }
  g.version_ = c.Read<std::uint32_t>();
  if (g.version_ != 3) {
    g.Close();
    throw std::runtime_error("gguf: only version 3 is supported");
  }
  const auto tensor_count = c.Read<std::uint64_t>();
  const auto kv_count = c.Read<std::uint64_t>();
  for (std::uint64_t i = 0; i < kv_count; ++i) {
    auto key = c.ReadString();
    const auto kind = c.Read<std::uint32_t>();
    g.meta_.emplace(std::move(key), ReadValue(c, kind));
  }
  g.tensors_.reserve(static_cast<std::size_t>(tensor_count));
  for (std::uint64_t i = 0; i < tensor_count; ++i) {
    TensorInfo t;
    t.name = c.ReadString();
    const auto n_dims = c.Read<std::uint32_t>();
    if (n_dims == 0 || n_dims > 4) {
      g.Close();
      throw std::runtime_error("gguf: bad dimension count");
    }
    t.dims.resize(n_dims);
    for (auto& d : t.dims) d = c.Read<std::uint64_t>();
    const auto type = c.Read<std::uint32_t>();
    if (type >= kTypeCount) {
      g.Close();
      throw std::runtime_error("gguf: unknown tensor type");
    }
    t.type = static_cast<GgmlType>(type);
    t.offset = c.Read<std::uint64_t>();
    std::uint64_t elements = 1;
    for (const auto d : t.dims) elements *= d;
    t.elements = elements;
    const auto block_elements = BlockElements(t.type);
    if (elements % block_elements != 0) {
      g.Close();
      throw std::runtime_error("gguf: elements not block aligned in " + t.name);
    }
    t.bytes = (elements / block_elements) * BlockBytes(t.type);
    g.tensors_.push_back(std::move(t));
  }

  // GGUF aligns the tensor data region to general.alignment (32 by default).
  std::uint64_t alignment = 32;
  if (const auto* meta = g.Meta("general.alignment")) {
    if (meta->kind == MetadataValue::Kind::kUInt && meta->u != 0) {
      alignment = meta->u;
    }
  }
  const auto used = static_cast<std::uint64_t>(c.p - g.base_);
  g.data_offset_ = (used + alignment - 1) / alignment * alignment;
  if (g.data_offset_ > g.size_) {
    g.Close();
    throw std::runtime_error("gguf: tensor data offset past end of file");
  }
  return g;
}

const TensorInfo* Gguf::Find(const std::string& name) const {
  for (const auto& t : tensors_) {
    if (t.name == name) return &t;
  }
  return nullptr;
}

const MetadataValue* Gguf::Meta(const std::string& key) const {
  const auto it = meta_.find(key);
  return it == meta_.end() ? nullptr : &it->second;
}

Gguf Gguf::OpenResident(const std::string& path) {
  Gguf g = Open(path);
  constexpr std::size_t kHuge = std::size_t{2} << 20;
  const std::size_t bytes = (g.size_ + kHuge - 1) & ~(kHuge - 1);
  void* raw = ::mmap(nullptr, bytes + kHuge, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (raw == MAP_FAILED) throw std::runtime_error("gguf: cannot reserve memory for " + path);
  // 2 MB aligned, so whole huge pages map it
  const auto r = reinterpret_cast<std::uintptr_t>(raw);
  const std::uintptr_t a = (r + kHuge - 1) & ~(kHuge - 1);
  if (a > r) ::munmap(raw, a - r);
  if (r + kHuge > a) ::munmap(reinterpret_cast<void*>(a + bytes), r + kHuge - a);
  auto* p = reinterpret_cast<std::uint8_t*>(a);
  ::madvise(p, bytes, MADV_HUGEPAGE);
  for (std::size_t off = 0; off < g.size_;) {
    const ssize_t n = ::pread(g.fd_, p + off, std::min<std::size_t>(g.size_ - off, std::size_t{1} << 30),
                              static_cast<off_t>(off));
    if (n <= 0) {
      ::munmap(p, bytes);
      throw std::runtime_error("gguf: read failed: " + path);
    }
    off += static_cast<std::size_t>(n);
  }
  if (::mlock(p, bytes) != 0) std::fprintf(stderr, "gguf: cannot lock the model in memory (RLIMIT_MEMLOCK)\n");
  ::munmap(g.base_, g.map_bytes_);
  g.base_ = p, g.map_bytes_ = bytes;
  return g;
}

}  // namespace yah::core
