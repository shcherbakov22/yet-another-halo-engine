// Thin RAII wrappers over the HRX native API: device, stream, executable, buffer, and a dispatch helper.
#ifndef YAH_MODEL_LOOM_RUNTIME_HPP_
#define YAH_MODEL_LOOM_RUNTIME_HPP_

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "hrx_runtime.h"

namespace yah::model {

class LoomError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

inline void LoomCheck(hrx_status_t status, const char* what) {
  if (hrx_status_is_ok(status)) return;
  std::string text = what;
  char* message = nullptr;
  size_t length = 0;
  if (hrx_status_is_ok(hrx_status_to_string(status, &message, &length)) && message) {
    text += ": ";
    text.append(message, length);
    hrx_status_free_message(message);
  }
  throw LoomError(text);
}

struct LoomBuffer {
  hrx_buffer_t handle = nullptr;
  size_t size = 0;

  LoomBuffer() = default;
  LoomBuffer(LoomBuffer&& other) noexcept : handle(other.handle), size(other.size) {
    other.handle = nullptr;
    other.size = 0;
  }
  LoomBuffer& operator=(LoomBuffer&& other) noexcept {
    if (this != &other) {
      reset();
      handle = other.handle;
      size = other.size;
      other.handle = nullptr;
      other.size = 0;
    }
    return *this;
  }
  LoomBuffer(const LoomBuffer&) = delete;
  LoomBuffer& operator=(const LoomBuffer&) = delete;
  ~LoomBuffer() { reset(); }
  void reset() {
    if (handle) hrx_buffer_release(handle);
    handle = nullptr;
    size = 0;
  }
};

struct LoomEvent {
  hrx_event_t handle = nullptr;
  LoomEvent() = default;
  LoomEvent(LoomEvent&& other) noexcept : handle(other.handle) { other.handle = nullptr; }
  LoomEvent& operator=(LoomEvent&& other) noexcept {
    if (this != &other) {
      reset();
      handle = other.handle;
      other.handle = nullptr;
    }
    return *this;
  }
  LoomEvent(const LoomEvent&) = delete;
  LoomEvent& operator=(const LoomEvent&) = delete;
  ~LoomEvent() { reset(); }
  void reset() {
    if (handle) hrx_event_release(handle);
    handle = nullptr;
  }
};

struct LoomExecutable {
  hrx_executable_t handle = nullptr;
  std::vector<std::string> names;
  std::vector<hrx_executable_export_info_t> infos;

  LoomExecutable() = default;
  LoomExecutable(LoomExecutable&& other) noexcept
      : handle(other.handle), names(std::move(other.names)), infos(std::move(other.infos)) {
    other.handle = nullptr;
    other.infos.clear();
  }
  LoomExecutable& operator=(LoomExecutable&& other) noexcept {
    if (this != &other) {
      reset();
      handle = other.handle;
      names = std::move(other.names);
      infos = std::move(other.infos);
      other.handle = nullptr;
      other.infos.clear();
    }
    return *this;
  }
  LoomExecutable(const LoomExecutable&) = delete;
  LoomExecutable& operator=(const LoomExecutable&) = delete;
  ~LoomExecutable() { reset(); }
  void reset() {
    if (handle) hrx_executable_release(handle);
    handle = nullptr;
    names.clear();
    infos.clear();
  }
  [[nodiscard]] uint32_t Ordinal(const std::string& name) const {
    for (size_t i = 0; i < names.size(); ++i) {
      if (names[i] == name) return static_cast<uint32_t>(i);
    }
    throw LoomError("export not found: " + name);
  }
  [[nodiscard]] uint32_t OrdinalOrZero(const std::string& name) const {
    for (size_t i = 0; i < names.size(); ++i) {
      if (names[i] == name) return static_cast<uint32_t>(i);
    }
    return 0;
  }
  // The workgroup size this export was compiled with, or 0 if the metadata does not carry it.
  // Do not hardcode it: a wave64 kernel launched with 32 threads computes a wrong tile and looks fast.
  [[nodiscard]] uint32_t WorkgroupSize(uint32_t ordinal) const {
    if (ordinal < infos.size() && infos[ordinal].workgroup_size[0] != 0) {
      return infos[ordinal].workgroup_size[0];
    }
    return 0;
  }
  // How many buffers this export's dispatch binds. The GEMM family is not uniform:
  // IQ grid formats bind (weight, grid, [ksigns], input, wstage, ostage, out): 7 for iq3xxs/iq2xxs/iq2xs, 6 for iq3s.
  // Every other format binds (weight, input, wstage, ostage, out) = 5.
  [[nodiscard]] uint32_t BindingCount(uint32_t ordinal) const {
    if (ordinal < infos.size()) return infos[ordinal].binding_count;
    return 0;
  }
};

class LoomDevice {
 public:
  LoomDevice() {
    LoomCheck(hrx_gpu_initialize(0), "hrx_gpu_initialize");
    initialized_ = true;
    if (!hrx_status_is_ok(hrx_gpu_device_get(0, &device_))) {
      hrx_gpu_shutdown();
      throw LoomError("hrx_gpu_device_get");
    }
    LoomCheck(hrx_stream_create(device_, 0, &stream_), "hrx_stream_create");
  }
  LoomDevice(const LoomDevice&) = delete;
  LoomDevice& operator=(const LoomDevice&) = delete;
  ~LoomDevice() {
    if (profiling_) hrx_device_profile_dispatches_end(device_);
    if (stream_) hrx_stream_release(stream_);
    // device_ is borrowed (hrx_gpu_device_get does not retain it); hrx_gpu_shutdown releases it.
    // Do not release it here: that clears the device early and HRX_PROFILE_FILE gets no dispatch events or session_end.
    if (initialized_) hrx_gpu_shutdown();
  }

  [[nodiscard]] hrx_device_t device() const { return device_; }

  // Device timestamps of completed dispatches (HRX patch 0006). ProfileFlush / ProfileEnd hand the dispatches
  // completed so far to the sink, with the count of records the device dropped; Synchronize first to get them all.
  using DispatchSink = std::function<void(const hrx_profile_dispatch_t*, size_t, uint64_t)>;
  void ProfileBegin(DispatchSink sink) {
    sink_ = std::move(sink);
    LoomCheck(hrx_device_profile_dispatches_begin(device_, &LoomDevice::OnDispatches, this),
              "hrx_device_profile_dispatches_begin");
    profiling_ = true;
  }
  void ProfileFlush() { LoomCheck(hrx_device_profile_dispatches_flush(device_), "hrx_device_profile_dispatches_flush"); }
  void ProfileEnd() {
    profiling_ = false;
    LoomCheck(hrx_device_profile_dispatches_end(device_), "hrx_device_profile_dispatches_end");
  }
  [[nodiscard]] bool profiling() const { return profiling_; }
  [[nodiscard]] hrx_stream_t stream() const { return stream_; }

  [[nodiscard]] LoomExecutable Load(const std::string& path, const char* target_key = "gfx1151") {
    LoomExecutable executable;
    LoomCheck(hrx_executable_load_file(device_, path.c_str(), "amdgpu", target_key, &executable.handle),
              "hrx_executable_load_file");
    size_t count = 0;
    LoomCheck(hrx_executable_export_count(executable.handle, &count), "hrx_executable_export_count");
    executable.names.reserve(count);
    executable.infos.resize(count);
    for (size_t i = 0; i < count; ++i) {
      LoomCheck(hrx_executable_export_info(executable.handle, static_cast<uint32_t>(i), &executable.infos[i]),
                "hrx_executable_export_info");
      executable.names.emplace_back(executable.infos[i].name ? executable.infos[i].name : "");
    }
    return executable;
  }

  [[nodiscard]] LoomBuffer Allocate(size_t bytes) {
    LoomBuffer buffer;
    buffer.size = bytes;
    LoomCheck(
        hrx_buffer_allocate(stream_, bytes, HRX_MEMORY_TYPE_DEVICE_LOCAL, HRX_BUFFER_USAGE_DEFAULT, &buffer.handle),
        "hrx_buffer_allocate");
    return buffer;
  }

  // Import an external host pointer (e.g. a GGUF mmap window) as an HRX buffer.
  // The caller must keep the mapping alive while the buffer is in use.
  [[nodiscard]] LoomBuffer Import(void* host_ptr, size_t bytes) {
    LoomBuffer buffer;
    buffer.size = bytes;
    hrx_buffer_params_t params = {};
    params.type = HRX_MEMORY_TYPE_DEVICE_VISIBLE;
    params.access = HRX_MEMORY_ACCESS_READ;
    params.usage = HRX_BUFFER_USAGE_DEFAULT;
    params.queue_affinity = 0;
    LoomCheck(hrx_allocator_import_buffer(hrx_device_allocator(device_), params, host_ptr, bytes, &buffer.handle),
              "hrx_allocator_import_buffer");
    return buffer;
  }

  // Fill the whole buffer with a 32-bit pattern, in stream order.
  void Fill(const LoomBuffer& buffer, uint32_t pattern) {
    LoomCheck(hrx_stream_fill_buffer(stream_, buffer.handle, 0, buffer.size, &pattern, 4), "hrx_stream_fill_buffer");
  }
  void H2D(const LoomBuffer& buffer, const void* host, size_t bytes, size_t offset = 0) {
    LoomCheck(hrx_synchronous_h2d(device_, host, buffer.handle, offset, bytes), "hrx_synchronous_h2d");
  }
  void D2H(const LoomBuffer& buffer, void* host, size_t bytes, size_t offset = 0) {
    LoomCheck(hrx_synchronous_d2h(device_, buffer.handle, offset, host, bytes), "hrx_synchronous_d2h");
  }

  void Dispatch(const LoomExecutable& executable, uint32_t ordinal, const hrx_dispatch_config_t& config,
                const void* constants, size_t constants_size, const hrx_buffer_ref_t* bindings, size_t binding_count) {
    const uint32_t flags = no_barrier_ok_ ? next_flags_ : 0;
    next_flags_ = 0;
    hrx_status_t status = hrx_stream_dispatch(stream_, executable.handle, ordinal, &config, constants, constants_size,
                                              bindings, binding_count, flags);
    // Stock HRX rejects the no-barrier flag up front (nothing recorded): retry with ordered dispatch from now on.
    if (flags && hrx_status_code(status) == HRX_STATUS_INVALID_ARGUMENT) {
      hrx_status_ignore(status);
      no_barrier_ok_ = false;
      status = hrx_stream_dispatch(stream_, executable.handle, ordinal, &config, constants, constants_size, bindings,
                                   binding_count, 0);
    }
    LoomCheck(status, "hrx_stream_dispatch");
  }

  // The next Dispatch may overlap the one after it (no trailing ordering barrier). Needs the local libhrx flag
  // HRX_DISPATCH_FLAG_NO_ORDERING_BARRIER (bit 2, not upstream); on stock HRX dispatches stay ordered.
  void NoBarrierNext() { next_flags_ = 1u << 2; }
  // Sleep-poll synchronize: with us > 0, poll an event at the stream tail and sleep us between checks.
  // The runtime's blocking wait busy-polls a host core (ROCr); a long queued run (the prefill) opts in.
  // Off by default: on short single waits the sleep slack shows in the timing.
  void SetSleepSync(long us) { sleep_us_ = us; }
  void Synchronize() {
    if (sleep_us_ > 0) {
      // hrx_stream_query reports complete while the stream timepoint is 0 (true for plain dispatches): poll an event.
      LoomEvent tail;
      LoomCheck(hrx_event_create(device_, HRX_EVENT_FLAG_NONE, &tail.handle), "hrx_event_create");
      LoomCheck(hrx_event_record(tail.handle, stream_), "hrx_event_record");
      bool complete = false;
      for (;;) {
        LoomCheck(hrx_event_query(tail.handle, &complete), "hrx_event_query");
        if (complete) break;
        std::this_thread::sleep_for(std::chrono::microseconds(sleep_us_));
      }
    }
    LoomCheck(hrx_stream_synchronize(stream_), "sync");
  }

  [[nodiscard]] LoomEvent NewEvent() {
    LoomEvent event;
    LoomCheck(hrx_event_create(device_, HRX_EVENT_FLAG_NONE, &event.handle), "hrx_event_create");
    return event;
  }
  void Record(LoomEvent& event) { LoomCheck(hrx_event_record(event.handle, stream_), "hrx_event_record"); }
  float Elapsed(LoomEvent& start, LoomEvent& stop) {
    float ms = 0.0f;
    LoomCheck(hrx_event_elapsed_time(start.handle, stop.handle, &ms), "hrx_event_elapsed_time");
    return ms;
  }

  static hrx_dispatch_config_t Config(uint32_t gx, uint32_t gy, uint32_t gz, uint32_t sx, uint32_t sy, uint32_t sz,
                                      uint32_t subgroup = 32) {
    hrx_dispatch_config_t config = {};
    config.workgroup_count[0] = gx;
    config.workgroup_count[1] = gy;
    config.workgroup_count[2] = gz;
    config.workgroup_size[0] = sx;
    config.workgroup_size[1] = sy;
    config.workgroup_size[2] = sz;
    config.subgroup_size = subgroup;
    return config;
  }

 private:
  hrx_device_t device_ = nullptr;
  hrx_stream_t stream_ = nullptr;
  uint32_t next_flags_ = 0;
  bool no_barrier_ok_ = true;
  long sleep_us_ = 0;
  bool initialized_ = false;
  bool profiling_ = false;
  DispatchSink sink_;

  static void OnDispatches(void* user, const hrx_profile_dispatch_t* events, size_t count, uint64_t dropped) {
    auto* self = static_cast<LoomDevice*>(user);
    if (self->sink_) self->sink_(events, count, dropped);
  }
};

// Records dispatches into one HRX graph and launches it on the device stream. Each dispatch waits only for what it
// touches: the last earlier write to any byte range it reads or writes, and, for a write, the reads since then.
// Independent dispatches get no edge and HRX records no barrier between them, so they can run at the same time on the
// GPU. (Stream dispatches always end with an ordering barrier; graphs are the stock-HRX way to overlap kernels.)
class LoomGraph {
 public:
  explicit LoomGraph(LoomDevice& gpu) : gpu_(gpu) {
    LoomCheck(hrx_graph_create(gpu.device(), 0, &graph_), "hrx_graph_create");
  }
  ~LoomGraph() {
    if (exec_) hrx_graph_exec_release(exec_);
    if (graph_) hrx_graph_release(graph_);
  }
  LoomGraph(const LoomGraph&) = delete;
  LoomGraph& operator=(const LoomGraph&) = delete;

  // A buffer no recorded dispatch writes (weights, lookup tables): it orders nothing.
  void ReadOnly(hrx_buffer_t buffer) { read_only_.insert(buffer); }

  // writes: bit i set if the dispatch may write binding i. A missing bit is a race; an extra one only costs overlap.
  // Returns the node's index: dispatch timestamps of the launched graph carry it as their command index.
  // after: ranges this dispatch is ordered after as if it read them, though it does not (it gets the same barrier as a
  // dispatch that does, so the two can run together); not recorded as accesses.
  // also_writes: ranges recorded as written by this dispatch though it does not bind them (a wait that stands for
  // another engine's writes).
  size_t Dispatch(const LoomExecutable& executable, uint32_t ordinal, const hrx_dispatch_config_t& config,
                  const hrx_buffer_ref_t* bindings, size_t binding_count, uint64_t writes,
                  const std::vector<hrx_buffer_ref_t>* after = nullptr,
                  const std::vector<hrx_buffer_ref_t>* also_writes = nullptr) {
    std::vector<hrx_graph_node_t> deps;
    for (size_t i = 0; i < binding_count; ++i)
      if (!read_only_.count(bindings[i].buffer)) Depend(bindings[i], (writes >> i) & 1, &deps);
    if (after)
      for (const hrx_buffer_ref_t& r : *after)
        if (!read_only_.count(r.buffer)) Depend(r, false, &deps);
    if (also_writes)
      for (const hrx_buffer_ref_t& r : *also_writes) Depend(r, true, &deps);
    Unique(&deps);
    // The graph keeps the binding pointer until it is instantiated.
    const auto& b = bindings_.emplace_back(bindings, bindings + binding_count);
    hrx_graph_kernel_node_attrs_t attrs{};
    attrs.executable = executable.handle;
    attrs.export_ordinal = ordinal;
    attrs.config = config;
    attrs.bindings = b.data();
    attrs.binding_count = b.size();
    hrx_graph_node_t node = nullptr;
    LoomCheck(hrx_graph_add_kernel_node(graph_, deps.data(), deps.size(), &attrs, &node), "hrx_graph_add_kernel_node");
    for (size_t i = 0; i < binding_count; ++i)
      if (!read_only_.count(bindings[i].buffer))
        accesses_[bindings[i].buffer].push_back({bindings[i].offset, bindings[i].length, ((writes >> i) & 1) != 0, node});
    if (also_writes)
      for (const hrx_buffer_ref_t& r : *also_writes) accesses_[r.buffer].push_back({r.offset, r.length, true, node});
    grids_.push_back({config.workgroup_count[0], config.workgroup_count[1], config.workgroup_count[2]});
    return grids_.size() - 1;
  }
  // Stores value to target (4 bytes) after everything that wrote reads (HRX patch 0015). With release + system scope the
  // command processor makes every earlier write visible to the host and other devices first. writes: ranges recorded as
  // written by the store (another engine writes them once it sees the value).
  void AtomicStore(const hrx_buffer_ref_t& target, uint32_t value, uint32_t flags,
                   const std::vector<hrx_buffer_ref_t>& reads, const std::vector<hrx_buffer_ref_t>& writes) {
    std::vector<hrx_graph_node_t> deps;
    Depend(target, true, &deps);
    for (const hrx_buffer_ref_t& r : reads) Depend(r, false, &deps);
    for (const hrx_buffer_ref_t& r : writes) Depend(r, true, &deps);
    Unique(&deps);
    hrx_graph_atomic_store_node_attrs_t attrs{};
    attrs.target = target, attrs.value = value, attrs.flags = flags;
    hrx_graph_node_t node = nullptr;
    LoomCheck(hrx_graph_add_atomic_store_node(graph_, deps.data(), deps.size(), &attrs, &node),
              "hrx_graph_add_atomic_store_node");
    accesses_[target.buffer].push_back({target.offset, target.length, true, node});
    for (const hrx_buffer_ref_t& r : reads) accesses_[r.buffer].push_back({r.offset, r.length, false, node});
    for (const hrx_buffer_ref_t& r : writes) accesses_[r.buffer].push_back({r.offset, r.length, true, node});
  }
  // The workgroup count node i was recorded with.
  [[nodiscard]] const std::array<uint32_t, 3>& Grid(size_t i) const { return grids_[i]; }
  [[nodiscard]] size_t size() const { return grids_.size(); }

  // Instantiates the graph now, so that Launch only queues it (segments joined by host waits launch back to back).
  void Instantiate() {
    if (!exec_) LoomCheck(hrx_graph_instantiate(graph_, 0, &exec_), "hrx_graph_instantiate");
  }
  // Queues the graph on the device stream, after everything queued before (instantiates it first if needed).
  void Launch() {
    Instantiate();
    LoomCheck(hrx_graph_exec_launch(exec_, gpu_.stream()), "hrx_graph_exec_launch");
  }

 private:
  static void Unique(std::vector<hrx_graph_node_t>* deps) {
    std::sort(deps->begin(), deps->end());
    deps->erase(std::unique(deps->begin(), deps->end()), deps->end());
  }
  struct Access {
    size_t offset, length;
    bool write;
    hrx_graph_node_t node;
  };
  // Newest first: every overlapping earlier write orders this access, a write also waits for the overlapping reads.
  // A write that covers the whole range was itself ordered after everything older, so the scan stops there.
  void Depend(const hrx_buffer_ref_t& b, bool write, std::vector<hrx_graph_node_t>* deps) {
    const auto& list = accesses_[b.buffer];
    for (auto it = list.rbegin(); it != list.rend(); ++it) {
      if (it->offset >= b.offset + b.length || b.offset >= it->offset + it->length) continue;
      if (it->write) {
        deps->push_back(it->node);
        if (it->offset <= b.offset && it->offset + it->length >= b.offset + b.length) return;
      } else if (write) {
        deps->push_back(it->node);
      }
    }
  }

  LoomDevice& gpu_;
  hrx_graph_t graph_ = nullptr;
  hrx_graph_exec_t exec_ = nullptr;
  std::set<hrx_buffer_t> read_only_;
  std::map<hrx_buffer_t, std::vector<Access>> accesses_;
  std::deque<std::vector<hrx_buffer_ref_t>> bindings_;
  std::vector<std::array<uint32_t, 3>> grids_;
};

// The model's tensor-data region (the GGUF mmap) imported once as one device-visible buffer; every tensor is an offset
// into it. Import needs a page-aligned start, so delta() is the offset of the region inside the import.
class LoomWeights {
 public:
  LoomWeights(LoomDevice& gpu, const void* base, size_t bytes) {
    const uintptr_t page = 4096;
    const uintptr_t start = reinterpret_cast<uintptr_t>(base) & ~(page - 1);
    delta_ = reinterpret_cast<uintptr_t>(base) - start;
    buffer_ = gpu.Import(reinterpret_cast<void*>(start), bytes + delta_);
  }
  [[nodiscard]] hrx_buffer_t handle() const { return buffer_.handle; }
  [[nodiscard]] size_t delta() const { return delta_; }

 private:
  LoomBuffer buffer_;
  size_t delta_ = 0;
};

}  // namespace yah::model

#endif  // YAH_MODEL_LOOM_RUNTIME_HPP_