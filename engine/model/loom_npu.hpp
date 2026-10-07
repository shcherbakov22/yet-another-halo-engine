// NPU (XDNA2) GEMMs beside the GPU: libamdf for the device, the HRX xdna loader for .xdna images.
// Operands live in HRX device buffers (full GPU bandwidth), exported once as a DMA-BUF through HSA and imported per binding view into the NPU.
// A relay thread waits for a GPU timeline point, submits the NPU work and signals an HRX semaphore the GPU stream waits on.
// Needs HRX patches 0013 / 0014 (see engine/hrx/patches).
#ifndef YAH_MODEL_LOOM_NPU_HPP_
#define YAH_MODEL_LOOM_NPU_HPP_

#include <dlfcn.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <sys/mman.h>
#include <deque>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "amdf/amdf.h"
#include "amdf/xdna.h"
#include "experimental/xdna/amdf_status.h"
#include "experimental/xdna/executable.h"
#include "iree/base/byte_sequence.h"
#include "iree/hal/drivers/amd/xdna/image/aie2p/npu2.h"
#include "model/loom_runtime.hpp"
#include "model/npu_split.hpp"

namespace yah::model {

inline void NpuCheck(iree_status_t status, const char* what) {
  if (iree_status_is_ok(status)) return;
  char* text = nullptr;
  iree_host_size_t length = 0;
  std::string message = std::string("npu: ") + what;
  const iree_allocator_t allocator = iree_allocator_system();
  if (iree_status_to_string(status, &allocator, &text, &length)) {
    message += ": " + std::string(text, length);
    iree_allocator_free(iree_allocator_system(), text);
  }
  iree_status_free(status);
  throw LoomError(message);
}
#define YAH_AMDF(expr, what) NpuCheck(IREE_HAL_AMD_STATUS_FROM_AMDF((expr), what), what)

class LoomNpu {
 public:
  // A GPU buffer shared with the NPU (views of it imported per binding through its DMA-BUF).
  struct Shared {
    LoomBuffer gpu;
    void* device = nullptr;
    std::size_t bytes = 0;
    int fd = -1;
    std::uint64_t fd_offset = 0;
    void* host = nullptr;              // host pages (CreateShared host), else none
    amdf_memory_t* native = nullptr;   // their libamdf registration, bound directly
  };
  struct View {
    Shared* shared;
    std::size_t offset, length;
  };
  // One loaded instance of an image with its bindings and native command storage.
  struct Kernel {
    const void* image_key = nullptr;
    amdf_xdna_kernel_command_t first{}, repeat{};
    // A streamed image (gen_npu_gemm groups: five invocations) queues a call with push[parity] and retires the
    // oldest queued call with wait; lead is the even push with its weight fills unpaced, for the first call of a run
    // (nothing computes yet that pacing would protect); first sets the array up and runs one whole call.
    bool stream = false;
    amdf_xdna_kernel_command_t push[2]{}, wait{}, lead{};
    std::vector<iree_hal_amd_xdna_executable_storage_t> storage;
    std::vector<amdf_host_mapping_t*> storage_maps;
    std::vector<amdf_memory_t*> imports;
    std::vector<iree_hal_buffer_t*> buffers;
  };

  LoomNpu(LoomDevice& gpu, std::uint32_t columns) : gpu_(gpu), columns_(columns) {
    YAH_AMDF(amdf_query_api(AMDF_ABI_VERSION_LATEST, AMDF_ABI_VERSION_LATEST, &api_), "query_api");
    const void* ext = nullptr;
    YAH_AMDF(api_->query_extension(AMDF_EXTENSION_XDNA, AMDF_XDNA_EXTENSION_VERSION_1,
                                   AMDF_XDNA_EXTENSION_VERSION_LATEST, &ext),
             "query_extension(XDNA)");
    xdna_ = static_cast<const amdf_xdna_api_t*>(ext);
    amdf_instance_create_info_t ici{};
    ici.type = AMDF_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.structure_size = sizeof(ici);
    YAH_AMDF(api_->instance_create(&ici, &instance_), "instance_create");
    SelectScope();
    OpenEndpoint();
    CreateDevice();
    // HRX's own libhsa (already loaded) exports its allocations as DMA-BUFs.
    void* hsa = dlopen("libhsa-runtime64.so.1", RTLD_NOW | RTLD_NOLOAD);
    if (hsa) export_dmabuf_ = reinterpret_cast<ExportFn>(dlsym(hsa, "hsa_amd_portable_export_dmabuf"));
    if (!export_dmabuf_) throw LoomError("npu: hsa_amd_portable_export_dmabuf not found");
    LoomCheck(hrx_semaphore_create(gpu_.device(), 0, &done_), "hrx_semaphore_create");
    // Copies from / to a host-visible buffer carry system-scope acquire / release fences (HRX blits), the GPU cache
    // maintenance around the NPU's accesses that a graph segment boundary does not give.
    LoomCheck(hrx_buffer_allocate(gpu_.stream(), 64, HRX_MEMORY_TYPE_HOST_LOCAL, HRX_BUFFER_USAGE_TRANSFER, &fence_host_.handle),
              "hrx_buffer_allocate(fence host)");
    fence_host_.size = 64;
    fence_dev_ = gpu_.Allocate(64);
    relay_ = std::thread([this] { Relay(); });
  }
  LoomNpu(const LoomNpu&) = delete;
  LoomNpu& operator=(const LoomNpu&) = delete;
  ~LoomNpu() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    if (relay_.joinable()) relay_.join();
    for (auto& a : arenas_) {
      api_->host_mapping_destroy(a.map);
      api_->memory_destroy(a.storage.memory);
    }
    for (auto& k : kernels_) {
      for (auto* m : k->storage_maps) api_->host_mapping_destroy(m);
      for (auto& s : k->storage) api_->memory_destroy(s.memory);
      for (auto* b : k->buffers) iree_hal_buffer_release(b);
      for (auto* m : k->imports) api_->memory_destroy(m);
    }
    for (auto& [key, image] : images_) iree_hal_amd_xdna_image_destroy(image);
    for (auto& s : shared_) {
      if (s->host) {   // the HRX import first, then the registration, then the pages
        s->gpu.reset();
        api_->memory_destroy(s->native);
        munmap(s->host, s->bytes);
      } else {
        close(s->fd);
      }
    }
    if (queue_) api_->kernel_queue_destroy(queue_);
    if (context_) xdna_->context_destroy(context_);
    if (device_) api_->device_destroy(device_);
    if (endpoint_) api_->endpoint_close(endpoint_);
    if (instance_) api_->instance_destroy(instance_);
    if (done_) hrx_semaphore_release(done_);
  }

  // A zeroed GPU buffer the NPU can bind views of.
  // host: anonymous host pages registered with libamdf and imported into HRX instead of a device buffer the NPU imports.
  // C uses them: with all shared buffers in device memory, deferred data-fabric machine checks (some corrupting an NPU
  // output) were more frequent; their trigger is the SMU switching fabric / memory clocks during NPU work (results.md).
  Shared& CreateShared(std::size_t bytes, bool host = false) {
    auto s = std::make_unique<Shared>();
    s->bytes = Align(bytes);
    if (host) {
      s->host = mmap(nullptr, s->bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
      if (s->host == MAP_FAILED) throw LoomError("npu: mmap");
      amdf_memory_create_info_t ci{};
      ci.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
      ci.structure_size = sizeof(ci);
      ci.memory_profile_ordinal = RegisterProfile();
      ci.access_count = 1;
      ci.accesses = &access_;
      ci.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
      ci.byte_length = s->bytes;
      ci.minimum_alignment = 4096;
      ci.registered_host_pointer = s->host;
      ci.registered_host_cacheability = AMDF_HOST_CACHEABILITY_WRITE_BACK;
      YAH_AMDF(api_->memory_create(scope_, &ci, &s->native), "memory_create(registered host)");
      s->device = s->host;
      s->gpu.size = s->bytes;
      hrx_buffer_params_t params = {};
      params.type = HRX_MEMORY_TYPE_DEVICE_VISIBLE;
      params.access = HRX_MEMORY_ACCESS_READ | HRX_MEMORY_ACCESS_WRITE;
      params.usage = HRX_BUFFER_USAGE_DEFAULT;
      LoomCheck(hrx_allocator_import_buffer(hrx_device_allocator(gpu_.device()), params, s->host, s->bytes, &s->gpu.handle),
                "hrx_allocator_import_buffer(host)");
      gpu_.Fill(s->gpu, 0);
      gpu_.Synchronize();
      shared_.push_back(std::move(s));
      return *shared_.back();
    }
    s->gpu = gpu_.Allocate(s->bytes);
    gpu_.Fill(s->gpu, 0);
    gpu_.Synchronize();
    LoomCheck(hrx_buffer_get_device_ptr(s->gpu.handle, &s->device), "hrx_buffer_get_device_ptr");
    if (export_dmabuf_(s->device, s->bytes, &s->fd, &s->fd_offset) != 0) throw LoomError("npu: dma-buf export");
    shared_.push_back(std::move(s));
    return *shared_.back();
  }
  // Shared buffers round to pages.
  [[nodiscard]] static std::size_t Align(std::size_t bytes) { return (bytes + 4095) / 4096 * 4096; }

  // A new instance of the image at path (entry), its bindings the given views.
  Kernel& Load(const std::string& path, const std::string& entry, const std::vector<View>& views) {
    iree_hal_amd_xdna_image_t* image = Image(path);
    uint32_t entry_ordinal = 0;
    NpuCheck(iree_hal_amd_xdna_image_find_entry(image, iree_make_string_view(entry.data(), entry.size()),
                                                &entry_ordinal),
             "image_find_entry");
    const iree_hal_amd_xdna_image_tables_t* tables = iree_hal_amd_xdna_image_tables(image);
    const iree_xdna_elf_entry_record_t rec = iree_hal_amd_xdna_image_tables_entry(tables, entry_ordinal);
    if (rec.binding_count != views.size()) throw LoomError("npu: " + path + ": binding count");
    auto k = std::make_unique<Kernel>();
    k->image_key = image;
    std::vector<iree_hal_amd_xdna_executable_binding_t> bindings(views.size());
    for (std::size_t i = 0; i < views.size(); ++i) {
      const View& v = views[i];
      if (v.offset + v.length > v.shared->bytes) throw LoomError("npu: bad view");
      const iree_xdna_elf_binding_record_t contract =
          iree_hal_amd_xdna_image_tables_binding(tables, rec.first_binding + static_cast<uint32_t>(i));
      amdf_memory_import_info_t ii{};
      ii.type = AMDF_STRUCTURE_TYPE_MEMORY_IMPORT_INFO;
      ii.structure_size = sizeof(ii);
      ii.memory_profile_ordinal = profile_.ordinal;
      ii.access_count = 1;
      ii.minimum_alignment = contract.minimum_alignment;
      ii.accesses = &access_;
      amdf_external_memory_t em{};   // borrowed: the fd stays ours
      em.type = AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD;
      em.payload.file_descriptor = v.shared->fd;
      em.source_byte_offset = v.shared->fd_offset + v.offset;
      em.byte_length = v.length;
      amdf_memory_t* mem = v.shared->native;
      std::uint64_t addr = 0, mem_off = 0;
      if (mem) {   // registered host pages: bound at the view's offset
        mem_off = v.offset;
      } else {
        YAH_AMDF(api_->memory_import(scope_, &ii, &em, &mem), "memory_import");
        k->imports.push_back(mem);
      }
      YAH_AMDF(api_->memory_query_address(mem, 0, AMDF_MEMORY_ADDRESS_XDNA_DMA, &addr), "memory_query_address");
      addr += mem_off;
      // executable_bind checks this wrapper's range and access only; the span is never dereferenced.
      iree_hal_buffer_t* buffer = nullptr;
      NpuCheck(iree_hal_heap_buffer_wrap(
                   iree_hal_buffer_placement_undefined(),
                   IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
                       IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
                   IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE, IREE_HAL_BUFFER_USAGE_STORAGE,
                   v.length, iree_make_byte_span(static_cast<std::uint8_t*>(v.shared->device) + v.offset, v.length),
                   iree_hal_buffer_release_callback_null(), iree_allocator_system(), &buffer),
               "heap_buffer_wrap");
      k->buffers.push_back(buffer);
      bindings[i].buffer_ref = iree_hal_make_buffer_ref(buffer, 0, v.length);
      bindings[i].memory = mem;
      bindings[i].memory_byte_offset = mem_off;
      bindings[i].device_address = addr;
    }
    k->storage.resize(rec.allocation_use_count);
    k->storage_maps.resize(rec.allocation_use_count);
    for (uint32_t i = 0; i < rec.allocation_use_count; ++i) {
      const uint32_t ord = iree_hal_amd_xdna_image_tables_allocation_use(tables, rec.first_allocation_use + i);
      AllocateStorage(iree_hal_amd_xdna_image_tables_allocation(tables, ord), &k->storage[i], &k->storage_maps[i]);
    }
    const auto n = static_cast<iree_host_size_t>(k->storage.size());
    NpuCheck(iree_hal_amd_xdna_executable_load(image, entry_ordinal, n, k->storage.data()), "executable_load");
    NpuCheck(iree_hal_amd_xdna_executable_bind(image, entry_ordinal, n, k->storage.data(), bindings.size(),
                                               bindings.data()),
             "executable_bind");
    for (uint32_t i = 0; i < n; ++i)
      YAH_AMDF(api_->host_mapping_cache_control(k->storage_maps[i], AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
                                                k->storage[i].mapping.data_length),
               "host_mapping_cache_control(storage)");
    NpuCheck(iree_hal_amd_xdna_executable_query_continuation(image, entry_ordinal, n, k->storage.data(), false,
                                                             &k->first),
             "query_invocation");
    NpuCheck(iree_hal_amd_xdna_executable_query_continuation(image, entry_ordinal, n, k->storage.data(), true,
                                                             &k->repeat),
             "query_continuation");
    if (rec.invocation_count == 5) {
      k->stream = true;
      amdf_xdna_kernel_command_t* const out[] = {&k->push[0], &k->push[1], &k->wait, &k->lead};
      for (uint32_t i = 0; i < 4; ++i)
        NpuCheck(iree_hal_amd_xdna_executable_query_invocation_ordinal(image, entry_ordinal, n, k->storage.data(), i + 1,
                                                                       out[i]),
                 "query_invocation_ordinal");
    }
    kernels_.push_back(std::move(k));
    return *kernels_.back();
  }

  // Queue NPU work behind the GPU stream's current position; the relay runs the kernels in order after it.
  // Pass the returned value to Join before the stream reads the results.
  std::uint64_t Enqueue(std::vector<Kernel*> kernels, std::string tag = {}) {
    // system-scope release: the GPU's writes to A / W reach memory before the NPU reads them
    LoomCheck(hrx_stream_copy_buffer(gpu_.stream(), fence_dev_.handle, 0, fence_host_.handle, 0, 64), "fence release");
    LoomCheck(hrx_stream_flush(gpu_.stream()), "hrx_stream_flush");
    hrx_timeline_point_t after{};
    LoomCheck(hrx_stream_get_timeline_position(gpu_.stream(), &after), "hrx_stream_get_timeline_position");
    std::lock_guard<std::mutex> lock(mu_);
    jobs_.push_back({after, std::move(kernels), ++issued_, std::move(tag)});
    cv_.notify_one();
    return issued_;
  }
  // Orders the stream after the NPU work Enqueue returned value for; queue the GPU work that runs beside the NPU first.
  // The host waits for the NPU here, so the stream wait sees a signaled semaphore.
  // A stream wait on a semaphore the host signals later is resolved in software after the stream drains (~0.1-0.2 ms idle GPU).
  void Join(std::uint64_t value) {
    LoomCheck(hrx_stream_flush(gpu_.stream()), "hrx_stream_flush");
    LoomCheck(hrx_semaphore_wait(done_, value, UINT64_MAX), "hrx_semaphore_wait(npu)");
    LoomCheck(hrx_stream_wait_on(gpu_.stream(), {done_, value}), "hrx_stream_wait_on");
    // system-scope acquire: no GPU cache keeps lines of C from an earlier read (the wait above, already signaled, is
    // resolved in software without a fence)
    LoomCheck(hrx_stream_copy_buffer(gpu_.stream(), fence_host_.handle, 0, fence_dev_.handle, 0, 64), "fence acquire");
  }
  // Per job tag: jobs, calls, NPU busy time (first submission to completion, host clock).
  struct TagStats {
    std::size_t jobs = 0, calls = 0;
    double busy_ms = 0, join_ms = 0;
  };
  std::map<std::string, TagStats> Stats() {
    std::lock_guard<std::mutex> lock(mu_);
    return stats_;
  }
  void AddJoinMs(const std::string& tag, double ms) {
    std::lock_guard<std::mutex> lock(mu_);
    stats_[tag].join_ms += ms;
  }
  // Host time of the last finished job from its first submission to the NPU's completion.
  [[nodiscard]] double LastJobMs() const { return last_job_ms_.load(); }
  // Throws if the relay saw an NPU failure (the fence signals on errors too).
  void CheckHealth() {
    std::lock_guard<std::mutex> lock(mu_);
    if (!failure_.empty()) throw LoomError(failure_);
  }

 private:
  // A streamed run of calls as one command: [lead c0][push c1][wait][push c2][wait] ... [wait][wait] (pushes on
  // alternating descriptor sets, at most two calls queued), the bodies of the calls' own relocated commands under one
  // native transaction header (format 0.1: operation count at byte 8, byte length at 12). Built once per call list
  // into command arenas (allocating per command cost ~1 ms on the relay thread).
  struct Arena {
    iree_hal_amd_xdna_executable_storage_t storage{};
    amdf_host_mapping_t* map = nullptr;
    std::size_t used = 0;
  };
  static constexpr std::size_t kArenaBytes = 8u << 20, kCommandAlign = 32768;
  static const std::uint8_t* CommandBytes(const Kernel& k, const amdf_xdna_kernel_command_t& c) {
    for (const auto& st : k.storage)
      if (st.memory == c.memory) return static_cast<const std::uint8_t*>(st.mapping.data) + (c.byte_offset - st.memory_byte_offset);
    throw LoomError("npu: command outside its kernel storage");
  }
  const amdf_xdna_kernel_command_t& FusedCommand(const std::vector<Kernel*>& calls) {
    if (const auto it = fused_.find(calls); it != fused_.end()) return it->second;
    std::vector<std::uint8_t> bytes;
    std::uint32_t ops = 0;
    auto append = [&](const Kernel& k, const amdf_xdna_kernel_command_t& c) {
      const std::uint8_t* src = CommandBytes(k, c);
      std::uint32_t n = 0, size = 0;
      std::memcpy(&n, src + 8, 4), std::memcpy(&size, src + 12, 4);
      if (size != c.byte_length) throw LoomError("npu: unexpected command header");
      if (bytes.empty()) bytes.assign(src, src + 16);
      bytes.insert(bytes.end(), src + 16, src + size);
      ops += n;
    };
    int queued = 0;
    for (std::size_t i = 0; i < calls.size(); ++i) {
      append(*calls[i], i == 0 ? calls[i]->lead : calls[i]->push[i & 1]);
      if (++queued == 2) append(*calls[i], calls[i]->wait), --queued;
    }
    for (; queued > 0; --queued) append(*calls.back(), calls.back()->wait);
    const std::uint32_t size = static_cast<std::uint32_t>(bytes.size());
    std::memcpy(&bytes[8], &ops, 4), std::memcpy(&bytes[12], &size, 4);
    if (size > kArenaBytes) throw LoomError("npu: fused command exceeds its arena");
    if (arenas_.empty() || arenas_.back().used + size > kArenaBytes) {
      arenas_.emplace_back();
      iree_xdna_elf_allocation_record_t req{};
      req.domain = IREE_XDNA_ELF_ALLOCATION_DOMAIN_COMMAND;
      req.byte_length = kArenaBytes;
      req.alignment = kCommandAlign;
      AllocateStorage(req, &arenas_.back().storage, &arenas_.back().map);
    }
    Arena& a = arenas_.back();
    std::memcpy(static_cast<std::uint8_t*>(a.storage.mapping.data) + a.used, bytes.data(), size);
    YAH_AMDF(api_->host_mapping_cache_control(a.map, AMDF_HOST_CACHE_OPERATION_FLUSH, a.used, size),
             "host_mapping_cache_control(fused)");
    amdf_xdna_kernel_command_t c{};
    c.memory = a.storage.memory;
    c.access_ordinal = a.storage.access_ordinal;
    c.byte_offset = a.storage.memory_byte_offset + a.used;
    c.byte_length = size;
    a.used += (size + kCommandAlign - 1) / kCommandAlign * kCommandAlign;
    return fused_[calls] = c;
  }
  std::map<std::vector<Kernel*>, amdf_xdna_kernel_command_t> fused_;
  std::deque<Arena> arenas_;

  struct Job {
    hrx_timeline_point_t after;
    std::vector<Kernel*> kernels;
    std::uint64_t signal;
    std::string tag;
  };

  void Relay() {
    for (;;) {
      Job job;
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [this] { return stop_ || !jobs_.empty(); });
        if (jobs_.empty()) return;
        job = std::move(jobs_.front());
        jobs_.pop_front();
      }
      try {
        LoomCheck(hrx_semaphore_wait(job.after.semaphore, job.after.value, UINT64_MAX), "relay: GPU wait");
        const auto t0 = std::chrono::steady_clock::now();
        std::uint64_t submission = 0;
        auto submit = [&](const amdf_xdna_kernel_command_t* command) {
          amdf_xdna_kernel_queue_submission_info_t si{};
          si.type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_SUBMISSION_INFO;
          si.structure_size = sizeof(si);
          si.command_count = 1;
          si.commands = command;
          YAH_AMDF(xdna_->kernel_queue_submit(queue_, &si, &submission), "kernel_queue_submit");
        };
        // A run of streamed calls goes out as one fused command (FusedCommand): calls overlap inside it, and no
        // command boundary falls while a call's DMA work is in flight (separate commands at that point hung the NPU
        // firmware under DRAM contention).
        std::size_t i = 0;
        while (i < job.kernels.size()) {
          Kernel* k = job.kernels[i];
          // The control-only continuation needs the image's array state resident.
          // The first run of an image, or a run after another image, sets it up.
          const bool resident = resident_ == k->image_key;
          if (!k->stream || !resident) {
            submit(resident ? &k->repeat : &k->first);
            resident_ = k->image_key;
            ++i;
            continue;
          }
          std::size_t j = i;
          while (j < job.kernels.size() && job.kernels[j]->stream && job.kernels[j]->image_key == k->image_key) ++j;
          submit(&FusedCommand(std::vector<Kernel*>(job.kernels.begin() + i, job.kernels.begin() + j)));
          i = j;
        }
        YAH_AMDF(api_->kernel_queue_wait(queue_, submission, AMDF_TIMEOUT_INFINITE, 0), "kernel_queue_wait");
        last_job_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        {
          std::lock_guard<std::mutex> lock(mu_);
          auto& st = stats_[job.tag];
          st.jobs += 1, st.calls += job.kernels.size(), st.busy_ms += last_job_ms_.load();
        }
      } catch (const LoomError& e) {
        std::lock_guard<std::mutex> lock(mu_);
        if (failure_.empty()) failure_ = e.what();
        resident_ = nullptr;
      }
      hrx_status_ignore(hrx_semaphore_signal(done_, job.signal));
    }
  }

  // The memory profile that registers caller-owned host pages for NPU access (CreateShared host).
  uint32_t RegisterProfile() {
    for (uint32_t ord = 0;; ++ord) {
      amdf_memory_profile_t p{};
      p.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE;
      p.structure_size = sizeof(p);
      amdf_memory_access_capabilities_t caps{};
      caps.type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES;
      caps.structure_size = sizeof(caps);
      const amdf_status_t st = api_->memory_scope_query_device_profile(scope_, ord, 1, &access_, &p, &caps);
      if (amdf_status_code(st) == AMDF_STATUS_CODE_OUT_OF_RANGE) throw LoomError("npu: no host registration profile");
      if (st == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) continue;
      YAH_AMDF(st, "memory_scope_query_device_profile(register)");
      if (p.roles & AMDF_MEMORY_PROFILE_ROLE_REGISTER) return p.ordinal;
    }
  }

  void SelectScope() {
    uint32_t count = 0;
    api_->instance_enumerate_memory_scopes(instance_, 0, nullptr, &count);
    std::vector<amdf_memory_scope_t*> scopes(count);
    YAH_AMDF(api_->instance_enumerate_memory_scopes(instance_, count, scopes.data(), &count),
             "instance_enumerate_memory_scopes");
    for (auto* s : scopes) {
      amdf_memory_scope_info_t info{};
      info.type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO;
      info.structure_size = sizeof(info);
      YAH_AMDF(api_->memory_scope_query_info(s, &info), "memory_scope_query_info");
      if (info.kind == AMDF_MEMORY_SCOPE_KIND_SYSTEM) {
        scope_ = s;
        return;
      }
    }
    throw LoomError("npu: no system memory scope");
  }

  void OpenEndpoint() {
    uint32_t count = 0;
    YAH_AMDF(api_->endpoint_enumerate(instance_, 0, nullptr, &count), "endpoint_enumerate(count)");
    std::vector<amdf_endpoint_summary_t> summaries(count);
    YAH_AMDF(api_->endpoint_enumerate(instance_, count, summaries.data(), &count), "endpoint_enumerate");
    for (auto& s : summaries) {
      if (s.engine_kind != AMDF_ENGINE_KIND_XDNA) continue;
      YAH_AMDF(api_->endpoint_open(instance_, &s.id, &endpoint_), "endpoint_open");
      return;
    }
    throw LoomError("npu: no XDNA endpoint");
  }

  void CreateDevice() {
    amdf_xdna_endpoint_info_t xi{};
    xi.type = AMDF_STRUCTURE_TYPE_XDNA_ENDPOINT_INFO;
    xi.structure_size = sizeof(xi);
    YAH_AMDF(xdna_->endpoint_query_info(endpoint_, &xi), "xdna.endpoint_query_info");
    NpuCheck(iree_hal_amd_xdna_aie2p_npu2_target_initialize(iree_make_cstring_view(xi.target_id),
                                                            static_cast<uint16_t>(columns_), &target_),
             "npu2_target_initialize");
    amdf_xdna_device_create_info_t dci{};
    dci.type = AMDF_STRUCTURE_TYPE_XDNA_DEVICE_CREATE_INFO;
    dci.structure_size = sizeof(dci);
    YAH_AMDF(xdna_->device_create(endpoint_, &dci, &device_), "xdna.device_create");
    amdf_xdna_device_info_t di{};
    di.type = AMDF_STRUCTURE_TYPE_XDNA_DEVICE_INFO;
    di.structure_size = sizeof(di);
    YAH_AMDF(xdna_->device_query_info(device_, &di), "xdna.device_query_info");
    target_.instruction_alignment = di.instruction.address_alignment;
    amdf_endpoint_info_t ei{};
    ei.type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO;
    ei.structure_size = sizeof(ei);
    YAH_AMDF(api_->endpoint_query_info(endpoint_, &ei), "endpoint_query_info");
    uint32_t family = UINT32_MAX;
    for (uint32_t i = 0; i < ei.queue_family_count && family == UINT32_MAX; ++i) {
      amdf_queue_family_info_t f{};
      f.type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO;
      f.structure_size = sizeof(f);
      YAH_AMDF(api_->endpoint_query_queue_family_info(endpoint_, i, &f), "endpoint_query_queue_family_info");
      if (f.command_type == AMDF_QUEUE_COMMAND_TYPE_XDNA && (f.publication_modes & AMDF_QUEUE_PUBLICATION_MODE_KERNEL))
        family = i;
    }
    if (family == UINT32_MAX) throw LoomError("npu: no XDNA kernel queue family");
    access_.device = device_;
    access_.requirements.access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
    access_.requirements.flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
    access_.requirements.address_kinds = UINT64_C(1) << AMDF_MEMORY_ADDRESS_XDNA_DMA;
    profile_.ordinal = AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN;
    for (uint32_t ord = 0;; ++ord) {
      amdf_memory_profile_t p{};
      p.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE;
      p.structure_size = sizeof(p);
      amdf_memory_access_capabilities_t caps{};
      caps.type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES;
      caps.structure_size = sizeof(caps);
      const amdf_status_t st = api_->memory_scope_query_device_profile(scope_, ord, 1, &access_, &p, &caps);
      if (amdf_status_code(st) == AMDF_STATUS_CODE_OUT_OF_RANGE) break;
      if (st == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) continue;
      YAH_AMDF(st, "memory_scope_query_device_profile");
      bool dmabuf = false;
      for (uint32_t e = 0; e < p.external_memory_support_count; ++e)
        dmabuf |= p.external_memory_support[e].type == AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD &&
                  (p.external_memory_support[e].flags & AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_IMPORT) &&
                  (p.external_memory_support[e].flags & AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_SOURCE_OFFSET);
      if ((p.roles & AMDF_MEMORY_PROFILE_ROLE_IMPORT) && dmabuf) {
        profile_ = p;
        break;
      }
    }
    if (profile_.ordinal == AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN) throw LoomError("npu: no DMA-BUF import profile");
    amdf_xdna_context_create_info_t cci{};
    cci.type = AMDF_STRUCTURE_TYPE_XDNA_CONTEXT_CREATE_INFO;
    cci.structure_size = sizeof(cci);
    cci.logical_column_count = columns_;
    cci.physical_column_origin = AMDF_XDNA_PHYSICAL_COLUMN_ORIGIN_ANY;
    cci.acceptable_scheduling_modes = AMDF_XDNA_SCHEDULING_MODE_TIME_SLICED;
    YAH_AMDF(xdna_->context_create(device_, &cci, &context_), "xdna.context_create");
    amdf_xdna_kernel_queue_create_info_t qi{};
    qi.type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_CREATE_INFO;
    qi.structure_size = sizeof(qi);
    qi.queue_family_ordinal = family;
    YAH_AMDF(xdna_->kernel_queue_create(context_, &qi, &queue_), "xdna.kernel_queue_create");
  }

  iree_hal_amd_xdna_image_t* Image(const std::string& path) {
    auto it = images_.find(path);
    if (it != images_.end()) return it->second;
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw LoomError("npu: cannot open " + path);
    const auto size = static_cast<iree_host_size_t>(f.tellg());
    f.seekg(0);
    void* data = nullptr;
    NpuCheck(iree_allocator_malloc(iree_allocator_system(), size, &data), "malloc(image)");
    f.read(static_cast<char*>(data), static_cast<std::streamsize>(size));
    iree_byte_sequence_t* seq = nullptr;
    iree_byte_span_t span = iree_make_byte_span(data, size);
    NpuCheck(iree_byte_sequence_create_from_span_move(&span, iree_allocator_system(), &seq), "byte_sequence");
    iree_hal_amd_xdna_image_t* image = nullptr;
    const iree_status_t st = iree_hal_amd_xdna_image_create(seq, &target_, iree_allocator_system(), &image);
    iree_byte_sequence_release(seq);
    NpuCheck(st, ("image_create " + path).c_str());
    images_[path] = image;
    return image;
  }

  void AllocateStorage(const iree_xdna_elf_allocation_record_t& req, iree_hal_amd_xdna_executable_storage_t* out,
                       amdf_host_mapping_t** out_map) {
    const bool command = req.domain == IREE_XDNA_ELF_ALLOCATION_DOMAIN_COMMAND;
    amdf_memory_scope_t* scope = scope_;
    if (command) {
      uint32_t count = 0;
      YAH_AMDF(xdna_->context_enumerate_memory_scopes(context_, 1, &scope, &count),
               "xdna.context_enumerate_memory_scopes");
    }
    const amdf_memory_address_kind_t kind = command ? AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE : AMDF_MEMORY_ADDRESS_XDNA_DMA;
    amdf_memory_device_access_t access{};
    access.device = device_;
    access.requirements.access =
        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE | (command ? AMDF_MEMORY_ACCESS_EXECUTE : 0);
    access.requirements.flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
    access.requirements.address_kinds = UINT64_C(1) << kind;
    amdf_memory_profile_t p{};
    const amdf_memory_profile_roles_t roles = AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP;
    for (uint32_t ord = 0;; ++ord) {
      p = {};
      p.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE;
      p.structure_size = sizeof(p);
      amdf_memory_access_capabilities_t caps{};
      caps.type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES;
      caps.structure_size = sizeof(caps);
      const amdf_status_t st = api_->memory_scope_query_device_profile(scope, ord, 1, &access, &p, &caps);
      if (amdf_status_code(st) == AMDF_STATUS_CODE_OUT_OF_RANGE) throw LoomError("npu: no storage profile");
      if (st == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) continue;
      YAH_AMDF(st, "memory_scope_query_device_profile(storage)");
      if ((p.roles & roles) == roles && (p.supported_flags & AMDF_MEMORY_FLAG_HOST_VISIBLE)) break;
    }
    const std::uint64_t g = p.allocation.byte_length_granularity;
    amdf_memory_create_info_t ci{};
    ci.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
    ci.structure_size = sizeof(ci);
    ci.memory_profile_ordinal = p.ordinal;
    ci.access_count = 1;
    ci.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
    ci.byte_length = (req.byte_length + g - 1) / g * g;
    ci.minimum_alignment = p.allocation.minimum_alignment;
    ci.accesses = &access;
    YAH_AMDF(api_->memory_create(scope, &ci, &out->memory), "memory_create(storage)");
    YAH_AMDF(api_->memory_query_address(out->memory, 0, kind, &out->device_address), "memory_query_address(storage)");
    amdf_memory_map_info_t mi{};
    mi.type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO;
    mi.structure_size = sizeof(mi);
    mi.byte_length = req.byte_length;
    mi.flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
    YAH_AMDF(api_->memory_map(out->memory, &mi, out_map), "memory_map(storage)");
    amdf_host_mapping_info_t info{};
    info.type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO;
    info.structure_size = sizeof(info);
    YAH_AMDF(api_->host_mapping_query_info(*out_map, &info), "host_mapping_query_info(storage)");
    out->mapping = iree_make_byte_span(info.pointer, static_cast<iree_host_size_t>(req.byte_length));
  }

  LoomDevice& gpu_;
  std::uint32_t columns_;
  const amdf_api_t* api_ = nullptr;
  const amdf_xdna_api_t* xdna_ = nullptr;
  amdf_instance_t* instance_ = nullptr;
  amdf_memory_scope_t* scope_ = nullptr;
  amdf_endpoint_t* endpoint_ = nullptr;
  amdf_device_t* device_ = nullptr;
  amdf_xdna_context_t* context_ = nullptr;
  amdf_kernel_queue_t* queue_ = nullptr;
  amdf_memory_device_access_t access_{};
  amdf_memory_profile_t profile_{};
  using ExportFn = int (*)(const void*, std::size_t, int*, std::uint64_t*);
  ExportFn export_dmabuf_ = nullptr;
  iree_hal_amd_xdna_aie2p_target_t target_{};
  std::map<std::string, iree_hal_amd_xdna_image_t*> images_;
  std::vector<std::unique_ptr<Shared>> shared_;
  std::vector<std::unique_ptr<Kernel>> kernels_;
  hrx_semaphore_t done_ = nullptr;
  LoomBuffer fence_host_, fence_dev_;  // Enqueue / Join cache fences
  const void* resident_ = nullptr;  // relay thread only
  std::atomic<double> last_job_ms_{0};
  std::map<std::string, TagStats> stats_;  // under mu_
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<Job> jobs_;
  std::uint64_t issued_ = 0;
  bool stop_ = false;
  std::string failure_;
  std::thread relay_;
};

// The NPU side of LoomPrefill's column split: the shared A / W / C buffers and the calls bound on views of them.
class LoomNpuSplit : public NpuSplit {
 public:
  LoomNpuSplit(LoomDevice& gpu, const NpuPlan& plan)
      : npu_(gpu, 8),
        a_(npu_.CreateShared(plan.a_bytes)),
        w_(npu_.CreateShared(plan.w_bytes)),
        c_(npu_.CreateShared(plan.c_bytes, true)) {}   // the NPU writes C (see CreateShared)
  const LoomBuffer& A() const override { return a_.gpu; }
  const LoomBuffer& W() const override { return w_.gpu; }
  const LoomBuffer& C() const override { return c_.gpu; }
  std::uint32_t Bind(const std::string& image, NpuView a, NpuView w, NpuView c) override {
    const auto key = std::make_tuple(image, a.offset, a.length, w.offset, w.length, c.offset, c.length);
    if (const auto it = ids_.find(key); it != ids_.end()) return it->second;
    kernels_.push_back(&npu_.Load(image, "npu_gemm",
                                  {{&a_, a.offset, a.length}, {&w_, w.offset, w.length}, {&c_, c.offset, c.length}}));
    return ids_[key] = static_cast<std::uint32_t>(kernels_.size() - 1);
  }
  std::uint64_t Enqueue(const std::vector<std::uint32_t>& calls, const std::string& tag) override {
    npu_.CheckHealth();
    std::vector<LoomNpu::Kernel*> k;
    for (const std::uint32_t id : calls) k.push_back(kernels_.at(id));
    last_tag_ = tag;
    return npu_.Enqueue(std::move(k), tag);
  }
  void Join(std::uint64_t value) override {
    const auto t0 = std::chrono::steady_clock::now();
    npu_.Join(value);
    npu_.AddJoinMs(last_tag_, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    npu_.CheckHealth();
  }
  // One line per job tag: jobs, calls, NPU busy time, and host time spent waiting for the NPU in Join.
  void Report(std::FILE* f) {
    for (const auto& [tag, st] : npu_.Stats())
      std::fprintf(f, "npu: %-5s %4zu jobs %5zu calls, NPU busy %7.1f ms, join wait %6.1f ms\n", tag.c_str(), st.jobs,
                   st.calls, st.busy_ms, st.join_ms);
  }

 private:
  LoomNpu npu_;
  LoomNpu::Shared &a_, &w_, &c_;
  std::vector<LoomNpu::Kernel*> kernels_;
  std::string last_tag_;
  std::map<std::tuple<std::string, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t>,
           std::uint32_t>
      ids_;
};

}  // namespace yah::model

#endif  // YAH_MODEL_LOOM_NPU_HPP_
