// NPU (XDNA2) GEMMs beside the GPU: libamdf for the device, the HRX xdna loader for .xdna images.
// Operands live in HRX device buffers (full GPU bandwidth), exported once as a DMA-BUF through HSA and imported per binding view into the NPU.
// Gated images only (gen_npu_gemm GATE): the host submits a chunk's jobs at once.
// The NPU waits for each job's ready word itself, and a reaper thread retires the commands (NpuSplit: the protocol).
// Needs HRX patches 0013 / 0014 (see engine/hrx/patches).
#ifndef YAH_MODEL_LOOM_NPU_HPP_
#define YAH_MODEL_LOOM_NPU_HPP_

#include <dlfcn.h>
#include <immintrin.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
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
    bool external = false;             // RegisterHost: the caller's pages (no HRX import, not unmapped)
  };
  struct View {
    Shared* shared;
    std::size_t offset, length;
  };
  // One loaded instance of an image with its bindings and native command storage.
  struct Kernel {
    const void* image_key = nullptr;
    iree_hal_amd_xdna_image_t* image = nullptr;
    uint32_t entry_ordinal = 0;
    // Decoder-column images (gen_npu_gemm dcol 2) share all array state but their decoder tiles: family names the set
    // of such images (nullptr: the image alone), dec the decoder format, swap the image whose setup re-programs only
    // the decoder tiles (LOOM_EXP_SETUP_TILES; spliced into a command where the format changes).
    const void* family = nullptr;
    int dec = -1;
    const Kernel* swap = nullptr;
    // a second instance of the image whose first (the array setup and one ungated call, the state the gate expects
    // before its first job) writes a scratch C; spliced in where a job's family differs from the array's
    // (FusedCommand); nullptr: none
    const Kernel* setup = nullptr;
    amdf_xdna_kernel_command_t first{};
    // first sets the array up and runs one whole call (any data). gate opens a job once its ready word reaches the
    // job's gate value; done writes the done word after the job's last C (gen_npu_gemm GATE).
    // Streamed (seven invocations): push[parity] queues a call and wait retires the oldest queued one; lead is the even
    // push with its weight fills unpaced, for the first call of a job (nothing computes yet that pacing would
    // protect). Not streamed (four invocations, one replay group, HRX patch 0017): call runs one whole call.
    bool streamed = true;
    amdf_xdna_kernel_command_t push[2]{}, wait{}, lead{}, gate{}, done{}, call{};
    std::vector<iree_hal_amd_xdna_executable_storage_t> storage;
    std::vector<amdf_host_mapping_t*> storage_maps;
    std::vector<iree_hal_buffer_t*> buffers;
  };
  static const void* FamilyOf(const Kernel* k) { return k->family ? k->family : k->image_key; }
  // One call: its kernel instance's invocation bytes as bound for it (an instance serves many calls: Rebind).
  struct Call {
    const Kernel* k = nullptr;
    std::vector<std::uint8_t> push[2], wait, lead, gate, done, call;
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
    reaper_ = std::thread([this] { Reaper(); });
  }
  LoomNpu(const LoomNpu&) = delete;
  LoomNpu& operator=(const LoomNpu&) = delete;
  ~LoomNpu() {
    // no GPU work may still read the shared buffers (flag words, C) once they are freed: on this GPU a read of freed
    // memory hangs the shader and the GPU's scheduler
    try {
      gpu_.Synchronize();
    } catch (...) {
    }
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
    }
    reap_cv_.notify_all();
    if (reaper_.joinable()) reaper_.join();
    // the array first: after its last job a gated image's head keeps polling its flag words (and writing its tick
    // scratch) for the rest of its supply, and streamed fills stay armed; with the IOMMU in passthrough a page freed
    // under a live context is written by the NPU after the kernel reused it (the GPU wedged after runs, 2026-10-09)
    if (queue_) api_->kernel_queue_destroy(queue_), queue_ = nullptr;
    if (context_) xdna_->context_destroy(context_), context_ = nullptr;
    for (auto& a : arenas_) {
      api_->host_mapping_destroy(a.map);
      api_->memory_destroy(a.storage.memory);
    }
    for (auto& k : kernels_) {
      for (auto* m : k->storage_maps) api_->host_mapping_destroy(m);
      for (auto& s : k->storage) api_->memory_destroy(s.memory);
      for (auto* b : k->buffers) iree_hal_buffer_release(b);
    }
    for (auto& [key, m] : imports_) api_->memory_destroy(m);
    for (auto& [key, image] : images_) iree_hal_amd_xdna_image_destroy(image);
    for (auto& s : shared_) {
      if (s->external) {
        api_->memory_destroy(s->native);
      } else if (s->host) {   // the HRX import first, then the registration, then the pages
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
  // The caller's page-aligned host pages (e.g. raw GGUF weight rows) registered for NPU reads; they stay the caller's.
  // libamdf pins them for writing, so they must be writable (a private file mapping gets copies of the pages it pins).
  Shared& RegisterHost(void* pages, std::size_t bytes) {
    auto s = std::make_unique<Shared>();
    s->bytes = bytes;
    s->external = true;
    amdf_memory_create_info_t ci{};
    ci.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
    ci.structure_size = sizeof(ci);
    ci.memory_profile_ordinal = RegisterProfile();
    ci.access_count = 1;
    ci.accesses = &access_;
    ci.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
    ci.byte_length = bytes;
    ci.minimum_alignment = 4096;
    ci.registered_host_pointer = pages;
    ci.registered_host_cacheability = AMDF_HOST_CACHEABILITY_WRITE_BACK;
    YAH_AMDF(api_->memory_create(scope_, &ci, &s->native), "memory_create(registered weights)");
    s->device = pages;
    shared_.push_back(std::move(s));
    return *shared_.back();
  }

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
    k->image = image;
    k->entry_ordinal = entry_ordinal;
    const std::vector<iree_hal_amd_xdna_executable_binding_t> bindings = Bindings(*k, views);
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
    if (rec.invocation_count != 7 && rec.invocation_count != 4)
      throw LoomError("npu: " + path + " has no gate (re-emit it)");
    k->streamed = rec.invocation_count == 7;
    amdf_xdna_kernel_command_t* const streamed[] = {&k->push[0], &k->push[1], &k->wait, &k->lead, &k->gate, &k->done};
    amdf_xdna_kernel_command_t* const single[] = {&k->call, &k->gate, &k->done};
    for (uint32_t i = 0; i + 1 < rec.invocation_count; ++i)
      NpuCheck(iree_hal_amd_xdna_executable_query_invocation_ordinal(image, entry_ordinal, n, k->storage.data(), i + 1,
                                                                     k->streamed ? streamed[i] : single[i]),
               "query_invocation_ordinal");
    kernels_.push_back(std::move(k));
    return *kernels_.back();
  }

  // The executable bindings of views for instance k (k.buffers keeps the range wrappers). DMA-BUF views are imported
  // once per view; registered host pages bind at the view's offset.
  std::vector<iree_hal_amd_xdna_executable_binding_t> Bindings(Kernel& k, const std::vector<View>& views) {
    const iree_hal_amd_xdna_image_tables_t* tables = iree_hal_amd_xdna_image_tables(k.image);
    const iree_xdna_elf_entry_record_t rec = iree_hal_amd_xdna_image_tables_entry(tables, k.entry_ordinal);
    if (rec.binding_count != views.size()) throw LoomError("npu: binding count");
    std::vector<iree_hal_amd_xdna_executable_binding_t> bindings(views.size());
    for (std::size_t i = 0; i < views.size(); ++i) {
      const View& v = views[i];
      if (v.offset + v.length > v.shared->bytes) throw LoomError("npu: bad view");
      const iree_xdna_elf_binding_record_t contract =
          iree_hal_amd_xdna_image_tables_binding(tables, rec.first_binding + static_cast<uint32_t>(i));
      amdf_memory_t* mem = v.shared->native;
      std::uint64_t addr = 0, mem_off = 0;
      if (mem) {   // registered host pages: bound at the view's offset
        mem_off = v.offset;
      } else if (const auto it = imports_.find({v.shared, v.offset, v.length}); it != imports_.end()) {
        mem = it->second;
      } else {
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
        YAH_AMDF(api_->memory_import(scope_, &ii, &em, &mem), "memory_import");
        imports_[{v.shared, v.offset, v.length}] = mem;
      }
      YAH_AMDF(api_->memory_query_address(mem, 0, AMDF_MEMORY_ADDRESS_XDNA_DMA, &addr), "memory_query_address");
      addr += mem_off;
      // executable_bind checks this wrapper's range and access only; the span is never dereferenced (the address is
      // device_address). Wrappers start 64-byte aligned: raw weight rows need not.
      std::uint8_t* span = static_cast<std::uint8_t*>(v.shared->device) + v.offset;
      const std::size_t lead = reinterpret_cast<std::uintptr_t>(span) & 63;
      iree_hal_buffer_t* buffer = nullptr;
      NpuCheck(iree_hal_heap_buffer_wrap(
                   iree_hal_buffer_placement_undefined(),
                   IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
                       IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
                   IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE, IREE_HAL_BUFFER_USAGE_STORAGE,
                   v.length + lead, iree_make_byte_span(span - lead, v.length + lead),
                   iree_hal_buffer_release_callback_null(), iree_allocator_system(), &buffer),
               "heap_buffer_wrap");
      k.buffers.push_back(buffer);
      bindings[i].buffer_ref = iree_hal_make_buffer_ref(buffer, 0, v.length + lead);
      bindings[i].memory = mem;
      bindings[i].memory_byte_offset = mem_off;
      bindings[i].device_address = addr;
    }
    return bindings;
  }
  // Bind instance k to other views (one instance of an image serves many calls: its invocation bytes are copied per
  // call, Snapshot).
  void Rebind(Kernel& k, const std::vector<View>& views) {
    const std::vector<iree_hal_amd_xdna_executable_binding_t> bindings = Bindings(k, views);
    const auto n = static_cast<iree_host_size_t>(k.storage.size());
    NpuCheck(iree_hal_amd_xdna_executable_bind(k.image, k.entry_ordinal, n, k.storage.data(), bindings.size(),
                                               bindings.data()),
             "executable_bind");
    for (uint32_t i = 0; i < n; ++i)
      YAH_AMDF(api_->host_mapping_cache_control(k.storage_maps[i], AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
                                                k.storage[i].mapping.data_length),
               "host_mapping_cache_control(storage)");
  }
  // The invocation bytes of k as bound now.
  Call Snapshot(const Kernel& k) const {
    Call c;
    c.k = &k;
    auto copy = [&](const amdf_xdna_kernel_command_t& cmd, std::vector<std::uint8_t>& out) {
      if (!cmd.byte_length) return;
      const std::uint8_t* src = CommandBytes(k, cmd);
      out.assign(src, src + cmd.byte_length);
    };
    copy(k.push[0], c.push[0]), copy(k.push[1], c.push[1]), copy(k.wait, c.wait), copy(k.lead, c.lead);
    copy(k.gate, c.gate), copy(k.done, c.done), copy(k.call, c.call);
    return c;
  }

  // A gated image's job (NpuSplit::Enqueue): its calls, done word and gate value; the NPU waits for its ready word
  // itself and writes the gate value to *done.
  struct GatedJob {
    std::vector<const Call*> calls;
    volatile std::uint32_t* done = nullptr;
    std::uint32_t gate = 0;
    std::string tag;
  };
  // Gated jobs as one command, submitted now (no host between the jobs). The first job of an image first runs its setup
  // as its own command (one ungated call, any data; the host waits for it once), after which the gate waits for job 1.
  // After a failure the reaper stores gate | kGateFailed to the done words of this and every later command's jobs, so
  // the GPU's waits end and report it.
  // fresh: set the first job's image up even if its family is resident (a chunk's first command: the gate's job
  // sequence restarts there, NpuSplit::BeginChunk).
  void EnqueueGated(std::vector<GatedJob> jobs, bool fresh = false) {
    CheckHealth();
    const auto t0 = std::chrono::steady_clock::now();
    const Kernel* k = jobs[0].calls[0]->k;
    const void* family = k->family ? k->family : k->image_key;
    bool resident = false;
    int dec = -1;
    {
      std::lock_guard<std::mutex> lock(mu_);
      resident = resident_ == family;
      dec = resident_dec_;
    }
    if ((!resident || fresh) && k->setup) {
      // FusedCommand splices the setup twin in front of the first job
      std::lock_guard<std::mutex> lock(mu_);
      family = nullptr;
    } else if (!resident) {
      YAH_AMDF(api_->kernel_queue_wait(queue_, Submit(k->first), AMDF_TIMEOUT_INFINITE, 0), "kernel_queue_wait(setup)");
      std::lock_guard<std::mutex> lock(mu_);
      resident_ = family;
      dec = resident_dec_ = k->dec;
    }
    std::vector<std::vector<const Call*>> runs;
    for (const GatedJob& g : jobs) runs.push_back(g.calls);
    // commands run in submission order: the next one starts with this one's last family and decoder
    const Fused fused = FusedCommand(runs, dec, family);
    const amdf_xdna_kernel_command_t command = fused.command;
    {
      std::lock_guard<std::mutex> lock(mu_);
      resident_dec_ = fused.dec;
      resident_ = fused.family;
    }
    const std::uint64_t submission = Submit(command);
    {
      std::lock_guard<std::mutex> lock(mu_);
      pending_.push_back({submission, std::move(jobs), t0});
      ++submitted_;
    }
    reap_cv_.notify_one();
  }
  // Waits until the reaper retired every submitted command (its stats and failures are in).
  void Drain() {
    std::unique_lock<std::mutex> lock(mu_);
    retired_cv_.wait(lock, [this] { return retired_ == submitted_; });
  }
  // Per job tag: jobs, calls.
  struct TagStats {
    std::size_t jobs = 0, calls = 0;
  };
  std::map<std::string, TagStats> Stats() {
    std::lock_guard<std::mutex> lock(mu_);
    return stats_;
  }
  // Commands: count, NPU time (each from the later of its submission and the previous command's completion, its
  // waits for the GPU included) and the longest (the driver kills a command after amdxdna tdr_timeout_ms, 2000 ms).
  struct CommandStats {
    std::size_t commands = 0;
    double busy_ms = 0, max_ms = 0;
  };
  CommandStats Commands() {
    std::lock_guard<std::mutex> lock(mu_);
    return commands_;
  }
  // NPU time of the last retired command (CommandStats).
  [[nodiscard]] double LastCommandMs() const { return last_command_ms_.load(); }
  // Throws if a command failed.
  void CheckHealth() {
    std::lock_guard<std::mutex> lock(mu_);
    if (!failure_.empty()) throw LoomError(failure_);
  }

 private:
  // Gated jobs as one command, each [gate c0][lead c0][push c1][wait][push c2][wait] ... [wait][wait][done cN], or for
  // an image that is not streamed [gate c0][call c0] ... [call cN][done cN].
  // Pushes alternate descriptor sets, at most two calls queued: calls overlap, and no command boundary falls while a
  // call's DMA work is in flight (that hung the NPU firmware under DRAM contention). The bodies of the calls' own
  // relocated commands go under one transaction header (libamdf's public format 0.1,
  // AMDF_XDNA_TRANSACTION_FORMAT_VERSION_0_1: operation count at byte 8, byte length at 12).
  // Built per command into a ring of command arenas (allocating per command cost ~1 ms); an arena is reused once its
  // commands retired (family switches splice whole array setups: commands of several MB, and the NPU's command memory
  // is small).
  struct Arena {
    iree_hal_amd_xdna_executable_storage_t storage{};
    amdf_host_mapping_t* map = nullptr;
    std::size_t used = 0;
    std::uint64_t last = 0;   // the last command built into it (its submission count)
  };
  static constexpr std::size_t kArenaBytes = 8u << 20, kCommandAlign = 32768, kArenas = 3;
  static const std::uint8_t* CommandBytes(const Kernel& k, const amdf_xdna_kernel_command_t& c) {
    for (const auto& st : k.storage)
      if (st.memory == c.memory) return static_cast<const std::uint8_t*>(st.mapping.data) + (c.byte_offset - st.memory_byte_offset);
    throw LoomError("npu: command outside its kernel storage");
  }
  // Decoder-column images: where a call's decoder format differs from the one set up, the queued calls drain, the
  // format's swap setup follows and the call starts like a job's first (lead; the parity restarts there). Where a job's
  // family differs from the one set up (the previous job has drained by its done), its image's setup twin runs first.
  // Returns the command and the family and decoder format set up at its end.
  struct Fused {
    amdf_xdna_kernel_command_t command{};
    int dec = -1;
    const void* family = nullptr;
  };
  Fused FusedCommand(const std::vector<std::vector<const Call*>>& runs, int dec, const void* fam) {
    std::vector<std::uint8_t> bytes;
    std::uint32_t ops = 0;
    auto append_raw = [&](const std::uint8_t* src, std::size_t length) {
      std::uint32_t n = 0, size = 0;
      std::memcpy(&n, src + 8, 4), std::memcpy(&size, src + 12, 4);
      if (size != length) throw LoomError("npu: unexpected command header");
      if (bytes.empty()) bytes.assign(src, src + 16);
      bytes.insert(bytes.end(), src + 16, src + size);
      ops += n;
    };
    auto append = [&](const std::vector<std::uint8_t>& b) { append_raw(b.data(), b.size()); };
    for (const auto& calls : runs) {
      int queued = 0, since = 0;
      if (const Kernel* k0 = calls[0]->k; FamilyOf(k0) != fam) {
        if (!k0->setup) throw LoomError("npu: a job switches image families without a setup twin");
        append_raw(CommandBytes(*k0->setup, k0->setup->first), k0->setup->first.byte_length);
        fam = FamilyOf(k0), dec = k0->dec;
      }
      append(calls[0]->gate);
      if (!calls[0]->k->streamed) {
        for (const Call* c : calls) append(c->call);
        append(calls.back()->done);
        continue;
      }
      for (std::size_t i = 0; i < calls.size(); ++i) {
        const Call& c = *calls[i];
        if (c.k->swap && c.k->dec != dec) {
          for (; queued > 0; --queued) append(calls[i - 1]->wait);
          append_raw(CommandBytes(*c.k->swap, c.k->swap->first), c.k->swap->first.byte_length);
          dec = c.k->dec, since = 0;
        }
        append(since == 0 ? c.lead : c.push[since & 1]);
        ++since;
        if (++queued == 2) append(c.wait), --queued;
      }
      for (; queued > 0; --queued) append(calls.back()->wait);
      append(calls.back()->done);
    }
    const std::uint32_t size = static_cast<std::uint32_t>(bytes.size());
    std::memcpy(&bytes[8], &ops, 4), std::memcpy(&bytes[12], &size, 4);
    if (size > kArenaBytes) throw LoomError("npu: fused command exceeds its arena");
    if (arenas_.empty() || arenas_[arena_].used + size > kArenaBytes) {
      if (arenas_.size() < kArenas) {
        arenas_.emplace_back();
        iree_xdna_elf_allocation_record_t req{};
        req.domain = IREE_XDNA_ELF_ALLOCATION_DOMAIN_COMMAND;
        req.byte_length = kArenaBytes;
        req.alignment = kCommandAlign;
        AllocateStorage(req, &arenas_.back().storage, &arenas_.back().map);
        arena_ = arenas_.size() - 1;
      } else {   // the next arena, once the commands in it retired
        arena_ = (arena_ + 1) % kArenas;
        std::unique_lock<std::mutex> lock(mu_);
        retired_cv_.wait(lock, [&] { return retired_ >= arenas_[arena_].last; });
        arenas_[arena_].used = 0;
      }
    }
    Arena& a = arenas_[arena_];
    {
      std::lock_guard<std::mutex> lock(mu_);
      a.last = submitted_ + 1;   // the command about to be submitted (EnqueueGated)
    }
    std::memcpy(static_cast<std::uint8_t*>(a.storage.mapping.data) + a.used, bytes.data(), size);
    YAH_AMDF(api_->host_mapping_cache_control(a.map, AMDF_HOST_CACHE_OPERATION_FLUSH, a.used, size),
             "host_mapping_cache_control(fused)");
    amdf_xdna_kernel_command_t c{};
    c.memory = a.storage.memory;
    c.access_ordinal = a.storage.access_ordinal;
    c.byte_offset = a.storage.memory_byte_offset + a.used;
    c.byte_length = size;
    a.used += (size + kCommandAlign - 1) / kCommandAlign * kCommandAlign;
    return {c, dec, fam};
  }
  std::deque<Arena> arenas_;
  std::size_t arena_ = 0;   // the arena being filled

  static constexpr std::uint32_t kGateFailed = NpuSplit::kGateFailed;
  static void Fail(const std::vector<GatedJob>& jobs) {
    for (const GatedJob& g : jobs) __atomic_store_n(g.done, g.gate | kGateFailed, __ATOMIC_RELEASE);
  }
  std::uint64_t Submit(const amdf_xdna_kernel_command_t& command) {
    amdf_xdna_kernel_queue_submission_info_t si{};
    si.type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_SUBMISSION_INFO;
    si.structure_size = sizeof(si);
    si.command_count = 1;
    si.commands = &command;
    std::uint64_t submission = 0;
    YAH_AMDF(xdna_->kernel_queue_submit(queue_, &si, &submission), "kernel_queue_submit");
    return submission;
  }

  // Retires gated commands in order (EnqueueGated): stats.
  // On a failure it stores gate | kGateFailed to the done words of this and every later command's jobs, so no GPU wait
  // is left spinning.
  struct Pending {
    std::uint64_t submission;
    std::vector<GatedJob> jobs;
    std::chrono::steady_clock::time_point t0;
  };
  std::deque<Pending> pending_;
  std::chrono::steady_clock::time_point last_retired_{};  // reaper thread only
  std::condition_variable reap_cv_;
  std::thread reaper_;
  void Reaper() {
    for (;;) {
      Pending p;
      {
        std::unique_lock<std::mutex> lock(mu_);
        reap_cv_.wait(lock, [this] { return stop_ || !pending_.empty(); });
        if (pending_.empty()) return;
        p = std::move(pending_.front());
        pending_.pop_front();
      }
      bool ok = true;
      try {
        YAH_AMDF(api_->kernel_queue_wait(queue_, p.submission, AMDF_TIMEOUT_INFINITE, 0), "kernel_queue_wait(gated)");
      } catch (const LoomError& e) {
        ok = false;
        std::lock_guard<std::mutex> lock(mu_);
        if (failure_.empty()) failure_ = e.what();
        resident_ = nullptr;
        resident_dec_ = -1;
      }
      const auto now = std::chrono::steady_clock::now();
      const double ms = std::chrono::duration<double, std::milli>(now - std::max(p.t0, last_retired_)).count();
      last_retired_ = now;
      {
        std::lock_guard<std::mutex> lock(mu_);
        if (!failure_.empty()) ok = false;
        if (ok) {
          for (const GatedJob& g : p.jobs) {
            auto& st = stats_[g.tag];
            st.jobs += 1, st.calls += g.calls.size();
          }
          commands_.commands += 1, commands_.busy_ms += ms, commands_.max_ms = std::max(commands_.max_ms, ms);
          last_command_ms_ = ms;
        }
      }
      if (!ok) Fail(p.jobs);
      {
        std::lock_guard<std::mutex> lock(mu_);
        ++retired_;
      }
      retired_cv_.notify_all();
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
  const void* resident_ = nullptr;  // the image (family) whose array state is set up; under mu_
  int resident_dec_ = -1;            // its decoder format at the end of the last submitted command; under mu_
  std::map<std::tuple<const Shared*, std::size_t, std::size_t>, amdf_memory_t*> imports_;   // DMA-BUF views
  std::atomic<double> last_command_ms_{0};
  std::map<std::string, TagStats> stats_;  // under mu_
  CommandStats commands_;                  // under mu_
  std::mutex mu_;
  std::uint64_t submitted_ = 0, retired_ = 0;  // commands; under mu_
  std::condition_variable retired_cv_;
  bool stop_ = false;
  std::string failure_;
};

// The NPU side of LoomPrefill's column split: the shared A / W / C buffers, the flag words and the calls bound on them.
class LoomNpuSplit : public NpuSplit {
 public:
  LoomNpuSplit(LoomDevice& gpu, const NpuPlan& plan)
      : npu_(gpu, 8),
        a_(npu_.CreateShared(plan.a_bytes)),
        w_(npu_.CreateShared(plan.w_bytes)),
        c_(npu_.CreateShared(plan.c_bytes, true)),  // the NPU writes C (see CreateShared)
        flags_(npu_.CreateShared(plan.dcol ? kDcolFlagBytes : 65536, true)),
        cscratch_(npu_.CreateShared(plan.dcol ? plan.c_bytes : 4096, true)),
        plan_(plan) {
    if (!plan.gate_calls) throw LoomError("npu: the set has no gate protocol (dispatch.txt npugate; re-emit it)");
    if (!GpuPinnedHigh())
      throw LoomError("npu: pin the GPU performance level first (echo high > /sys/class/drm/card*/device/"
                      "power_dpm_force_performance_level): fabric / memory clock switches corrupt NPU outputs");
    if (kSignal + 2 * plan.gate_record > kGateSlot) throw LoomError("npu: the gate records do not fit a gate slot");
  }
  const LoomBuffer& A() const override { return a_.gpu; }
  const LoomBuffer& W() const override { return w_.gpu; }
  const LoomBuffer& C() const override { return c_.gpu; }
  std::uint32_t Bind(const std::string& image, NpuView a, NpuView w, NpuView c) override {
    const auto key = std::make_tuple(image, a.offset, a.length, w.offset, w.length, c.offset, c.length);
    if (const auto it = ids_.find(key); it != ids_.end()) return it->second;
    // the call's gate slot: flag record, tick scratch, signal (go, done)
    const std::size_t slot = GateSlot();
    calls_.push_back(npu_.Snapshot(npu_.Load(image, "npu_gemm",
                                             {{&a_, a.offset, a.length},
                                              {&w_, w.offset, w.length},
                                              {&c_, c.offset, c.length},
                                              {&flags_, slot, plan_.gate_record},
                                              {&flags_, slot + kTick, plan_.gate_record},
                                              {&flags_, slot + kSignal, 2 * plan_.gate_record}})));
    return ids_[key] = static_cast<std::uint32_t>(calls_.size() - 1);
  }
  void RegisterRaw(const void* p, std::size_t bytes) override {
    const auto b = reinterpret_cast<std::uintptr_t>(p) & ~std::uintptr_t{4095};
    const auto e = (reinterpret_cast<std::uintptr_t>(p) + bytes + 4095) & ~std::uintptr_t{4095};
    for (const Raw& r : raw_)
      if (r.begin <= b && e <= r.end) return;
    // libamdf pins for writing: the private mapping gets its own copies of these pages (the same bytes; the page cache
    // ones stay reclaimable, and the GPU's userptr import follows the new pages). The NPU does not snoop the CPU's
    // caches, so the copies are flushed to memory.
    auto* pages = reinterpret_cast<void*>(b);
    if (mprotect(pages, e - b, PROT_READ | PROT_WRITE)) throw LoomError("npu: mprotect(weights)");
    LoomNpu::Shared& s = npu_.RegisterHost(pages, e - b);
    for (std::uintptr_t a = b; a < e; a += 64) _mm_clflushopt(reinterpret_cast<void*>(a));
    _mm_sfence();
    mprotect(pages, e - b, PROT_READ);
    raw_.push_back({b, e, &s});
  }
  std::uint32_t BindRaw(const std::string& image, const std::string& swap, int dec, NpuView a, const void* raw,
                        std::size_t raw_bytes, NpuView c, const std::string& family) override {
    const auto r0 = reinterpret_cast<std::uintptr_t>(raw);
    const auto key = std::make_tuple(image, a.offset + (a.in_c ? std::size_t{1} << 62 : 0), a.length, std::size_t{r0},
                                     raw_bytes, c.offset, c.length);
    if (const auto it = ids_.find(key); it != ids_.end()) return it->second;
    const Raw* r = nullptr;
    for (const Raw& x : raw_)
      if (x.begin <= r0 && r0 + raw_bytes <= x.end) r = &x;
    if (!r) throw LoomError("npu: raw weights outside the registered ranges");
    const std::size_t slot = GateSlot();
    // bindings: A, the panel (not read), C, the gate's flag / tick / signal, the fill sink (not written), raw rows
    const std::vector<LoomNpu::View> views = {{a.in_c ? &c_ : &a_, a.offset, a.length},
                                              {&w_, 0, plan_.w_bytes},
                                              {&c_, c.offset, c.length},
                                              {&flags_, slot, plan_.gate_record},
                                              {&flags_, slot + kTick, plan_.gate_record},
                                              {&flags_, slot + kSignal, 2 * plan_.gate_record},
                                              {&w_, 0, plan_.w_bytes},
                                              {r->shared, r0 - r->begin, raw_bytes}};
    // one instance per image, re-bound per call (an instance's storage is ~0.7 MB, mostly its array setup)
    LoomNpu::Kernel*& k = instances_[image];
    if (!k) {
      k = &npu_.Load(image, "npu_gemm", views);
      LoomNpu::Kernel*& sw = instances_[swap];
      if (!sw) sw = &npu_.Load(swap, "npu_gemm", views);   // only its setup runs
      k->family = &families_[family], k->dec = dec, k->swap = sw;
      std::vector<LoomNpu::View> scratch = views;   // the setup instance: its call writes the scratch C
      scratch[2] = {&cscratch_, 0, c.length};
      k->setup = &npu_.Load(image, "npu_gemm", scratch);
    } else {
      npu_.Rebind(*k, views);
    }
    calls_.push_back(npu_.Snapshot(*k));
    return ids_[key] = static_cast<std::uint32_t>(calls_.size() - 1);
  }
  // A flag word from / to the host, through memory (the NPU does not snoop CPU caches).
  void HostStore(std::uint32_t word, std::uint32_t value) override {
    auto* w = static_cast<volatile std::uint32_t*>(flags_.host) + word;
    __atomic_store_n(w, value, __ATOMIC_RELEASE);
    __builtin_ia32_clflush(const_cast<std::uint32_t*>(w));
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
  }
  std::uint32_t HostLoad(std::uint32_t word) override {
    auto* w = static_cast<volatile std::uint32_t*>(flags_.host) + word;
    __builtin_ia32_clflush(const_cast<std::uint32_t*>(w));
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    return __atomic_load_n(w, __ATOMIC_ACQUIRE);
  }
  void CheckHealthNow() { npu_.CheckHealth(); }
  // NPU time of the last command, after every submitted one retired.
  double LastCommandMs() {
    npu_.Drain();
    return npu_.LastCommandMs();
  }
  [[nodiscard]] const LoomBuffer* Flags() const override { return &flags_.gpu; }
  [[nodiscard]] std::uint32_t Word(std::uint32_t word) const override {
    return __atomic_load_n(static_cast<const std::uint32_t*>(flags_.host) + word, __ATOMIC_ACQUIRE);
  }
  Job NewJob(const std::vector<std::uint32_t>& calls) override {
    if (calls.empty() || calls.size() >= plan_.gate_calls) throw LoomError("npu: too many calls for one NPU job");
    Job j;
    j.ready = static_cast<std::uint32_t>((kGateBase + kGateSlot * calls.front()) / 4);
    j.done = static_cast<std::uint32_t>((kGateBase + kGateSlot * calls.back() + kSignal + plan_.gate_record) / 4);
    // decoder-column sets: the gate counts jobs from 1 after each image setup, which precedes every family switch and
    // every chunk's first job (BeginChunk, FusedCommand)
    if (plan_.dcol) {
      const void* fam = LoomNpu::FamilyOf(calls_.at(calls.front()).k);
      if (fam != job_family_) job_family_ = fam, seq_ = 0;
    }
    j.gate = ++seq_ * plan_.gate_calls + static_cast<std::uint32_t>(calls.size());
    return j;
  }
  void BeginChunk() override {
    if (!plan_.dcol) return;
    // the previous chunk's jobs are done (their words are reused); then every gate word reads zero again
    npu_.Drain();
    auto* w = static_cast<volatile std::uint32_t*>(flags_.host);
    for (std::size_t b = kGateBase; b < flags_.bytes; b += 4) w[b / 4] = 0;
    for (std::size_t b = kGateBase; b < flags_.bytes; b += 64)
      __builtin_ia32_clflush(const_cast<std::uint32_t*>(w + b / 4));
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    job_family_ = nullptr, fresh_ = true;
  }
  void Enqueue(const std::vector<Queued>& jobs) override {
    std::size_t queued = 0;
    try {
      npu_.CheckHealth();
      if (Word(kFlagStatus)) throw LoomError("npu: a GPU wait for the NPU failed");
      while (queued < jobs.size()) {
        std::vector<LoomNpu::GatedJob> command;
        std::size_t calls = 0;
        const std::size_t per = plan_.dcol ? kDcolJobsPerCommand : kJobsPerCommand;
        for (std::size_t i = queued + command.size(); i < jobs.size() && command.size() < per; ++i) {
          if (!command.empty() && calls + jobs[i].calls.size() > kCallsPerCommand) break;
          LoomNpu::GatedJob g;
          for (const std::uint32_t id : jobs[i].calls) g.calls.push_back(&calls_.at(id));
          g.done = static_cast<volatile std::uint32_t*>(flags_.host) + jobs[i].words.done;
          g.gate = jobs[i].words.gate;
          g.tag = jobs[i].tag;
          calls += jobs[i].calls.size();
          command.push_back(std::move(g));
        }
        const std::size_t n = command.size();
        npu_.EnqueueGated(std::move(command), fresh_);
        fresh_ = false;
        queued += n;
      }
    } catch (...) {
      for (std::size_t i = queued; i < jobs.size(); ++i)
        HostStore(jobs[i].words.done, jobs[i].words.gate | kGateFailed);
      throw;
    }
  }
  // One line per job tag (jobs, calls), then the commands (LoomNpu::CommandStats).
  void Report(std::FILE* f) {
    npu_.Drain();
    if (Word(kFlagStatus)) std::fprintf(f, "npu: ERROR a GPU wait for the NPU failed\n");
    for (const auto& [tag, st] : npu_.Stats())
      std::fprintf(f, "npu: %-5s %4zu jobs %5zu calls\n", tag.c_str(), st.jobs, st.calls);
    const LoomNpu::CommandStats c = npu_.Commands();
    std::fprintf(f, "npu: %zu commands, NPU %.1f ms (waits for the GPU included), longest %.1f ms\n", c.commands,
                 c.busy_ms, c.max_ms);
  }

 private:
  // The amdgpu card's DPM performance level is "high" (results.md: deferred data-fabric errors otherwise).
  static bool GpuPinnedHigh() {
    for (int card = 0; card < 8; ++card) {
      std::ifstream f("/sys/class/drm/card" + std::to_string(card) + "/device/power_dpm_force_performance_level");
      std::string level;
      if (f >> level) return level == "high";
    }
    return false;
  }
  // Gate slots, one per bound call from byte kGateBase of the flag words (below it: kFlagStatus), every view 64-byte
  // aligned: flag record (+0: ready), tick scratch (+kTick), signal (+kSignal: go, then done one record later).
  static constexpr std::size_t kGateBase = 256, kGateSlot = 192, kTick = 64, kSignal = 128;
  // Jobs per NPU command: few commands, while the longest (its waits for the GPU included) stays far below the driver's
  // 2000 ms command limit.
  static constexpr std::size_t kJobsPerCommand = 16, kCallsPerCommand = 256;
  // decoder-column sets: layers without NPU work leave longer waits between jobs (up to the gate's ~400 ms supply)
  static constexpr std::size_t kDcolJobsPerCommand = 8;
  // Decoder-column sets bind a few thousand calls (a gate slot each).
  static constexpr std::size_t kDcolFlagBytes = 2u << 20;
  std::size_t GateSlot() const {
    const std::size_t slot = kGateBase + kGateSlot * calls_.size();
    if (slot + kGateSlot > flags_.bytes) throw LoomError("npu: more NPU calls than gate slots");
    return slot;
  }
  struct Raw {
    std::uintptr_t begin, end;
    LoomNpu::Shared* shared;
  };
  LoomNpu npu_;
  LoomNpu::Shared &a_, &w_, &c_, &flags_;
  LoomNpu::Shared& cscratch_;   // the setup instances' C (BindRaw)
  NpuPlan plan_;
  std::uint32_t seq_ = 0;
  const void* job_family_ = nullptr;   // NewJob's current family run (decoder-column sets)
  bool fresh_ = false;                 // the next command starts a chunk (BeginChunk)
  std::deque<LoomNpu::Call> calls_;   // stable addresses (LoomNpu::GatedJob)
  std::vector<Raw> raw_;
  std::map<std::string, LoomNpu::Kernel*> instances_;
  std::map<std::string, char> families_;   // the decoder-column image families (BindRaw), by name
  std::map<std::tuple<std::string, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t>,
           std::uint32_t>
      ids_;
};

}  // namespace yah::model

#endif  // YAH_MODEL_LOOM_NPU_HPP_
