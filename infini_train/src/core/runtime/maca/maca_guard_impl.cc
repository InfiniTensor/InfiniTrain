#include "infini_train/src/core/runtime/maca/maca_guard_impl.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "infini_train/include/common/maca/common_maca.h"
#include "infini_train/include/core/runtime/runtime_common.h"
#include "infini_train/include/device.h"

#include "infini_train/src/core/runtime/maca/maca_runtime_common.h"

namespace infini_train::core::maca {
namespace {
constexpr int kMaxGpus = 8;
constexpr size_t kBytesPerMB = 1024ULL * 1024ULL;

// Read /proc/self/cmdline and return --tensor_parallel value, or 1 if absent /
// unparseable. Must be callable from static init (before main runs), so we
// cannot use gflags here.
int ReadTensorParallelFromCmdline() {
    std::ifstream in("/proc/self/cmdline", std::ios::binary);
    if (!in) {
        return 1;
    }
    std::vector<std::string> args;
    std::string cur;
    char c;
    while (in.get(c)) {
        if (c == '\0') {
            if (!cur.empty()) {
                args.push_back(std::move(cur));
                cur.clear();
            }
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) {
        args.push_back(std::move(cur));
    }
    for (size_t i = 0; i < args.size(); ++i) {
        const auto &a = args[i];
        std::string value;
        if (a.rfind("--tensor_parallel=", 0) == 0) {
            value = a.substr(std::string("--tensor_parallel=").size());
        } else if (a == "--tensor_parallel" && i + 1 < args.size()) {
            value = args[i + 1];
        } else {
            continue;
        }
        try {
            return std::stoi(value);
        } catch (...) { return 1; }
    }
    return 1;
}

static std::array<std::unique_ptr<MacaStream>, kMaxGpus> maca_streams;
static std::array<std::unique_ptr<MacaBlasHandle>, kMaxGpus> maca_blas_handles;

static std::array<std::once_flag, kMaxGpus> device_stream_flags;
static std::array<std::once_flag, kMaxGpus> device_handle_flags;

// Serialize host-side MemcpyAsync across threads. On MACA, concurrent
// mcMemcpyAsync from multiple threads during init-time bursts
// (Module::To uploads, Adam state fills, ...) races with the runtime's
// auto P2P peer-mapping and produces "readonly page" faults or
// mcErrorInvalidValue. The lock is held only for the brief window of the
// API call itself; actual GPU work remains async on the caller's stream.
static std::mutex g_memcpy_mutex;

struct AllocationRecord {
    int device_index = 0;
    size_t size = 0;
};

struct AllocationStats {
    size_t current = 0;
    size_t high = 0;
};

// Temporary peak-memory accounting for the synchronous mcMalloc fallback below.
// CUDA does not need this because cudaMallocAsync allocations are covered by
// the default mempool high-watermark stats. Once MACA mcMallocAsync is safe to
// restore, delete this tracker and rely on mcMemPoolGetAttribute only.
static std::mutex g_allocation_mutex;
static std::unordered_map<void *, AllocationRecord> g_allocations;
static std::array<AllocationStats, kMaxGpus> g_allocation_stats;

struct CachedBlock {
    void *ptr = nullptr;
    size_t requested_size = 0;
    size_t block_size = 0;
    int device_index = -1;
    mcEvent_t ready_event = nullptr;
};

struct DevicePool {
    std::mutex mutex;
    std::multimap<size_t, CachedBlock *> free_blocks;
    size_t cached_bytes = 0;
    size_t reserved_bytes = 0;
    size_t high_reserved_bytes = 0;
};

static std::array<DevicePool, kMaxGpus> g_device_pools;
static std::mutex g_cache_active_mutex;
static std::unordered_map<void *, CachedBlock *> g_cache_active_blocks;

inline bool EnvFlagEnabled(const char *name) {
    const char *value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

inline bool CachingDisabled() {
    static const bool disabled = EnvFlagEnabled("INFINI_MACA_DISABLE_CACHE");
    return disabled;
}

inline size_t CacheMaxBytes() {
    static const size_t max_bytes = [] {
        const char *value = std::getenv("INFINI_MACA_CACHE_MAX_MB");
        if (value == nullptr || value[0] == '\0') {
            return std::numeric_limits<size_t>::max();
        }
        char *end = nullptr;
        const unsigned long long max_mb = std::strtoull(value, &end, 10);
        if (end == value) {
            return std::numeric_limits<size_t>::max();
        }
        constexpr size_t kMaxMB = std::numeric_limits<size_t>::max() / kBytesPerMB;
        return static_cast<size_t>(std::min<unsigned long long>(max_mb, kMaxMB)) * kBytesPerMB;
    }();
    return max_bytes;
}

inline size_t RoundUpToBlock(size_t size) {
    if (size <= 512) {
        return 512;
    }
    if (size <= kBytesPerMB) {
        size_t block_size = 512;
        while (block_size < size) { block_size <<= 1; }
        return block_size;
    }
    constexpr size_t kLargeBlockAlignment = 2 * kBytesPerMB;
    return ((size + kLargeBlockAlignment - 1) / kLargeBlockAlignment) * kLargeBlockAlignment;
}

inline bool CacheBlockReady(CachedBlock *block) {
    if (block->ready_event == nullptr) {
        return true;
    }

    int current_device = -1;
    MACA_CHECK(mcGetDevice(&current_device));
    if (current_device != block->device_index) {
        MACA_CHECK(mcSetDevice(block->device_index));
    }
    const mcError_t status = mcEventQuery(block->ready_event);
    if (current_device != block->device_index) {
        MACA_CHECK(mcSetDevice(current_device));
    }

    if (status == mcSuccess) {
        return true;
    }
    if (status == mcErrorNotReady) {
        return false;
    }
    MACA_CHECK(status);
    return false;
}

static void InsertActiveBlock(CachedBlock *block) {
    std::lock_guard<std::mutex> lock(g_cache_active_mutex);
    auto [_, inserted] = g_cache_active_blocks.emplace(block->ptr, block);
    CHECK(inserted) << "MACA caching allocator active block already exists.";
}

static CachedBlock *TakeActiveBlock(void *ptr) {
    std::lock_guard<std::mutex> lock(g_cache_active_mutex);
    auto it = g_cache_active_blocks.find(ptr);
    if (it == g_cache_active_blocks.end()) {
        return nullptr;
    }
    CachedBlock *block = it->second;
    g_cache_active_blocks.erase(it);
    return block;
}

static CachedBlock *PopFreeBlockCandidate(int device_index, size_t min_block_size) {
    auto &pool = g_device_pools.at(device_index);
    std::lock_guard<std::mutex> lock(pool.mutex);
    auto it = pool.free_blocks.lower_bound(min_block_size);
    if (it == pool.free_blocks.end()) {
        return nullptr;
    }

    CachedBlock *block = it->second;
    CHECK_GE(pool.cached_bytes, block->block_size) << "MACA cache cached bytes underflow.";
    pool.cached_bytes -= block->block_size;
    pool.free_blocks.erase(it);
    return block;
}

static void ReturnFreeBlocksToPool(int device_index, std::vector<CachedBlock *> *blocks) {
    if (blocks->empty()) {
        return;
    }

    auto &pool = g_device_pools.at(device_index);
    std::lock_guard<std::mutex> lock(pool.mutex);
    for (CachedBlock *block : *blocks) {
        pool.free_blocks.emplace(block->block_size, block);
        pool.cached_bytes += block->block_size;
    }
    blocks->clear();
}

static void AccountReleasedBlocks(int device_index, size_t released_bytes) {
    if (released_bytes == 0) {
        return;
    }

    auto &pool = g_device_pools.at(device_index);
    std::lock_guard<std::mutex> lock(pool.mutex);
    CHECK_GE(pool.reserved_bytes, released_bytes) << "MACA cache reserved bytes underflow.";
    pool.reserved_bytes -= released_bytes;
}

static void FreeCachedBlocksOnDevice(int device_index, std::vector<CachedBlock *> blocks) {
    if (blocks.empty()) {
        return;
    }

    int current_device = -1;
    MACA_CHECK(mcGetDevice(&current_device));
    if (current_device != device_index) {
        MACA_CHECK(mcSetDevice(device_index));
    }

    for (CachedBlock *block : blocks) {
        if (block->ready_event != nullptr) {
            MACA_CHECK(mcEventDestroy(block->ready_event));
        }
        MACA_CHECK(mcFree(block->ptr));
        delete block;
    }

    if (current_device != device_index) {
        MACA_CHECK(mcSetDevice(current_device));
    }
}

static size_t ReleaseCachedBlocks(int device_index, size_t target_bytes) {
    std::vector<CachedBlock *> blocks_to_free;
    std::vector<CachedBlock *> skipped_blocks;
    size_t released_bytes = 0;

    while (released_bytes < target_bytes) {
        CachedBlock *block = PopFreeBlockCandidate(device_index, 0);
        if (block == nullptr) {
            break;
        }
        if (!CacheBlockReady(block)) {
            skipped_blocks.push_back(block);
            continue;
        }
        released_bytes += block->block_size;
        blocks_to_free.push_back(block);
    }

    ReturnFreeBlocksToPool(device_index, &skipped_blocks);
    AccountReleasedBlocks(device_index, released_bytes);
    FreeCachedBlocksOnDevice(device_index, std::move(blocks_to_free));
    return released_bytes;
}

static void ReleaseReadyCachedBlocks(int device_index) {
    ReleaseCachedBlocks(device_index, std::numeric_limits<size_t>::max());
}

static void TrimCacheToLimit(int device_index) {
    const size_t max_bytes = CacheMaxBytes();
    if (max_bytes == std::numeric_limits<size_t>::max()) {
        return;
    }

    size_t bytes_to_release = 0;
    {
        auto &pool = g_device_pools.at(device_index);
        std::lock_guard<std::mutex> lock(pool.mutex);
        if (pool.cached_bytes <= max_bytes) {
            return;
        }
        bytes_to_release = pool.cached_bytes - max_bytes;
    }
    ReleaseCachedBlocks(device_index, bytes_to_release);
}

static void CachingMalloc(void **ptr, size_t size, int device_index) {
    if (size == 0) {
        *ptr = nullptr;
        return;
    }

    const size_t block_size = RoundUpToBlock(size);
    TrimCacheToLimit(device_index);

    auto &pool = g_device_pools.at(device_index);
    CachedBlock *block = nullptr;
    std::vector<CachedBlock *> skipped_blocks;

    while (true) {
        CachedBlock *candidate = PopFreeBlockCandidate(device_index, block_size);
        if (candidate == nullptr) {
            break;
        }
        if (!CacheBlockReady(candidate)) {
            skipped_blocks.push_back(candidate);
            continue;
        }
        block = candidate;
        block->requested_size = size;
        break;
    }
    ReturnFreeBlocksToPool(device_index, &skipped_blocks);

    if (block == nullptr) {
        void *dev_ptr = nullptr;
        mcError_t status = mcMalloc(&dev_ptr, block_size);
        if (status == mcErrorMemoryAllocation) {
            ReleaseReadyCachedBlocks(device_index);
            status = mcMalloc(&dev_ptr, block_size);
            if (status == mcErrorMemoryAllocation) {
                for (int i = 0; i < kMaxGpus; ++i) {
                    if (i != device_index) {
                        ReleaseReadyCachedBlocks(i);
                    }
                }
                status = mcMalloc(&dev_ptr, block_size);
            }
        }
        MACA_CHECK(status);

        block = new CachedBlock{dev_ptr, size, block_size, device_index, nullptr};
        {
            std::lock_guard<std::mutex> lock(pool.mutex);
            pool.reserved_bytes += block_size;
            pool.high_reserved_bytes = std::max(pool.high_reserved_bytes, pool.reserved_bytes);
        }
    }

    InsertActiveBlock(block);
    *ptr = block->ptr;
}

static void CachingFree(void *ptr, mcStream_t stream, bool already_ready, int fallback_device_index) {
    if (ptr == nullptr) {
        return;
    }

    CachedBlock *block = TakeActiveBlock(ptr);
    if (block == nullptr) {
        int current_device = -1;
        if (fallback_device_index >= 0) {
            MACA_CHECK(mcGetDevice(&current_device));
            if (current_device != fallback_device_index) {
                MACA_CHECK(mcSetDevice(fallback_device_index));
            }
        }
        MACA_CHECK(mcFree(ptr));
        if (fallback_device_index >= 0 && current_device != fallback_device_index) {
            MACA_CHECK(mcSetDevice(current_device));
        }
        return;
    }

    if (!already_ready) {
        CHECK(stream != nullptr) << "MACA caching allocator async free requires a stream.";
        int current_device = -1;
        MACA_CHECK(mcGetDevice(&current_device));
        if (current_device != block->device_index) {
            MACA_CHECK(mcSetDevice(block->device_index));
        }
        if (block->ready_event == nullptr) {
            MACA_CHECK(mcEventCreateWithFlags(&block->ready_event, mcEventDisableTiming));
        }
        MACA_CHECK(mcEventRecord(block->ready_event, stream));
        if (current_device != block->device_index) {
            MACA_CHECK(mcSetDevice(current_device));
        }
    }

    auto &pool = g_device_pools.at(block->device_index);
    {
        std::lock_guard<std::mutex> lock(pool.mutex);
        pool.free_blocks.emplace(block->block_size, block);
        pool.cached_bytes += block->block_size;
    }
    TrimCacheToLimit(block->device_index);
}

static void TrackAllocation(void *ptr, int device_index, size_t size) {
    if (ptr == nullptr) {
        return;
    }

    CHECK(device_index >= 0 && device_index < kMaxGpus)
        << std::format("MACA current device index {} out of allocation stats range [0, {}).", device_index, kMaxGpus);

    std::lock_guard<std::mutex> lock(g_allocation_mutex);
    g_allocations[ptr] = AllocationRecord{device_index, size};
    auto &stats = g_allocation_stats.at(device_index);
    stats.current += size;
    stats.high = std::max(stats.high, stats.current);
}

static AllocationRecord UntrackAllocation(void *ptr) {
    AllocationRecord record{-1, 0};
    if (ptr == nullptr) {
        return record;
    }

    std::lock_guard<std::mutex> lock(g_allocation_mutex);
    auto it = g_allocations.find(ptr);
    if (it == g_allocations.end()) {
        return record;
    }

    record = it->second;
    auto &stats = g_allocation_stats.at(it->second.device_index);
    CHECK_GE(stats.current, it->second.size) << "MACA allocation tracker underflow.";
    stats.current -= it->second.size;
    g_allocations.erase(it);
    return record;
}

inline void CheckMacaDevice(Device device) {
    CHECK(device.type() == Device::DeviceType::kMACA) << std::format(
        "MacaGuardImpl expects MACA device, but got type={} index={}", static_cast<int>(device.type()), device.index());
    const int idx = device.index();
    CHECK(idx >= 0 && idx < kMaxGpus) << std::format("MACA device index {} out of cache range [0, {}).", idx, kMaxGpus);
}

inline mcEvent_t GetMacaEvent(Event *event) {
    auto *maca_event = dynamic_cast<MacaEvent *>(event);
    CHECK_NOTNULL(maca_event);
    return maca_event->maca_event();
}

inline mcStream_t GetMacaStream(Stream *stream) {
    auto *maca_stream = dynamic_cast<MacaStream *>(stream);
    CHECK_NOTNULL(maca_stream);
    return maca_stream->maca_stream();
}
} // namespace

void MacaGuardImpl::InitSingleStream(Device device) {
    CheckMacaDevice(device);

    int current_device = -1;
    MACA_CHECK(mcGetDevice(&current_device));
    MACA_CHECK(mcSetDevice(device.index()));

    maca_streams[device.index()] = std::make_unique<MacaStream>();

    MACA_CHECK(mcSetDevice(current_device));
}

void MacaGuardImpl::InitSingleHandle(Device device) {
    CheckMacaDevice(device);

    int current_device = -1;
    MACA_CHECK(mcGetDevice(&current_device));
    MACA_CHECK(mcSetDevice(device.index()));

    std::call_once(device_stream_flags.at(device.index()), InitSingleStream, device);

    maca_blas_handles[device.index()] = std::make_unique<MacaBlasHandle>(maca_streams[device.index()].get());

    MACA_CHECK(mcSetDevice(current_device));
}

MacaGuardImpl::MacaGuardImpl() {
    // Force synchronous kernel launches on MACA before initializing the runtime.
    // Multi-thread DDP races MCCL P2P buffer setup against concurrent user-tensor
    // kernel launches; without launch-blocking, threads crash during init or
    // step 0 with "Writing to readonly page" / xnack ATU faults on 64MB P2P
    // buffers.  setenv() from main() is too late because mcInit(0) runs during
    // static initialization (before main), so we setenv here in the ctor
    // just prior to mcInit(0).  Users can override by setting the env var
    // themselves before launch.
    setenv("MACA_LAUNCH_BLOCKING", "1", 0);

    // When TP > 1 on MACA, disable the MCCL-level P2P path to prevent multi-PG
    // init deadlocks (threads concurrently creating both DP and TP comms hang
    // in mcclCommInitAll). Must be set before mcInit(0); setenv from main() is
    // too late because this ctor runs at static init. Peek at /proc/self/cmdline
    // to keep single-card / DP-only / PP-only runs on the P2P fast path.
    if (ReadTensorParallelFromCmdline() > 1) {
        setenv("MCCL_P2P_DISABLE", "1", 0);
    }

    // The MACA runtime requires an explicit mcInit(0) before any other call.
    // CUDA has no equivalent; mirroring the DeviceManager ctor from 87390cd.
    MACA_CHECK(mcInit(0));
}

// device
Device MacaGuardImpl::GetDevice() const {
    int current_device = -1;
    MACA_CHECK(mcGetDevice(&current_device));
    return Device(Device::DeviceType::kMACA, current_device);
}

void MacaGuardImpl::SetDevice(Device device) const {
    CheckMacaDevice(device);
    MACA_CHECK(mcSetDevice(device.index()));
}

int MacaGuardImpl::DeviceCount() const {
    int device_count = 0;
    MACA_CHECK(mcGetDeviceCount(&device_count));
    return device_count;
}

Device::DeviceType MacaGuardImpl::Type() const { return Device::DeviceType::kMACA; }

// stream
Stream *MacaGuardImpl::GetStream(Device device) const {
    CheckMacaDevice(device);
    // FIXME(dcj): call_once is process-scoped and assumes single initialization.
    std::call_once(device_stream_flags.at(device.index()), InitSingleStream, device);
    return maca_streams.at(device.index()).get();
}

Stream *MacaGuardImpl::CreateStream(Device device) const {
    CheckMacaDevice(device);
    int current_device = -1;
    MACA_CHECK(mcGetDevice(&current_device));
    MACA_CHECK(mcSetDevice(device.index()));

    Stream *stream = new MacaStream();

    MACA_CHECK(mcSetDevice(current_device));
    return stream;
}

Stream *MacaGuardImpl::CreateStreamWithPriority(Device device, int priority) const {
    CheckMacaDevice(device);
    int current_device = -1;
    MACA_CHECK(mcGetDevice(&current_device));
    MACA_CHECK(mcSetDevice(device.index()));

    Stream *stream = new MacaStream(priority);

    MACA_CHECK(mcSetDevice(current_device));
    return stream;
}

void MacaGuardImpl::DestroyStream(Stream *stream) const {
    if (stream == nullptr) {
        return;
    }
    auto *maca_stream = dynamic_cast<MacaStream *>(stream);
    CHECK_NOTNULL(maca_stream);
    delete maca_stream;
}

void MacaGuardImpl::GetStreamPriorityRange(int *low, int *high) const {
    MACA_CHECK(mcDeviceGetStreamPriorityRange(low, high));
}

// event
void MacaGuardImpl::EventCreate(Event **event) const { *event = new MacaEvent(); }

void MacaGuardImpl::EventCreateWithFlags(Event **event, EventFlag flags) const { *event = new MacaEvent(flags); }

void MacaGuardImpl::EventDestroy(Event *event) const {
    if (event == nullptr) {
        return;
    }
    delete event;
}

void MacaGuardImpl::EventRecord(Event *event, Stream *stream) const {
    auto maca_event = GetMacaEvent(event);
    auto maca_stream = GetMacaStream(stream);
    MACA_CHECK(mcEventRecord(maca_event, maca_stream));
}

void MacaGuardImpl::StreamWaitEvent(Stream *stream, Event *event, uint32_t flags) const {
    auto maca_event = GetMacaEvent(event);
    auto maca_stream = GetMacaStream(stream);
    MACA_CHECK(mcStreamWaitEvent(maca_stream, maca_event, flags));
}

RuntimeStatus MacaGuardImpl::EventSynchronize(Event *event) const {
    auto maca_event = GetMacaEvent(event);
    mcError_t status = mcEventSynchronize(maca_event);
    if (status == mcSuccess) {
        return RuntimeStatus::kSuccess;
    }
    if (status == mcErrorNotReady) {
        return RuntimeStatus::kNotReady;
    }
    LOG(ERROR) << "MacaGuardImpl::EventSynchronize failed: " << mcGetErrorString(status);
    return RuntimeStatus::kError;
}

RuntimeStatus MacaGuardImpl::EventQuery(Event *event) const {
    auto maca_event = GetMacaEvent(event);
    mcError_t status = mcEventQuery(maca_event);
    if (status == mcSuccess) {
        return RuntimeStatus::kSuccess;
    }
    if (status == mcErrorNotReady) {
        return RuntimeStatus::kNotReady;
    }
    LOG(ERROR) << "MacaGuardImpl::EventQuery failed: " << mcGetErrorString(status);
    return RuntimeStatus::kError;
}

float MacaGuardImpl::EventElapsedTime(Event *start_event, Event *stop_event) const {
    auto start_maca_event = GetMacaEvent(start_event);
    auto stop_maca_event = GetMacaEvent(stop_event);
    float elapsed_ms = 0.0f;
    MACA_CHECK(mcEventElapsedTime(&elapsed_ms, start_maca_event, stop_maca_event));
    return elapsed_ms;
}

// sync
void MacaGuardImpl::SynchronizeDevice(Device device) const {
    auto original_device = GetDevice();
    SetDevice(device);

    MACA_CHECK(mcDeviceSynchronize());

    SetDevice(original_device);
}

void MacaGuardImpl::SynchronizeStream(Stream *stream) const {
    auto maca_stream = GetMacaStream(stream);
    MACA_CHECK(mcStreamSynchronize(maca_stream));
}

// blas
BlasHandle *MacaGuardImpl::GetBlasHandle(Device device) const {
    CheckMacaDevice(device);
    std::call_once(device_handle_flags.at(device.index()), InitSingleHandle, device);
    return maca_blas_handles.at(device.index()).get();
}

// memory
void MacaGuardImpl::Malloc(void **dev_ptr, size_t size) {
    int current_device = -1;
    MACA_CHECK(mcGetDevice(&current_device));
    CHECK(current_device >= 0 && current_device < kMaxGpus)
        << std::format("MACA current device index {} out of allocation stats range [0, {}).", current_device, kMaxGpus);

    if (CachingDisabled()) {
        MACA_CHECK(mcMalloc(dev_ptr, size));
    } else {
        CachingMalloc(dev_ptr, size, current_device);
    }
    TrackAllocation(*dev_ptr, current_device, size);
}

void MacaGuardImpl::MallocAsync(void **dev_ptr, size_t size, Stream *stream) {
    // NOTE(dcj): mcMallocAsync with a per-stream mempool gives a big speedup
    // (~2x on gpt2 DDP steady-state) vs synchronous mcMalloc, but under
    // multi-thread DDP init bursts (e.g. llama3 1B with nthread=8 uploading
    // hundreds of param tensors) it races with MACA's auto P2P peer-mapping
    // and produces mcErrorInvalidValue on subsequent mcMemcpyAsync, or
    // "readonly page" faults — no amount of mutex/stream-sync serialization
    // around the alloc call suppresses this. The caching allocator below keeps
    // the safe synchronous driver path but avoids repeated steady-state calls.
    // TODO(dcj): after MACA fixes mcMallocAsync correctness under multi-thread
    // runtime calls, reopen the async path:
    // auto maca_stream = GetMacaStream(stream);
    // MACA_CHECK(mcMallocAsync(dev_ptr, size, maca_stream));
    (void)stream;
    Malloc(dev_ptr, size);
}

void MacaGuardImpl::Free(void *dev_ptr) {
    const AllocationRecord record = UntrackAllocation(dev_ptr);
    if (CachingDisabled()) {
        MACA_CHECK(mcFree(dev_ptr));
        return;
    }

    if (dev_ptr != nullptr && record.device_index >= 0) {
        int current_device = -1;
        MACA_CHECK(mcGetDevice(&current_device));
        if (current_device != record.device_index) {
            MACA_CHECK(mcSetDevice(record.device_index));
        }
        MACA_CHECK(mcDeviceSynchronize());
        if (current_device != record.device_index) {
            MACA_CHECK(mcSetDevice(current_device));
        }
    }
    CachingFree(dev_ptr, nullptr, /*already_ready=*/true, record.device_index);
}

void MacaGuardImpl::FreeAsync(void *dev_ptr, Stream *stream) {
    // auto maca_stream = GetMacaStream(stream);
    // MACA_CHECK(mcFreeAsync(dev_ptr, maca_stream));
    const AllocationRecord record = UntrackAllocation(dev_ptr);
    if (CachingDisabled()) {
        MACA_CHECK(mcFree(dev_ptr));
        return;
    }

    auto maca_stream = dev_ptr == nullptr ? nullptr : GetMacaStream(stream);
    CachingFree(dev_ptr, maca_stream, /*already_ready=*/false, record.device_index);
}

void MacaGuardImpl::Memcpy(void *dst, const void *src, size_t count, MemcpyKind kind) {
    if (kind == MemcpyKind::kH2D) {
        MACA_CHECK(mcMemcpy(dst, src, count, mcMemcpyHostToDevice));
    } else if (kind == MemcpyKind::kD2H) {
        MACA_CHECK(mcMemcpy(dst, src, count, mcMemcpyDeviceToHost));
    } else if (kind == MemcpyKind::kD2D) {
        MACA_CHECK(mcMemcpy(dst, src, count, mcMemcpyDeviceToDevice));
    } else {
        LOG(FATAL) << std::format("MacaGuardImpl::Memcpy got invalid MemcpyKind={}", MemcpyKindToString(kind));
    }
}

void MacaGuardImpl::MemcpyAsync(void *dst, const void *src, size_t count, MemcpyKind kind, Stream *stream) {
    std::lock_guard<std::mutex> lock(g_memcpy_mutex);
    auto maca_stream = GetMacaStream(stream);

    switch (kind) {
    case MemcpyKind::kH2D:
        MACA_CHECK(mcMemcpyAsync(dst, src, count, mcMemcpyHostToDevice, maca_stream));
        break;
    case MemcpyKind::kD2H:
        MACA_CHECK(mcMemcpyAsync(dst, src, count, mcMemcpyDeviceToHost, maca_stream));
        break;
    case MemcpyKind::kD2D:
        MACA_CHECK(mcMemcpyAsync(dst, src, count, mcMemcpyDeviceToDevice, maca_stream));
        break;
    default:
        LOG(FATAL) << std::format("MacaGuardImpl::MemcpyAsync got invalid MemcpyKind={}", MemcpyKindToString(kind));
    }
}

void MacaGuardImpl::ResetMemPoolHighWatermarks(Device device) const {
    CheckMacaDevice(device);

    int current_device = -1;
    MACA_CHECK(mcGetDevice(&current_device));

    SetDevice(device);
    mcMemPool_t pool;
    MACA_CHECK(mcDeviceGetDefaultMemPool(&pool, device.index()));

    uint64_t zero = 0;
    // High watermark can only be reset to zero; non-zero is illegal.
    MACA_CHECK(mcMemPoolSetAttribute(pool, mcMemPoolAttrUsedMemHigh, &zero));
    MACA_CHECK(mcMemPoolSetAttribute(pool, mcMemPoolAttrReservedMemHigh, &zero));

    {
        std::lock_guard<std::mutex> lock(g_allocation_mutex);
        auto &stats = g_allocation_stats.at(device.index());
        stats.high = stats.current;
    }
    {
        auto &pool_stats = g_device_pools.at(device.index());
        std::lock_guard<std::mutex> lock(pool_stats.mutex);
        pool_stats.high_reserved_bytes = pool_stats.reserved_bytes;
    }

    MACA_CHECK(mcSetDevice(current_device));
}

std::pair<size_t, size_t> MacaGuardImpl::GetMemPoolPeakMB(Device device) const {
    CheckMacaDevice(device);

    int current_device = -1;
    MACA_CHECK(mcGetDevice(&current_device));

    SetDevice(device);
    mcMemPool_t pool;
    MACA_CHECK(mcDeviceGetDefaultMemPool(&pool, device.index()));

    uint64_t used = 0;
    MACA_CHECK(mcMemPoolGetAttribute(pool, mcMemPoolAttrUsedMemHigh, &used));

    uint64_t reserved = 0;
    MACA_CHECK(mcMemPoolGetAttribute(pool, mcMemPoolAttrReservedMemHigh, &reserved));

    size_t tracked_high = 0;
    {
        std::lock_guard<std::mutex> lock(g_allocation_mutex);
        tracked_high = g_allocation_stats.at(device.index()).high;
    }
    size_t reserved_high = tracked_high;
    {
        auto &pool_stats = g_device_pools.at(device.index());
        std::lock_guard<std::mutex> lock(pool_stats.mutex);
        reserved_high = std::max(reserved_high, pool_stats.high_reserved_bytes);
    }

    MACA_CHECK(mcSetDevice(current_device));

    const size_t used_peak = tracked_high + static_cast<size_t>(used);
    const size_t reserved_peak = reserved_high + static_cast<size_t>(reserved);
    return std::make_pair<size_t, size_t>(static_cast<size_t>(used_peak / kBytesPerMB),
                                          static_cast<size_t>(reserved_peak / kBytesPerMB));
}

INFINI_TRAIN_REGISTER_DEVICE_GUARD_IMPL(Device::DeviceType::kMACA, MacaGuardImpl)

} // namespace infini_train::core::maca
