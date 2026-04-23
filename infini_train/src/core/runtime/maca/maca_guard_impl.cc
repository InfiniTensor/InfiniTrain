#include "infini_train/src/core/runtime/maca/maca_guard_impl.h"

#include <array>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "infini_train/include/common/maca/common_maca.h"
#include "infini_train/include/core/runtime/runtime_common.h"
#include "infini_train/include/device.h"

#include "infini_train/src/core/runtime/maca/maca_runtime_common.h"

namespace infini_train::core::maca {
namespace {
constexpr int kMaxGpus = 8;

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

// Serialize host-side allocations across threads.  The MACA runtime/MCCL share
// a process-wide virtual address pool; concurrent mcMalloc on multiple threads
// can race with MCCL P2P buffer registration and produce "Writing to readonly
// page" faults on peer-mapped buffers.
static std::mutex g_malloc_mutex;

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

    // When TP > 1 on MACA, disable both the MACA runtime P2P mapping and the
    // MCCL-level P2P path to prevent multi-PG init deadlocks (threads
    // concurrently creating both DP and TP comms hang in mcclCommInitAll).
    // MACA_P2P_DISABLE alone is not sufficient for TP+SP / TP+SP+PP+VPP
    // configurations — MCCL still establishes its own P2P buffers during init,
    // so we must disable that too. Both must be set before mcInit(0); setenv
    // from main() is too late because this ctor runs at static init. Peek at
    // /proc/self/cmdline to keep single-card / DP-only / PP-only runs on the
    // P2P fast path.
    if (ReadTensorParallelFromCmdline() > 1) {
        setenv("MACA_P2P_DISABLE", "1", 0);
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
void MacaGuardImpl::Malloc(void **dev_ptr, size_t size) { MACA_CHECK(mcMalloc(dev_ptr, size)); }

void MacaGuardImpl::MallocAsync(void **dev_ptr, size_t size, Stream *stream) {
    // NOTE(dcj): mcMallocAsync uses a per-stream mempool on MACA and races with
    // MCCL P2P buffer management under multi-thread DDP.  Use the synchronous
    // mcMalloc path (serialized by g_malloc_mutex) so every buffer has a stable
    // mapping by the time any kernel or MCCL op touches it.
    Malloc(dev_ptr, size);
}

void MacaGuardImpl::Free(void *dev_ptr) { MACA_CHECK(mcFree(dev_ptr)); }

void MacaGuardImpl::FreeAsync(void *dev_ptr, Stream *stream) {
    // auto maca_stream = GetMacaStream(stream);
    // MACA_CHECK(mcFreeAsync(dev_ptr, maca_stream));
    Free(dev_ptr);
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
    std::lock_guard<std::mutex> lock(g_malloc_mutex);
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
    // TODO(dcj): MetaX SDK support for mcMemPoolGetAttribute / mcMemPoolAttrUsedMemHigh
    // is not confirmed. Keep this a no-op until verified against a working SDK.
    (void)device;
}

std::pair<size_t, size_t> MacaGuardImpl::GetMemPoolPeakMB(Device device) const {
    // TODO(dcj): see note in ResetMemPoolHighWatermarks.
    (void)device;
    return std::make_pair<size_t, size_t>(0, 0);
}

INFINI_TRAIN_REGISTER_DEVICE_GUARD_IMPL(Device::DeviceType::kMACA, MacaGuardImpl)

} // namespace infini_train::core::maca
