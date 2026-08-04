#include "infini_train/src/core/runtime/dcu/dcu_guard_impl.h"

#include <array>
#include <memory>
#include <mutex>

#include "infini_train/include/common/dcu/common_dcu.h"
#include "infini_train/include/core/runtime/runtime_common.h"
#include "infini_train/include/device.h"

#include "infini_train/src/core/runtime/dcu/dcu_runtime_common.h"

namespace infini_train::core::dcu {
namespace {
constexpr int kMaxGpus = 8;
constexpr size_t kBytesPerMB = 1024ULL * 1024ULL;

static std::array<std::unique_ptr<DcuStream>, kMaxGpus> hip_streams;
static std::array<std::unique_ptr<DcuBlasHandle>, kMaxGpus> hip_blas_handles;

static std::array<std::once_flag, kMaxGpus> device_stream_flags;
static std::array<std::once_flag, kMaxGpus> device_handle_flags;

inline void CheckDcuDevice(Device device) {
    CHECK(device.type() == Device::DeviceType::kDCU)
        << "DcuGuardImpl expects HIP device, but got type=" << static_cast<int>(device.type())
        << " index=" << static_cast<int>(device.index());
    const int idx = device.index();
    CHECK(idx >= 0 && idx < kMaxGpus) << "HIP device index " << idx << " out of cache range [0, " << kMaxGpus << ").";
}

inline hipEvent_t GetDcuEvent(Event *event) {
    auto *hip_event = dynamic_cast<DcuEvent *>(event);
    CHECK_NOTNULL(hip_event);
    return hip_event->hip_event();
}

inline hipStream_t GetDcuStream(Stream *stream) {
    auto *hip_stream = dynamic_cast<DcuStream *>(stream);
    CHECK_NOTNULL(hip_stream);
    return hip_stream->hip_stream();
}
} // namespace

void DcuGuardImpl::InitSingleStream(Device device) {
    CheckDcuDevice(device);

    int current_device = -1;
    HIP_CHECK(hipGetDevice(&current_device));
    HIP_CHECK(hipSetDevice(device.index()));

    hip_streams[device.index()] = std::make_unique<DcuStream>();

    HIP_CHECK(hipSetDevice(current_device));
}

void DcuGuardImpl::InitSingleHandle(Device device) {
    CheckDcuDevice(device);

    int current_device = -1;
    HIP_CHECK(hipGetDevice(&current_device));
    HIP_CHECK(hipSetDevice(device.index()));

    std::call_once(device_stream_flags.at(device.index()), InitSingleStream, device);

    hip_blas_handles[device.index()] = std::make_unique<DcuBlasHandle>(hip_streams[device.index()].get());

    HIP_CHECK(hipSetDevice(current_device));
}

DcuGuardImpl::DcuGuardImpl() {}

// device
Device DcuGuardImpl::GetDevice() const {
    int current_device = -1;
    HIP_CHECK(hipGetDevice(&current_device));
    return Device(Device::DeviceType::kDCU, current_device);
}

void DcuGuardImpl::SetDevice(Device device) const {
    CheckDcuDevice(device);
    HIP_CHECK(hipSetDevice(device.index()));
}

int DcuGuardImpl::DeviceCount() const {
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    return device_count;
}

Device::DeviceType DcuGuardImpl::Type() const { return Device::DeviceType::kDCU; }

// stream
Stream *DcuGuardImpl::GetStream(Device device) const {
    CheckDcuDevice(device);
    // FIXME(dcj): call_once is process-scoped and assumes single initialization.
    // This can be problematic if the HIP backend is initialized multiple
    // times within the same process (e.g. in unit tests).
    std::call_once(device_stream_flags.at(device.index()), InitSingleStream, device);
    return hip_streams.at(device.index()).get();
}

Stream *DcuGuardImpl::CreateStream(Device device) const {
    CheckDcuDevice(device);
    int current_device = -1;
    HIP_CHECK(hipGetDevice(&current_device));
    HIP_CHECK(hipSetDevice(device.index()));

    Stream *stream = new DcuStream();

    HIP_CHECK(hipSetDevice(current_device));
    return stream;
}

Stream *DcuGuardImpl::CreateStreamWithPriority(Device device, int priority) const {
    CheckDcuDevice(device);
    int current_device = -1;
    HIP_CHECK(hipGetDevice(&current_device));
    HIP_CHECK(hipSetDevice(device.index()));

    Stream *stream = new DcuStream(priority);

    HIP_CHECK(hipSetDevice(current_device));
    return stream;
}

void DcuGuardImpl::DestroyStream(Stream *stream) const {
    if (stream == nullptr) {
        return;
    }
    auto *hip_stream = dynamic_cast<DcuStream *>(stream);
    CHECK_NOTNULL(hip_stream);
    delete hip_stream;
}

void DcuGuardImpl::GetStreamPriorityRange(int *low, int *high) const {
    HIP_CHECK(hipDeviceGetStreamPriorityRange(low, high));
}

// event
void DcuGuardImpl::EventCreate(Event **event) const { *event = new DcuEvent(); }

void DcuGuardImpl::EventCreateWithFlags(Event **event, EventFlag flags) const { *event = new DcuEvent(flags); }

void DcuGuardImpl::EventDestroy(Event *event) const {
    if (event == nullptr) {
        return;
    }
    delete event;
}

void DcuGuardImpl::EventRecord(Event *event, Stream *stream) const {
    auto hip_event = GetDcuEvent(event);
    auto hip_stream = GetDcuStream(stream);
    HIP_CHECK(hipEventRecord(hip_event, hip_stream));
}

void DcuGuardImpl::StreamWaitEvent(Stream *stream, Event *event, uint32_t flags) const {
    auto hip_event = GetDcuEvent(event);
    auto hip_stream = GetDcuStream(stream);
    HIP_CHECK(hipStreamWaitEvent(hip_stream, hip_event, flags));
}

RuntimeStatus DcuGuardImpl::EventSynchronize(Event *event) const {
    auto hip_event = GetDcuEvent(event);
    hipError_t status = hipEventSynchronize(hip_event);
    if (status == hipSuccess) {
        return RuntimeStatus::kSuccess;
    }
    if (status == hipErrorNotReady) {
        return RuntimeStatus::kNotReady;
    }
    LOG(ERROR) << "DcuGuardImpl::EventSynchronize failed: " << hipGetErrorString(status);
    return RuntimeStatus::kError;
}

RuntimeStatus DcuGuardImpl::EventQuery(Event *event) const {
    auto hip_event = GetDcuEvent(event);
    hipError_t status = hipEventQuery(hip_event);
    if (status == hipSuccess) {
        return RuntimeStatus::kSuccess;
    }
    if (status == hipErrorNotReady) {
        return RuntimeStatus::kNotReady;
    }
    LOG(ERROR) << "DcuGuardImpl::EventQuery failed: " << hipGetErrorString(status);
    return RuntimeStatus::kError;
}

float DcuGuardImpl::EventElapsedTime(Event *start_event, Event *stop_event) const {
    auto start_hip_event = GetDcuEvent(start_event);
    auto stop_hip_event = GetDcuEvent(stop_event);
    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start_hip_event, stop_hip_event));
    return elapsed_ms;
}

// sync
void DcuGuardImpl::SynchronizeDevice(Device device) const {
    auto original_device = GetDevice();
    SetDevice(device);

    HIP_CHECK(hipDeviceSynchronize());

    SetDevice(original_device);
}

void DcuGuardImpl::SynchronizeStream(Stream *stream) const {
    auto hip_stream = GetDcuStream(stream);
    HIP_CHECK(hipStreamSynchronize(hip_stream));
}

// blas
BlasHandle *DcuGuardImpl::GetBlasHandle(Device device) const {
    CheckDcuDevice(device);
    std::call_once(device_handle_flags.at(device.index()), InitSingleHandle, device);
    return hip_blas_handles.at(device.index()).get();
}

// memory
void DcuGuardImpl::Malloc(void **dev_ptr, size_t size) { HIP_CHECK(hipMalloc(dev_ptr, size)); }

void DcuGuardImpl::MallocAsync(void **dev_ptr, size_t size, Stream *stream) {
    auto hip_stream = GetDcuStream(stream);
    HIP_CHECK(hipMallocAsync(dev_ptr, size, hip_stream));
}

void DcuGuardImpl::Free(void *dev_ptr) { HIP_CHECK(hipFree(dev_ptr)); }

void DcuGuardImpl::FreeAsync(void *dev_ptr, Stream *stream) {
    auto hip_stream = GetDcuStream(stream);
    HIP_CHECK(hipFreeAsync(dev_ptr, hip_stream));
}

void DcuGuardImpl::Memcpy(void *dst, const void *src, size_t count, MemcpyKind kind) {
    if (kind == MemcpyKind::kH2D) {
        HIP_CHECK(hipMemcpy(dst, src, count, hipMemcpyHostToDevice));
    } else if (kind == MemcpyKind::kD2H) {
        HIP_CHECK(hipMemcpy(dst, src, count, hipMemcpyDeviceToHost));
    } else if (kind == MemcpyKind::kD2D) {
        HIP_CHECK(hipMemcpy(dst, src, count, hipMemcpyDeviceToDevice));
    } else {
        LOG(FATAL) << "DcuGuardImpl::Memcpy got invalid MemcpyKind=" << MemcpyKindToString(kind);
    }
}

void DcuGuardImpl::MemcpyAsync(void *dst, const void *src, size_t count, MemcpyKind kind, Stream *stream) {
    auto hip_stream = GetDcuStream(stream);

    switch (kind) {
    case MemcpyKind::kH2D:
        HIP_CHECK(hipMemcpyAsync(dst, src, count, hipMemcpyHostToDevice, hip_stream));
        break;
    case MemcpyKind::kD2H:
        HIP_CHECK(hipMemcpyAsync(dst, src, count, hipMemcpyDeviceToHost, hip_stream));
        break;
    case MemcpyKind::kD2D:
        HIP_CHECK(hipMemcpyAsync(dst, src, count, hipMemcpyDeviceToDevice, hip_stream));
        break;
    default:
        LOG(FATAL) << "DcuGuardImpl::MemcpyAsync got invalid MemcpyKind=" << MemcpyKindToString(kind);
    }
}

void DcuGuardImpl::ResetMemPoolHighWatermarks(Device device) const {
    int current_device = -1;
    HIP_CHECK(hipGetDevice(&current_device));

    SetDevice(device);
    hipMemPool_t pool;
    HIP_CHECK(hipDeviceGetDefaultMemPool(&pool, device.index()));

    uint64_t zero = 0;
    // High watermark can only be reset to zero; non-zero is illegal.
    HIP_CHECK(hipMemPoolSetAttribute(pool, hipMemPoolAttrUsedMemHigh, &zero));
    HIP_CHECK(hipMemPoolSetAttribute(pool, hipMemPoolAttrReservedMemHigh, &zero));

    HIP_CHECK(hipSetDevice(current_device));
}

std::pair<size_t, size_t> DcuGuardImpl::GetMemPoolPeakMB(Device device) const {
    int current_device = -1;
    HIP_CHECK(hipGetDevice(&current_device));

    SetDevice(device);
    hipMemPool_t pool;
    HIP_CHECK(hipDeviceGetDefaultMemPool(&pool, device.index()));

    uint64_t used = 0;
    HIP_CHECK(hipMemPoolGetAttribute(pool, hipMemPoolAttrUsedMemHigh, &used));

    uint64_t reserved = 0;
    HIP_CHECK(hipMemPoolGetAttribute(pool, hipMemPoolAttrReservedMemHigh, &reserved));

    HIP_CHECK(hipSetDevice(current_device));

    return std::make_pair<size_t, size_t>(static_cast<size_t>(used / kBytesPerMB),
                                          static_cast<size_t>(reserved / kBytesPerMB));
}

INFINI_TRAIN_REGISTER_DEVICE_GUARD_IMPL(Device::DeviceType::kDCU, DcuGuardImpl)

} // namespace infini_train::core::dcu
