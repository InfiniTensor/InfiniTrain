#include "infini_train/src/core/runtime/dcu/dcu_runtime_common.h"

#include "infini_train/include/common/dcu/common_dcu.h"

namespace infini_train::core::dcu {
namespace {
uint32_t ToDcuEventFlags(EventFlag flags) {
    switch (flags) {
    case EventFlag::kDefault:
        return hipEventDefault;
    case EventFlag::kBlockingSync:
        return hipEventBlockingSync;
    case EventFlag::kDisableTiming:
        return hipEventDisableTiming;
    case EventFlag::kInterprocess:
        // HIP requires hipEventDisableTiming with interprocess events.
        return hipEventInterprocess | hipEventDisableTiming;
    default:
        LOG(FATAL) << "Unsupported EventFlag value: " << static_cast<uint32_t>(flags);
    }
    return hipEventDefault;
}
} // namespace

DcuEvent::DcuEvent(EventFlag flags) { HIP_CHECK(hipEventCreateWithFlags(&event_, ToDcuEventFlags(flags))); }

DcuEvent::~DcuEvent() {
    if (event_ != nullptr) {
        HIP_CHECK(hipEventDestroy(event_));
    }
}

hipEvent_t DcuEvent::hip_event() const { return event_; }

DcuStream::DcuStream() { HIP_CHECK(hipStreamCreate(&stream_)); }

DcuStream::DcuStream(int priority) {
    HIP_CHECK(hipStreamCreateWithPriority(&stream_, hipStreamNonBlocking, priority));
}

DcuStream::~DcuStream() {
    // Do nothing.
}

hipStream_t DcuStream::hip_stream() const { return stream_; }

DcuBlasHandle::DcuBlasHandle(Stream *stream) {
    HIPBLAS_CHECK(hipblasCreate(&hipblas_handle_));
    HIPBLAS_CHECK(hipblasSetStream(hipblas_handle_, dynamic_cast<DcuStream *>(stream)->hip_stream()));
}

DcuBlasHandle::~DcuBlasHandle() {
    // Do nothing.
}

hipblasHandle_t DcuBlasHandle::hipblas_handle() const { return hipblas_handle_; }

} // namespace infini_train::core::dcu
