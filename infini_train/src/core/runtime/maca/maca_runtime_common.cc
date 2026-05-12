#include "infini_train/src/core/runtime/maca/maca_runtime_common.h"

#include "infini_train/include/common/maca/common_maca.h"

namespace infini_train::core::maca {
namespace {
uint32_t ToMacaEventFlags(EventFlag flags) {
    switch (flags) {
    case EventFlag::kDefault:
        return mcEventDefault;
    case EventFlag::kBlockingSync:
        return mcEventBlockingSync;
    case EventFlag::kDisableTiming:
        return mcEventDisableTiming;
    case EventFlag::kInterprocess:
        // MACA (like CUDA) requires DisableTiming with Interprocess events.
        // NOTE(dcj): if the MACA SDK in use does not expose mcEventInterprocess,
        // this branch will need to be guarded and downgraded to a LOG(FATAL).
        return mcEventInterprocess | mcEventDisableTiming;
    default:
        LOG(FATAL) << "Unsupported EventFlag value: " << static_cast<uint32_t>(flags);
    }
    return mcEventDefault;
}
} // namespace

MacaEvent::MacaEvent(EventFlag flags) { MACA_CHECK(mcEventCreateWithFlags(&event_, ToMacaEventFlags(flags))); }

MacaEvent::~MacaEvent() {
    if (event_ != nullptr) {
        MACA_CHECK(mcEventDestroy(event_));
    }
}

mcEvent_t MacaEvent::maca_event() const { return event_; }

MacaStream::MacaStream() { MACA_CHECK(mcStreamCreate(&stream_)); }

MacaStream::MacaStream(int priority) {
    MACA_CHECK(mcStreamCreateWithPriority(&stream_, mcStreamNonBlocking, priority));
}

MacaStream::~MacaStream() {
    // Do nothing.
}

mcStream_t MacaStream::maca_stream() const { return stream_; }

MacaBlasHandle::MacaBlasHandle(Stream *stream) {
    MCBLAS_CHECK(mcblasCreate(&mcblas_handle_));
    MCBLAS_CHECK(mcblasSetStream(mcblas_handle_, dynamic_cast<MacaStream *>(stream)->maca_stream()));
}

MacaBlasHandle::~MacaBlasHandle() {
    // Do nothing.
}

mcblasHandle_t MacaBlasHandle::mcblas_handle() const { return mcblas_handle_; }

} // namespace infini_train::core::maca
