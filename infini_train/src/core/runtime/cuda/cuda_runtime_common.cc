#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

#include "infini_train/include/common/cuda/common_cuda.h"

namespace infini_train::core::cuda {
namespace {
uint32_t ToCudaEventFlags(EventFlag flags) {
    switch (flags) {
    case EventFlag::kDefault:
        return cudaEventDefault;
    case EventFlag::kBlockingSync:
        return cudaEventBlockingSync;
    case EventFlag::kDisableTiming:
        return cudaEventDisableTiming;
    case EventFlag::kInterprocess:
        // CUDA requires cudaEventDisableTiming with interprocess events.
        return cudaEventInterprocess | cudaEventDisableTiming;
    default:
        LOG(FATAL) << "Unsupported EventFlag value: " << static_cast<uint32_t>(flags);
    }
    return cudaEventDefault;
}
} // namespace

CudaEvent::CudaEvent(EventFlag flags) { CUDA_CHECK(cudaEventCreateWithFlags(&event_, ToCudaEventFlags(flags))); }

CudaEvent::~CudaEvent() {
    if (event_ != nullptr) {
        CUDA_CHECK(cudaEventDestroy(event_));
    }
}

cudaEvent_t CudaEvent::cuda_event() const { return event_; }

CudaStream::CudaStream() { CUDA_CHECK(cudaStreamCreate(&stream_)); }

CudaStream::CudaStream(int priority) {
    CUDA_CHECK(cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking, priority));
}

CudaStream::~CudaStream() {
    // Do nothing.
}

cudaStream_t CudaStream::cuda_stream() const { return stream_; }

CudaBlasHandle::CudaBlasHandle(Stream *stream) {
    CUBLAS_CHECK(cublasCreate(&cublas_handle_));
    CUBLAS_CHECK(cublasSetStream(cublas_handle_, dynamic_cast<CudaStream *>(stream)->cuda_stream()));
}

CudaBlasHandle::~CudaBlasHandle() {
    // Do nothing.
}

cublasHandle_t CudaBlasHandle::cublas_handle() const { return cublas_handle_; }

} // namespace infini_train::core::cuda
