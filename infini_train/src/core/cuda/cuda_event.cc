#include "infini_train/src/core/cuda/cuda_event.h"

#include "infini_train/include/common/cuda/common_cuda.h"

namespace infini_train::core::cuda {

CudaEvent::CudaEvent(uint32_t flags) { CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags)); }

CudaEvent::~CudaEvent() {
    if (event_ != nullptr) {
        CUDA_CHECK(cudaEventDestroy(event_));
    }
}

cudaEvent_t CudaEvent::cuda_event() const { return event_; }

} // namespace infini_train::core::cuda
