#include "infini_train/src/core/cuda/cuda_stream.h"

#include <cuda_runtime.h>

#include "infini_train/include/common/cuda/common_cuda.h"

namespace infini_train::core::cuda {

CudaStream::CudaStream() { CUDA_CHECK(cudaStreamCreate(&cuda_stream_)); }

CudaStream::~CudaStream() {
    // Do nothing.
}

cudaStream_t CudaStream::cuda_stream() const { return cuda_stream_; }

} // namespace infini_train::core::cuda
