#include "infini_train/src/core/cuda/cuda_stream.h"

#include <cuda_runtime.h>

#include "infini_train/include/common/cuda/common_cuda.h"

namespace infini_train::core::cuda {
CudaStream::CudaStream() { CUDA_CHECK(cudaStreamCreate(&stream_)); }

cudaStream_t CudaStream::cuda_stream() const { return stream_; }

} // namespace infini_train::core::cuda
