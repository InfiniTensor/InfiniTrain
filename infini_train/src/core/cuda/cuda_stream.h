#pragma once

#include <cuda_runtime.h>

#include "infini_train/include/core/stream.h"

namespace infini_train::core::cuda {

class CudaStream : public Stream {
public:
    CudaStream();

    cudaStream_t cuda_stream() const;

private:
    cudaStream_t stream_;
};

} // namespace infini_train::core::cuda
