#pragma once

#include <cuda_runtime.h>

#include "infini_train/include/core/stream.h"

namespace infini_train::core::cuda {

class CudaStream : public Stream {
public:
    CudaStream();

    // NOTE(dcj):
    // The CudaStream are "leaked": they are created but never destroyed because the
    // destruction of global variables could happen after the CUDA runtime has
    // already been destroyed and thus invoking cudaStreamDestroy could lead to a
    // crash. It's likely an issue in CUDA, but to be safe - let's just "forget"
    // the destruction.
    ~CudaStream() override;

    cudaStream_t cuda_stream() const;

private:
    cudaStream_t stream_;
};

} // namespace infini_train::core::cuda
