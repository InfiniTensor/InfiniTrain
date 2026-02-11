#pragma once

#include <cublas_v2.h>

#include "infini_train/include/core/blas_handle.h"

namespace infini_train::core {
class Stream;
}

namespace infini_train::core::cuda {

class CudaBlasHandle : public BlasHandle {
public:
    explicit CudaBlasHandle(Stream *stream);

    // NOTE(dcj):
    // The CudaBlasHandle are "leaked": they are created but never destroyed because the
    // destruction of global variables could happen after the CUDA runtime has
    // already been destroyed and thus invoking cudaStreamDestroy could lead to a
    // crash. It's likely an issue in CUDA, but to be safe - let's just "forget"
    // the destruction.
    ~CudaBlasHandle() override;

    cublasHandle_t cublas_handle() const;

private:
    cublasHandle_t cublas_handle_;
};

} // namespace infini_train::core::cuda
