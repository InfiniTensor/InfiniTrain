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

private:
    cublasHandle_t cublas_handle_;
};

} // namespace infini_train::core::cuda
