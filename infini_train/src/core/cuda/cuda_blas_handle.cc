
#include "infini_train/src/core/cuda/cuda_blas_handle.h"

#include "infini_train/include/common/cuda/common_cuda.h"

#include "infini_train/src/core/cuda/cuda_stream.h"

namespace infini_train::core::cuda {

CudaBlasHandle::CudaBlasHandle(Stream *stream) {
    CUBLAS_CHECK(cublasCreate(&cublas_handle_));
    CUBLAS_CHECK(cublasSetStream(cublas_handle_, dynamic_cast<CudaStream *>(stream)->cuda_stream()));
}

cublasHandle_t CudaBlasHandle::cublas_handle() const { return cublas_handle_; }

} // namespace infini_train::core::cuda
