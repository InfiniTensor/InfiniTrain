#include "infini_train/include/common/cuda/gemm.cuh"

#include <cublas_v2.h>

#include "glog/logging.h"

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/datatype.h"

#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

namespace infini_train::kernels::cuda {

cublasHandle_t GetCublasHandle(const Device &device) {
    return dynamic_cast<infini_train::core::cuda::CudaBlasHandle *>(
               infini_train::core::GetDeviceGuardImpl(device.type())->GetBlasHandle(device))
        ->cublas_handle();
}

cudaStream_t GetCudaStream(const Device &device) {
    return dynamic_cast<infini_train::core::cuda::CudaStream *>(
               infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
        ->cuda_stream();
}

namespace {

cudaDataType_t ToCudaDataType(DataType dt) {
    switch (dt) {
    case DataType::kFLOAT32:
        return CUDA_R_32F;
    case DataType::kBFLOAT16:
        return CUDA_R_16BF;
    case DataType::kFLOAT16:
        return CUDA_R_16F;
    default:
        LOG(FATAL) << "GemmCuda: unsupported DataType " << static_cast<int>(dt);
        return CUDA_R_32F; // unreachable
    }
}

} // namespace

void GemmCuda(const GemmParams &p) {
    DCHECK(p.blas_handle != nullptr);

    if (p.batch_count == 1) {
        // strides are unused in the non-batched path; assert they are left at 0
        // to catch accidental misuse early.
        DCHECK_EQ(p.stride_a, 0LL);
        DCHECK_EQ(p.stride_b, 0LL);
        DCHECK_EQ(p.stride_c, 0LL);
    }

    const cudaDataType_t type_a = ToCudaDataType(p.input_dtype);
    const cudaDataType_t type_b = ToCudaDataType(p.input_dtype);
    const cudaDataType_t type_c = ToCudaDataType(p.output_dtype);
    // Always use CUBLAS_COMPUTE_32F: required for bf16/fp16 correctness,
    // and fine for fp32 (same compute path).
    const cublasComputeType_t ctype = CUBLAS_COMPUTE_32F;

    if (p.batch_count == 1) {
        CUBLAS_CHECK(cublasGemmEx(p.blas_handle, p.trans_a, p.trans_b, p.m, p.n, p.k, &p.alpha, p.A, type_a, p.lda, p.B,
                                  type_b, p.ldb, &p.beta, p.C, type_c, p.ldc, ctype, CUBLAS_GEMM_DEFAULT));
    } else {
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(p.blas_handle, p.trans_a, p.trans_b, p.m, p.n, p.k, &p.alpha, p.A,
                                                type_a, p.lda, p.stride_a, p.B, type_b, p.ldb, p.stride_b, &p.beta, p.C,
                                                type_c, p.ldc, p.stride_c, p.batch_count, ctype, CUBLAS_GEMM_DEFAULT));
    }
}

} // namespace infini_train::kernels::cuda
