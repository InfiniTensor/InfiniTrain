#include "infini_train/src/kernels/cuda/common/gemm.cuh"

#include <cublas_v2.h>

#include "glog/logging.h"

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/dispatcher.h"

#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

namespace infini_train::kernels::cuda {

namespace {

cublasOperation_t ToCublasOperation(GemmTranspose op) {
    switch (op) {
    case GemmTranspose::kNoTranspose:
        return CUBLAS_OP_N;
    case GemmTranspose::kTranspose:
        return CUBLAS_OP_T;
    }
    LOG(FATAL) << "Gemm: unsupported transpose flag " << static_cast<int>(op);
    return CUBLAS_OP_N; // unreachable
}

cudaDataType_t ToCudaDataType(DataType dt) {
    switch (dt) {
    case DataType::kFLOAT32:
        return CUDA_R_32F;
    case DataType::kBFLOAT16:
        return CUDA_R_16BF;
    case DataType::kFLOAT16:
        return CUDA_R_16F;
    default:
        LOG(FATAL) << "Gemm: unsupported DataType " << static_cast<int>(dt);
        return CUDA_R_32F; // unreachable
    }
}

} // namespace

void Gemm(Device device, GemmParams p) {
    const cublasHandle_t blas_handle = dynamic_cast<infini_train::core::cuda::CudaBlasHandle *>(
                                           infini_train::core::GetDeviceGuardImpl(device.type())->GetBlasHandle(device))
                                           ->cublas_handle();

    if (p.batch_count == 1) {
        // strides are unused in the non-batched path; assert they are left at 0
        // to catch accidental misuse early.
        DCHECK_EQ(p.stride_a, 0LL);
        DCHECK_EQ(p.stride_b, 0LL);
        DCHECK_EQ(p.stride_c, 0LL);
    }

    const cublasOperation_t trans_a = ToCublasOperation(p.trans_a);
    const cublasOperation_t trans_b = ToCublasOperation(p.trans_b);
    const cudaDataType_t type_a = ToCudaDataType(p.input_dtype);
    const cudaDataType_t type_b = ToCudaDataType(p.input_dtype);
    const cudaDataType_t type_c = ToCudaDataType(p.output_dtype);
    // Always use CUBLAS_COMPUTE_32F: required for bf16/fp16 correctness,
    // and fine for fp32 (same compute path).
    const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;

    if (p.batch_count == 1) {
        CUBLAS_CHECK(cublasGemmEx(blas_handle, trans_a, trans_b, p.m, p.n, p.k, &p.alpha, p.A, type_a, p.lda, p.B,
                                  type_b, p.ldb, &p.beta, p.C, type_c, p.ldc, compute_type, CUBLAS_GEMM_DEFAULT));
    } else {
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(blas_handle, trans_a, trans_b, p.m, p.n, p.k, &p.alpha, p.A, type_a,
                                                p.lda, p.stride_a, p.B, type_b, p.ldb, p.stride_b, &p.beta, p.C, type_c,
                                                p.ldc, p.stride_c, p.batch_count, compute_type, CUBLAS_GEMM_DEFAULT));
    }
}

void SgemvCuda(const Device &device, const SgemvParams &p) {
    const cublasHandle_t blas_handle = dynamic_cast<infini_train::core::cuda::CudaBlasHandle *>(
                                           infini_train::core::GetDeviceGuardImpl(device.type())->GetBlasHandle(device))
                                           ->cublas_handle();
    CUBLAS_CHECK(cublasSgemv(blas_handle, ToCublasOperation(p.trans), p.m, p.n, &p.alpha, p.A, p.lda, p.x, p.incx,
                             &p.beta, p.y, p.incy));
}

} // namespace infini_train::kernels::cuda

REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, Gemm, infini_train::kernels::cuda::Gemm)
