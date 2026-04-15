#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"

namespace infini_train::kernels::cuda {

/**
 * Return the cuBLAS handle associated with the given device.
 * Shared by linear.cu, matmul.cu, and any future GEMM-using kernels.
 */
cublasHandle_t GetCublasHandle(const Device &device);

/**
 * Return the CUDA stream associated with the given device.
 * Shared by kernels that need to launch device-side code directly.
 */
cudaStream_t GetCudaStream(const Device &device);

/**
 * Parameter bundle for a single GEMM call:
 *   C = alpha * op(A) * op(B) + beta * C
 *
 * batch_count == 1  →  non-batched path  (cublasGemmEx)
 * batch_count  > 1  →  strided-batched   (cublasGemmStridedBatchedEx)
 *
 * When batch_count == 1, stride_a/b/c are unused and must be left at 0.
 */
struct GemmParams {
    cublasOperation_t trans_a = CUBLAS_OP_N;
    cublasOperation_t trans_b = CUBLAS_OP_N;

    int m = 0; // rows of op(A) and C
    int n = 0; // cols of op(B) and C
    int k = 0; // cols of op(A) == rows of op(B)

    const void *A = nullptr;
    int lda = 0;
    const void *B = nullptr;
    int ldb = 0;
    void *C = nullptr;
    int ldc = 0;

    float alpha = 1.0f;
    float beta = 0.0f;

    // batch_count=1: non-batched (Linear path); stride_a/b/c must be 0
    // batch_count>1: strided-batched (Matmul path)
    int batch_count = 1;
    long long stride_a = 0;
    long long stride_b = 0;
    long long stride_c = 0;

    DataType input_dtype;  // dtype of A and B
    DataType output_dtype; // dtype of C (may differ, e.g. bf16 in → fp32 out)

    cublasHandle_t blas_handle = nullptr;
};

/**
 * Execute the GEMM described by `p` via cuBLAS.
 * Dispatches to cublasGemmEx (batch_count==1) or
 * cublasGemmStridedBatchedEx (batch_count>1).
 * Uses CUBLAS_COMPUTE_32F for all input dtypes to ensure precision.
 * Aborts on cuBLAS error (via CUBLAS_CHECK / LOG(FATAL)).
 */
void GemmCuda(const GemmParams &p);

} // namespace infini_train::kernels::cuda
