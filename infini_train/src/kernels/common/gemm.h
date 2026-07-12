#pragma once

#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"

namespace infini_train::kernels {

enum class GemmTranspose : int {
    kNoTranspose = 0,
    kTranspose = 1,
};

/**
 * Parameter bundle for a single GEMM call:
 *   C = alpha * op(A) * op(B) + beta * C
 *
 * batch_count == 1 describes a non-batched GEMM. batch_count > 1 describes a
 * strided-batched GEMM. When batch_count == 1, stride_a/b/c are unused and must
 * be left at 0.
 */
struct GemmParams {
    GemmTranspose trans_a = GemmTranspose::kNoTranspose;
    GemmTranspose trans_b = GemmTranspose::kNoTranspose;

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

    int batch_count = 1;
    long long stride_a = 0;
    long long stride_b = 0;
    long long stride_c = 0;

    DataType input_dtype;
    DataType output_dtype;
};

} // namespace infini_train::kernels
