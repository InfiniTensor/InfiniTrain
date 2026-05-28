#pragma once

#include "infini_train/include/kernels/common/gemm.h"

namespace infini_train::kernels::cuda {

/**
 * Execute the GEMM described by `p` via the CUDA backend.
 *
 * Arguments are passed by value to match Dispatcher function erasure reliably.
 */
void Gemm(Device device, GemmParams p);

/**
 * Parameter bundle for a single SGEMV call (fp32 only):
 *   y = alpha * op(A) * x + beta * y
 *
 * op(A) is m_phys-by-n_phys when trans==N, or n_phys-by-m_phys when trans==T,
 * where m_phys and n_phys are the physical (pre-transpose) row/col counts of A.
 */
struct SgemvParams {
    GemmTranspose trans = GemmTranspose::kNoTranspose;
    int m = 0;
    int n = 0;
    const float *A = nullptr;
    int lda = 0;
    const float *x = nullptr;
    int incx = 1;
    float *y = nullptr;
    int incy = 1;
    float alpha = 1.0f;
    float beta = 0.0f;
};

void SgemvCuda(const Device &device, const SgemvParams &p);

} // namespace infini_train::kernels::cuda
