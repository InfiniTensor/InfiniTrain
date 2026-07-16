#pragma once

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

namespace infini_train::kernels::cuda::flash_attention_compat {

inline void CheckCuda(cudaError_t status, const char *expression, const char *file, int line) {
    if (status == cudaSuccess) {
        return;
    }
    std::fprintf(stderr, "FlashAttention CUDA error at %s:%d: %s failed: %s\n", file, line, expression,
                 cudaGetErrorString(status));
    std::abort();
}

} // namespace infini_train::kernels::cuda::flash_attention_compat

#define C10_CUDA_CHECK(expression)                                                                                     \
    ::infini_train::kernels::cuda::flash_attention_compat::CheckCuda((expression), #expression, __FILE__, __LINE__)
#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())
