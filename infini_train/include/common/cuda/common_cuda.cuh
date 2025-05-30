#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../common.h"
#include "infini_train/include/dispatcher.h"

namespace infini_train::common::cuda {

// Common CUDA Macros
#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t status = call;                                                                                     \
        if (status != cudaSuccess) {                                                                                   \
            LOG(FATAL) << "CUDA Error: " << cudaGetErrorString(status) << " at " << __FILE__ << ":" << __LINE__;       \
        }                                                                                                              \
    } while (0)

#define CUBLAS_CHECK(call)                                                                                             \
    do {                                                                                                               \
        cublasStatus_t status = call;                                                                                  \
        if (status != CUBLAS_STATUS_SUCCESS) {                                                                         \
            LOG(FATAL) << "CUBLAS Error: " << cublasGetStatusString(status) << " at " << __FILE__ << ":" << __LINE__;  \
        }                                                                                                              \
    } while (0)

/**
 * Converts a value between arbitrary types with specialized handling for
 * CUDA floating-point precisions. For primitive types, this offers perfect
 * forwarding which preserves value categories (lvalues/rvalues)
 *
 * @tparam DST Destination type (deduced)
 * @tparam SRC Source type (deduced)
 * @param x Input value (preserves const/volatile and value category)
 * @return Value converted to DST type
 *
 * Example:
 *   half h = Cast<half>(3.14f);       // float -> half (CUDA intrinsic)
 *   float f = Cast<float>(h);         // half -> float (CUDA intrinsic)
 *   int i = Cast<int>(2.718);         // double -> int (standard cast)
 */
template <typename DST, typename SRC> __host__ __device__ DST Cast(SRC &&x) {
    using SRC_base = std::remove_cv_t<std::remove_reference_t<SRC>>;
    using DST_base = std::remove_cv_t<std::remove_reference_t<DST>>;

    // nv_bfloat16 conversions
    if constexpr (std::is_same_v<SRC_base, nv_bfloat16>) {
        if constexpr (std::is_same_v<DST_base, float>) {
            return __bfloat162float(x);
        } else if constexpr (std::is_same_v<DST_base, double>) {
            return static_cast<double>(__bfloat162float(x));
        } else if constexpr (std::is_same_v<DST_base, half>) {
            return __bfloat162half(x);
        }
    }
    // half conversions
    else if constexpr (std::is_same_v<SRC_base, half>) {
        if constexpr (std::is_same_v<DST_base, float>) {
            return __half2float(x);
        } else if constexpr (std::is_same_v<DST_base, double>) {
            return static_cast<double>(__half2float(x));
        } else if constexpr (std::is_same_v<DST_base, nv_bfloat16>) {
            return __half2bfloat16(x);
        }
    }
    // float conversions to reduced precision
    else if constexpr (std::is_same_v<SRC_base, float>) {
        if constexpr (std::is_same_v<DST_base, nv_bfloat16>) {
            return __float2bfloat16(x);
        } else if constexpr (std::is_same_v<DST_base, half>) {
            return __float2half(x);
        }
    }
    // double conversions to reduced precision
    else if constexpr (std::is_same_v<SRC_base, double>) {
        if constexpr (std::is_same_v<DST_base, nv_bfloat16>) {
            return __double2bfloat16(x);
        } else if constexpr (std::is_same_v<DST_base, half>) {
            return __double2half(x);
        }
    }
    // Fallback for all other conversions
    return (DST)(std::forward<SRC>(x));
}

} // namespace infini_train::common::cuda
