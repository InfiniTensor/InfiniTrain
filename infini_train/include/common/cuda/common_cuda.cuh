#pragma once

#include "cuda.h"
#include "cuda_runtime.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "../common.h"
#include "infini_train/include/dispatcher.h"
#ifdef USE_NCCL
#include "nccl.h"
#endif

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

#define CUDA_DRIVER_CHECK(call)                                                                                        \
    do {                                                                                                               \
        CUresult status = call;                                                                                        \
        if (status != CUresult::CUDA_SUCCESS) {                                                                        \
            const char *err_str = nullptr;                                                                             \
            cuGetErrorString(status, &err_str);                                                                        \
            LOG(FATAL) << "CUDA Driver API error: " << #call << " failed with error (" << status                       \
                       << "): " << (err_str ? err_str : "Unknown error");                                              \
        }                                                                                                              \
    } while (0)

#ifdef USE_NCCL
#define NCCL_CHECK(expr)                                                                                               \
    do {                                                                                                               \
        ncclResult_t _status = (expr);                                                                                 \
        if (_status != ncclSuccess) {                                                                                  \
            LOG(FATAL) << "NCCL error: " << ncclGetErrorString(_status) << " at " << __FILE__ << ":" << __LINE__       \
                       << " (" << #expr << ")";                                                                        \
        }                                                                                                              \
    } while (0)
#endif

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
            return __half(x);
        }
    }
    // half conversions
    else if constexpr (std::is_same_v<SRC_base, half>) {
        if constexpr (std::is_same_v<DST_base, float>) {
            return __half2float(x);
        } else if constexpr (std::is_same_v<DST_base, double>) {
            return static_cast<double>(__half2float(x));
        } else if constexpr (std::is_same_v<DST_base, nv_bfloat16>) {
            return __nv_bfloat16(x);
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

template <typename T> __device__ __forceinline__ T Neg(const T &x) {
    if constexpr (std::is_same_v<T, nv_bfloat16> || std::is_same_v<T, half>) {
        return __hneg(x);
    } else {
        return -x;
    }
}

template <typename T> __device__ __forceinline__ T Reciprocal(const T &x) {
    if constexpr (std::is_same_v<T, half>) {
        return __hdiv(__float2half(1.0f), x);
    } else if constexpr (std::is_same_v<T, nv_bfloat16>) {
        return __hdiv(__float2bfloat16(1.0f), x);
    } else {
        return T(1) / x;
    }
}

template <typename T> __device__ __forceinline__ T Sin(const T &x) {
    if constexpr (std::is_same_v<T, half>) {
        return __float2half(__sinf(__half2float(x)));
    } else if constexpr (std::is_same_v<T, nv_bfloat16>) {
        return __float2bfloat16(__sinf(__bfloat162float(x)));
    } else if constexpr (std::is_same_v<T, float>) {
        return __sinf(x);
    } else {
        return std::sin(x);
    }
}

template <typename T> __device__ __forceinline__ T Cos(const T &x) {
    if constexpr (std::is_same_v<T, half>) {
        return __float2half(__cosf(__half2float(x)));
    } else if constexpr (std::is_same_v<T, nv_bfloat16>) {
        return __float2bfloat16(__cosf(__bfloat162float(x)));
    } else if constexpr (std::is_same_v<T, float>) {
        return __cosf(x);
    } else {
        return std::cos(x);
    }
}

template <typename T> __device__ __forceinline__ T Tanh(const T &x) {
    if constexpr (std::is_same_v<T, nv_bfloat16> || std::is_same_v<T, half>) {
        return htanh(x);
    } else if constexpr (std::is_same_v<T, float>) {
        return tanhf(x);
    } else {
        return std::tanh(x);
    }
}

template <typename T> __device__ __forceinline__ T Pow(const T &x, const T &exponent) {
    if constexpr (std::is_same_v<T, nv_bfloat16>) {
        float x_ = __bfloat162float(x);
        float exponent_ = __bfloat162float(exponent);
        float ans_f = __powf(x_, exponent_);
        return __float2bfloat16(isnan(ans_f) ? std::pow(x_, exponent_) : ans_f);
    } else if constexpr (std::is_same_v<T, half>) {
        float x_ = __half2float(x);
        float exponent_ = __half2float(exponent);
        float ans_f = __powf(x_, exponent_);
        return __float2half(isnan(ans_f) ? std::pow(x_, exponent_) : ans_f);
    } else if constexpr (std::is_same_v<T, float>) {
        return powf(x, exponent);
    } else {
        return std::pow(x, exponent);
    }
}

template <typename T> __device__ __forceinline__ T Rsqrt(const T &x) {
    if constexpr (std::is_same_v<T, half>) {
        return __float2half(rsqrtf(__half2float(x)));
    } else if constexpr (std::is_same_v<T, nv_bfloat16>) {
        return __float2bfloat16(rsqrtf(__bfloat162float(x)));
    } else if constexpr (std::is_same_v<T, float>) {
        return rsqrtf(x);
    } else {
        return T(1) / std::sqrt(T(x));
    }
}

template <typename T> __device__ __forceinline__ T Log(const T &x) {
    if constexpr (std::is_same_v<T, nv_bfloat16>) {
        return __float2bfloat16(__logf(__bfloat162float(x)));
    } else if constexpr (std::is_same_v<T, half>) {
        return __float2half(__logf(__half2float(x)));
    } else if constexpr (std::is_same_v<T, float>) {
        return __logf(x);
    } else {
        return std::log(x);
    }
}

template <typename T> __device__ __forceinline__ T Add(const T &a, const T &b) {
    if constexpr (std::is_same_v<T, nv_bfloat16> || std::is_same_v<T, half>) {
        return __hadd(a, b);
    } else {
        return a + b;
    }
}

template <typename T> __device__ __forceinline__ T Sub(const T &a, const T &b) {
    if constexpr (std::is_same_v<T, nv_bfloat16> || std::is_same_v<T, half>) {
        return __hsub(a, b);
    } else {
        return a - b;
    }
}

template <typename T> __device__ __forceinline__ T Mul(const T &a, const T &b) {
    if constexpr (std::is_same_v<T, nv_bfloat16> || std::is_same_v<T, half>) {
        return __hmul(a, b);
    } else {
        return a * b;
    }
}

template <typename T> __device__ __forceinline__ T Div(const T &a, const T &b) {
    if constexpr (std::is_same_v<T, nv_bfloat16> || std::is_same_v<T, half>) {
        return __hdiv(a, b);
    } else {
        return a / b;
    }
}

template <typename T> __device__ __forceinline__ T Sigmoid(const T &x) {
    if constexpr (std::is_same_v<T, float>) {
        return 1.0f / (1.0f + expf(-x));
    } else if constexpr (std::is_same_v<T, nv_bfloat16> || std::is_same_v<T, half>) {
        return __hdiv(T(1), T(1) + hexp(-x));
    } else {
        return T(1) / (T(1) + std::exp(-x));
    }
}

template <typename T> __device__ __forceinline__ T Max(const T &a, const T &b) {
    if constexpr (std::is_same_v<T, nv_bfloat16> || std::is_same_v<T, half>) {
        return __hle(a, b) ? b : a;
    } else if constexpr (std::is_same_v<T, float>) {
        return fmaxf(a, b);
    } else {
        return std::max(a, b);
    }
}

template <typename T> __device__ __forceinline__ T Min(const T &a, const T &b) {
    if constexpr (std::is_same_v<T, nv_bfloat16> || std::is_same_v<T, half>) {
        return __hle(a, b) ? a : b;
    } else if constexpr (std::is_same_v<T, float>) {
        return fminf(a, b);
    } else {
        return std::min(a, b);
    }
}

} // namespace infini_train::common::cuda
