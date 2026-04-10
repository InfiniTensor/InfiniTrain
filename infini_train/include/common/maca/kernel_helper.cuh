#pragma once

#include <common/maca_bfloat16.h>
#include <common/maca_fp16.h>

namespace infini_train::common::maca {
/**
 * Converts a value between arbitrary types with specialized handling for
 * MACA floating-point precisions. For primitive types, this offers perfect
 * forwarding which preserves value categories (lvalues/rvalues)
 *
 * @tparam DST Destination type (deduced)
 * @tparam SRC Source type (deduced)
 * @param x Input value (preserves const/volatile and value category)
 * @return Value converted to DST type
 *
 * Example:
 *   __half h = Cast<__half>(3.14f);   // float -> half (MACA intrinsic)
 *   float f = Cast<float>(h);         // half -> float (MACA intrinsic)
 *   int i = Cast<int>(2.718);         // double -> int (standard cast)
 */
// TODO(zbl): add support for half and __maca_bfloat16 conversions with integral types
template <typename DST, typename SRC> __host__ __device__ DST Cast(SRC &&x) {
    static_assert(!std::is_reference_v<DST>, "Cast cannot return reference types");

    using SRC_base = std::remove_cv_t<std::remove_reference_t<SRC>>;
    using DST_base = std::remove_cv_t<std::remove_reference_t<DST>>;

    // __maca_bfloat16 conversions
    if constexpr (std::is_same_v<SRC_base, __maca_bfloat16>) {
        if constexpr (std::is_same_v<DST_base, float>) {
            return __bfloat162float(x);
        } else if constexpr (std::is_same_v<DST_base, double>) {
            return static_cast<double>(__bfloat162float(x));
        } else if constexpr (std::is_same_v<DST_base, __half>) {
            return __half(__bfloat162float(x));
        }
    }
    // half conversions
    else if constexpr (std::is_same_v<SRC_base, __half>) {
        if constexpr (std::is_same_v<DST_base, float>) {
            return __half2float(x);
        } else if constexpr (std::is_same_v<DST_base, double>) {
            return static_cast<double>(__half2float(x));
        } else if constexpr (std::is_same_v<DST_base, __maca_bfloat16>) {
            return __maca_bfloat16(__half2float(x));
        }
    }
    // float conversions to reduced precision
    else if constexpr (std::is_same_v<SRC_base, float>) {
        if constexpr (std::is_same_v<DST_base, __maca_bfloat16>) {
            return __float2bfloat16(x);
        } else if constexpr (std::is_same_v<DST_base, __half>) {
            return __float2half(x);
        }
    }
    // double conversions to reduced precision
    else if constexpr (std::is_same_v<SRC_base, double>) {
        if constexpr (std::is_same_v<DST_base, __maca_bfloat16>) {
            return __double2bfloat16(x);
        } else if constexpr (std::is_same_v<DST_base, __half>) {
            return __double2half(x);
        }
    }
    // Fallback for all other conversions
    if constexpr (std::is_same_v<DST_base, __maca_bfloat16> || std::is_same_v<DST_base, __half>
                  || std::is_same_v<SRC_base, __maca_bfloat16> || std::is_same_v<SRC_base, __half>) {
        return (DST)(static_cast<float>(std::forward<SRC>(x)));;
    } else {
        return static_cast<DST>(std::forward<SRC>(x));;
    }
}

template <typename T> __device__ __forceinline__ T Neg(const T &x) {
    if constexpr (std::is_same_v<T, __maca_bfloat16> || std::is_same_v<T, __half>) {
        return __hneg(x);
    } else {
        return -x;
    }
}

template <typename T> __device__ __forceinline__ T Reciprocal(const T &x) {
    if constexpr (std::is_same_v<T, __half>) {
        return __hdiv(__float2half(1.0f), x);
    } else if constexpr (std::is_same_v<T, __maca_bfloat16>) {
        return __hdiv(__float2bfloat16(1.0f), x);
    } else {
        return T(1) / x;
    }
}

template <typename T> __device__ __forceinline__ T Sin(const T &x) {
    if constexpr (std::is_same_v<T, __half>) {
        return __float2half(__sinf(__half2float(x)));
    } else if constexpr (std::is_same_v<T, __maca_bfloat16>) {
        return __float2bfloat16(__sinf(__bfloat162float(x)));
    } else if constexpr (std::is_same_v<T, float>) {
        return __sinf(x);
    } else {
        return std::sin(x);
    }
}

template <typename T> __device__ __forceinline__ T Cos(const T &x) {
    if constexpr (std::is_same_v<T, __half>) {
        return __float2half(__cosf(__half2float(x)));
    } else if constexpr (std::is_same_v<T, __maca_bfloat16>) {
        return __float2bfloat16(__cosf(__bfloat162float(x)));
    } else if constexpr (std::is_same_v<T, float>) {
        return __cosf(x);
    } else {
        return std::cos(x);
    }
}

template <typename T> __device__ __forceinline__ T Tanh(const T &x) {
    if constexpr (std::is_same_v<T, __half>) {
        return __float2half(tanhf(__half2float(x)));
    } else if constexpr (std::is_same_v<T, __maca_bfloat16>) {
        return __float2bfloat16(tanhf(__bfloat162float(x)));
    } else if constexpr (std::is_same_v<T, float>) {
        return tanhf(x);
    } else {
        return std::tanh(x);
    }
}

template <typename T> __device__ __forceinline__ T Pow(const T &x, const T &exponent) {
    if constexpr (std::is_same_v<T, __maca_bfloat16>) {
        float x_ = __bfloat162float(x);
        float exponent_ = __bfloat162float(exponent);
        float ans_f = __powf(x_, exponent_);
        return __float2bfloat16(__isnan(ans_f) ? std::pow(x_, exponent_) : ans_f);
    } else if constexpr (std::is_same_v<T, __half>) {
        float x_ = __half2float(x);
        float exponent_ = __half2float(exponent);
        float ans_f = __powf(x_, exponent_);
        return __float2half(__isnan(ans_f) ? std::pow(x_, exponent_) : ans_f);
    } else if constexpr (std::is_same_v<T, float>) {
        return powf(x, exponent);
    } else {
        return std::pow(x, exponent);
    }
}

template <typename T> __device__ __forceinline__ T Rsqrt(const T &x) {
    if constexpr (std::is_same_v<T, __half>) {
        return __float2half(rsqrtf(__half2float(x)));
    } else if constexpr (std::is_same_v<T, __maca_bfloat16>) {
        return __float2bfloat16(rsqrtf(__bfloat162float(x)));
    } else if constexpr (std::is_same_v<T, float>) {
        return rsqrtf(x);
    } else {
        return T(1) / std::sqrt(T(x));
    }
}

template <typename T> __device__ __forceinline__ T Exp(const T &x) {
    if constexpr (std::is_same_v<T, __maca_bfloat16> || std::is_same_v<T, __half>) {
        return hexp(x);
    } else if constexpr (std::is_same_v<T, float>) {
        return __expf(x);
    } else {
        return std::exp(x);
    }
}

template <typename T> __device__ __forceinline__ T Log(const T &x) {
    if constexpr (std::is_same_v<T, __maca_bfloat16>) {
        return __float2bfloat16(__logf(__bfloat162float(x)));
    } else if constexpr (std::is_same_v<T, __half>) {
        return __float2half(__logf(__half2float(x)));
    } else if constexpr (std::is_same_v<T, float>) {
        return __logf(x);
    } else {
        return std::log(x);
    }
}

template <typename T> __device__ __forceinline__ T Add(const T &a, const T &b) {
    if constexpr (std::is_same_v<T, __maca_bfloat16> || std::is_same_v<T, __half>) {
        return __hadd(a, b);
    } else {
        return a + b;
    }
}

template <typename T> __device__ __forceinline__ T Sub(const T &a, const T &b) {
    if constexpr (std::is_same_v<T, __maca_bfloat16> || std::is_same_v<T, __half>) {
        return __hsub(a, b);
    } else {
        return a - b;
    }
}

template <typename T> __device__ __forceinline__ T Mul(const T &a, const T &b) {
    if constexpr (std::is_same_v<T, __maca_bfloat16> || std::is_same_v<T, __half>) {
        return __hmul(a, b);
    } else {
        return a * b;
    }
}

template <typename T> __device__ __forceinline__ T Div(const T &a, const T &b) {
    if constexpr (std::is_same_v<T, __maca_bfloat16> || std::is_same_v<T, __half>) {
        return __hdiv(a, b);
    } else {
        return a / b;
    }
}

template <typename T> __device__ __forceinline__ T Sigmoid(const T &x) {
    if constexpr (std::is_same_v<T, float>) {
        return 1.0f / (1.0f + expf(-x));
    } else if constexpr (std::is_same_v<T, __maca_bfloat16> || std::is_same_v<T, __half>) {
        return __hdiv(T(1), T(1) + hexp(-x));
    } else {
        return T(1) / (T(1) + std::exp(-x));
    }
}

template <typename T> __device__ __forceinline__ T Max(const T &a, const T &b) {
    if constexpr (std::is_same_v<T, __maca_bfloat16> || std::is_same_v<T, __half>) {
        return __hle(a, b) ? b : a;
    } else if constexpr (std::is_same_v<T, float>) {
        return fmaxf(a, b);
    } else {
        return std::max(a, b);
    }
}

template <typename T> __device__ __forceinline__ T Min(const T &a, const T &b) {
    if constexpr (std::is_same_v<T, __maca_bfloat16> || std::is_same_v<T, __half>) {
        return __hle(a, b) ? a : b;
    } else if constexpr (std::is_same_v<T, float>) {
        return fminf(a, b);
    } else {
        return std::min(a, b);
    }
}

template <typename T> __device__ __forceinline__ T Fma(const T &x, const T &y, const T &z) {
    if constexpr (std::is_same_v<T, __half>) {
        return __hfma(x, y, z);
    } else if constexpr (std::is_same_v<T, __maca_bfloat16>) {
        return __float2bfloat16(__fmaf_rn(__bfloat162float(x), __bfloat162float(y), __bfloat162float(z)));
    } else if constexpr (std::is_same_v<T, float>) {
        return __fmaf_rn(x, y, z);
    } else {
        return std::fma(x, y, z);
    }
}

template <typename scalar_t, typename index_t,
          typename std::enable_if_t<std::is_same<scalar_t, __half>::value> * = nullptr>
__device__ __forceinline__ void fastSpecializedAtomicAdd(scalar_t *tensor, index_t index, const index_t num_elements,
                                                         scalar_t value) {
    __half *target_addr = tensor + index;
    bool low_byte = ((reinterpret_cast<std::uintptr_t>(target_addr) & (sizeof(__half2) - 1)) == 0);

    if (low_byte && index < (num_elements - 1)) {
        __half2 value2 = __halves2half2(value, __float2half(0.0f));
        atomicAdd(reinterpret_cast<__half2 *>(target_addr), value2);

    } else if (!low_byte && index > 0) {
        __half2 value2 = __halves2half2(__float2half(0.0f), value);
        atomicAdd(reinterpret_cast<__half2 *>(target_addr - 1), value2);

    } else {
        atomicAdd(target_addr, value);
    }
}

template <typename scalar_t, typename index_t,
          typename std::enable_if_t<std::is_same<scalar_t, __maca_bfloat16>::value> * = nullptr>
__device__ __forceinline__ void fastSpecializedAtomicAdd(scalar_t *tensor, index_t index, const index_t num_elements,
                                                         scalar_t value) {
    __maca_bfloat16 *target_addr = tensor + index;
    bool low_byte = ((reinterpret_cast<std::uintptr_t>(target_addr) & (sizeof(__maca_bfloat162) - 1)) == 0);

    if (low_byte && index < (num_elements - 1)) {
        __maca_bfloat162 value2 = __halves2bfloat162(value, __maca_bfloat16(0.0f));
        atomicAdd(reinterpret_cast<__maca_bfloat162 *>(target_addr), value2);

    } else if (!low_byte && index > 0) {
        __maca_bfloat162 value2 = __halves2bfloat162(__maca_bfloat16(0.0f), value);
        atomicAdd(reinterpret_cast<__maca_bfloat162 *>(target_addr - 1), value2);

    } else {
        atomicAdd(target_addr, value);
    }
}

template <typename scalar_t, typename index_t,
          typename std::enable_if_t<!std::is_same<scalar_t, __half>::value
                                    && !std::is_same<scalar_t, __maca_bfloat16>::value> * = nullptr>
__device__ __forceinline__ void fastSpecializedAtomicAdd(scalar_t *tensor, index_t index,
                                                         const index_t /*num_elements*/, scalar_t value) {
    atomicAdd(tensor + index, value);
}

template <class scalar_t, class index_t>
__device__ __forceinline__ void fastAtomicAdd(scalar_t *tensor, index_t index, const index_t num_elements,
                                              scalar_t value, bool fast_atomics) {
    if (fast_atomics) {
        fastSpecializedAtomicAdd(tensor, index, num_elements, value);
    } else {
        atomicAdd(tensor + index, value);
    }
}
} // namespace infini_train::common::maca
