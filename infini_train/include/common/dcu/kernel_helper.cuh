#pragma once
/*
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>
*/
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

namespace infini_train::common::dcu {

template <typename T> __device__ __forceinline__ void AtomicAdd(T *address, T value) { atomicAdd(address, value); }

// HIP lacks a portable BF16 atomicAdd on older DCU targets. Update the
// containing 32-bit word with CAS so adjacent BF16 values remain intact.
__device__ __forceinline__ void AtomicAdd(hip_bfloat16 *address, hip_bfloat16 value) {
    const auto raw_address = reinterpret_cast<std::uintptr_t>(address);
    auto *base = reinterpret_cast<unsigned int *>(raw_address & ~std::uintptr_t{0x3});
    const bool upper = (raw_address & 0x2) != 0;

    unsigned int old = *base;
    unsigned int assumed = 0;
    do {
        assumed = old;
        hip_bfloat16 current;
        current.data = static_cast<uint16_t>(upper ? (assumed >> 16) : (assumed & 0xffff));
        hip_bfloat16 updated(static_cast<float>(current) + static_cast<float>(value));
        const unsigned int updated_bits = static_cast<unsigned int>(updated.data);
        const unsigned int replacement
            = upper ? ((assumed & 0x0000ffffU) | (updated_bits << 16))
                    : ((assumed & 0xffff0000U) | updated_bits);
        old = atomicCAS(base, assumed, replacement);
    } while (old != assumed);
}

/**
 * Converts a value between arbitrary types with specialized handling for
 * HIP floating-point precisions. For primitive types, this offers perfect
 * forwarding which preserves value categories (lvalues/rvalues)
 *
 * @tparam DST Destination type (deduced)
 * @tparam SRC Source type (deduced)
 * @param x Input value (preserves const/volatile and value category)
 * @return Value converted to DST type
 *
 * Example:
 *   half h = Cast<half>(3.14f);       // float -> half (HIP intrinsic)
 *   float f = Cast<float>(h);         // half -> float (HIP intrinsic)
 *   int i = Cast<int>(2.718);         // double -> int (standard cast)
 */
// TODO(lzm): add support for half and hip_bfloat16 conversions with integral types
template <typename DST, typename SRC> __host__ __device__ DST Cast(SRC &&x) {
    static_assert(!std::is_reference_v<DST>, "Cast cannot return reference types");

    using SRC_base = std::remove_cv_t<std::remove_reference_t<SRC>>;
    using DST_base = std::remove_cv_t<std::remove_reference_t<DST>>;

    // hip_bfloat16 conversions
    if constexpr (std::is_same_v<SRC_base, hip_bfloat16>) {
        if constexpr (std::is_same_v<DST_base, float>) {
            return static_cast<float>(x);
        } else if constexpr (std::is_same_v<DST_base, double>) {
            return static_cast<double>(static_cast<float>(x));
        } else if constexpr (std::is_same_v<DST_base, half>) {
            return __float2half(static_cast<float>(x));
        }
    }
    // half conversions
    else if constexpr (std::is_same_v<SRC_base, half>) {
        if constexpr (std::is_same_v<DST_base, float>) {
            return __half2float(x);
        } else if constexpr (std::is_same_v<DST_base, double>) {
            return static_cast<double>(__half2float(x));
        } else if constexpr (std::is_same_v<DST_base, hip_bfloat16>) {
            return hip_bfloat16(__half2float(x));
        }
    }
    // float conversions to reduced precision
    else if constexpr (std::is_same_v<SRC_base, float>) {
        if constexpr (std::is_same_v<DST_base, hip_bfloat16>) {
            return hip_bfloat16(x);
        } else if constexpr (std::is_same_v<DST_base, half>) {
            return __float2half(x);
        }
    }
    // double conversions to reduced precision
    else if constexpr (std::is_same_v<SRC_base, double>) {
        if constexpr (std::is_same_v<DST_base, hip_bfloat16>) {
            return hip_bfloat16(static_cast<float>(x));
        } else if constexpr (std::is_same_v<DST_base, half>) {
            return __double2half(x);
        }
    }
    // Fallback for all other conversions
    return (DST)(std::forward<SRC>(x));
}

template <typename T> __device__ __forceinline__ T Neg(const T &x) {
    if constexpr (std::is_same_v<T, hip_bfloat16>) {
        return hip_bfloat16(-static_cast<float>(x));
    } else if constexpr (std::is_same_v<T, half>) {
        return __hneg(x);
    } else {
        return -x;
    }
}

template <typename T> __device__ __forceinline__ T Reciprocal(const T &x) {
    if constexpr (std::is_same_v<T, half>) {
        return __hdiv(__float2half(1.0f), x);
    } else if constexpr (std::is_same_v<T, hip_bfloat16>) {
        return hip_bfloat16(1.0f / static_cast<float>(x));
    } else {
        return T(1) / x;
    }
}

template <typename T> __device__ __forceinline__ T Sin(const T &x) {
    if constexpr (std::is_same_v<T, half>) {
        return __float2half(__sinf(__half2float(x)));
    } else if constexpr (std::is_same_v<T, hip_bfloat16>) {
        return hip_bfloat16(__sinf(static_cast<float>(x)));
    } else if constexpr (std::is_same_v<T, float>) {
        return __sinf(x);
    } else {
        return std::sin(x);
    }
}

template <typename T> __device__ __forceinline__ T Cos(const T &x) {
    if constexpr (std::is_same_v<T, half>) {
        return __float2half(__cosf(__half2float(x)));
    } else if constexpr (std::is_same_v<T, hip_bfloat16>) {
        return hip_bfloat16(__cosf(static_cast<float>(x)));
    } else if constexpr (std::is_same_v<T, float>) {
        return __cosf(x);
    } else {
        return std::cos(x);
    }
}

template <typename T> __device__ __forceinline__ T Tanh(const T &x) {
    if constexpr (std::is_same_v<T, hip_bfloat16>) {
        return hip_bfloat16(tanhf(static_cast<float>(x)));
    } else if constexpr (std::is_same_v<T, half>) {
        return htanh(x);
    } else if constexpr (std::is_same_v<T, float>) {
        return tanhf(x);
    } else {
        return std::tanh(x);
    }
}

template <typename T> __device__ __forceinline__ T Pow(const T &x, const T &exponent) {
    if constexpr (std::is_same_v<T, hip_bfloat16>) {
        float x_ = static_cast<float>(x);
        float exponent_ = static_cast<float>(exponent);
        float ans_f = __powf(x_, exponent_);
        return hip_bfloat16(__isnan(ans_f) ? std::pow(x_, exponent_) : ans_f);
    } else if constexpr (std::is_same_v<T, half>) {
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
    if constexpr (std::is_same_v<T, half>) {
        return __float2half(rsqrtf(__half2float(x)));
    } else if constexpr (std::is_same_v<T, hip_bfloat16>) {
        return hip_bfloat16(rsqrtf(static_cast<float>(x)));
    } else if constexpr (std::is_same_v<T, float>) {
        return rsqrtf(x);
    } else {
        return T(1) / std::sqrt(T(x));
    }
}

template <typename T> __device__ __forceinline__ T Exp(const T &x) {
    if constexpr (std::is_same_v<T, hip_bfloat16>) {
        return hip_bfloat16(__expf(static_cast<float>(x)));
    } else if constexpr (std::is_same_v<T, half>) {
        return hexp(x);
    } else if constexpr (std::is_same_v<T, float>) {
        return __expf(x);
    } else {
        return std::exp(x);
    }
}

template <typename T> __device__ __forceinline__ T Log(const T &x) {
    if constexpr (std::is_same_v<T, hip_bfloat16>) {
        return hip_bfloat16(__logf(static_cast<float>(x)));
    } else if constexpr (std::is_same_v<T, half>) {
        return __float2half(__logf(__half2float(x)));
    } else if constexpr (std::is_same_v<T, float>) {
        return __logf(x);
    } else {
        return std::log(x);
    }
}

template <typename T> __device__ __forceinline__ T Add(const T &a, const T &b) {
    if constexpr (std::is_same_v<T, hip_bfloat16>) {
        return hip_bfloat16(static_cast<float>(a) + static_cast<float>(b));
    } else if constexpr (std::is_same_v<T, half>) {
        return __hadd(a, b);
    } else {
        return a + b;
    }
}

template <typename T> __device__ __forceinline__ T Sub(const T &a, const T &b) {
    if constexpr (std::is_same_v<T, hip_bfloat16>) {
        return hip_bfloat16(static_cast<float>(a) - static_cast<float>(b));
    } else if constexpr (std::is_same_v<T, half>) {
        return __hsub(a, b);
    } else {
        return a - b;
    }
}

template <typename T> __device__ __forceinline__ T Mul(const T &a, const T &b) {
    if constexpr (std::is_same_v<T, hip_bfloat16>) {
        return hip_bfloat16(static_cast<float>(a) * static_cast<float>(b));
    } else if constexpr (std::is_same_v<T, half>) {
        return __hmul(a, b);
    } else {
        return a * b;
    }
}

template <typename T> __device__ __forceinline__ T Div(const T &a, const T &b) {
    if constexpr (std::is_same_v<T, hip_bfloat16>) {
        return hip_bfloat16(static_cast<float>(a) / static_cast<float>(b));
    } else if constexpr (std::is_same_v<T, half>) {
        return __hdiv(a, b);
    } else {
        return a / b;
    }
}

template <typename T> __device__ __forceinline__ T Sigmoid(const T &x) {
    if constexpr (std::is_same_v<T, float>) {
        return 1.0f / (1.0f + expf(-x));
    } else if constexpr (std::is_same_v<T, hip_bfloat16>) {
        const float xf = static_cast<float>(x);
        return hip_bfloat16(1.0f / (1.0f + expf(-xf)));
    } else if constexpr (std::is_same_v<T, half>) {
        return __hdiv(T(1), T(1) + hexp(-x));
    } else {
        return T(1) / (T(1) + std::exp(-x));
    }
}

template <typename T> __device__ __forceinline__ T Max(const T &a, const T &b) {
    if constexpr (std::is_same_v<T, hip_bfloat16>) {
        return static_cast<float>(a) <= static_cast<float>(b) ? b : a;
    } else if constexpr (std::is_same_v<T, half>) {
        return __hle(a, b) ? b : a;
    } else if constexpr (std::is_same_v<T, float>) {
        return fmaxf(a, b);
    } else {
        return std::max(a, b);
    }
}

template <typename T> __device__ __forceinline__ T Min(const T &a, const T &b) {
    if constexpr (std::is_same_v<T, hip_bfloat16>) {
        return static_cast<float>(a) <= static_cast<float>(b) ? a : b;
    } else if constexpr (std::is_same_v<T, half>) {
        return __hle(a, b) ? a : b;
    } else if constexpr (std::is_same_v<T, float>) {
        return fminf(a, b);
    } else {
        return std::min(a, b);
    }
}

template <typename T> __device__ __forceinline__ T Fma(const T &x, const T &y, const T &z) {
    if constexpr (std::is_same_v<T, half>) {
        return __hfma(x, y, z);
    } else if constexpr (std::is_same_v<T, hip_bfloat16>) {
        return hip_bfloat16(__fmaf_rn(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)));
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
          typename std::enable_if_t<std::is_same<scalar_t, hip_bfloat16>::value> * = nullptr>
__device__ __forceinline__ void fastSpecializedAtomicAdd(scalar_t *tensor, index_t index, const index_t num_elements,
                                                         scalar_t value) {
    AtomicAdd(tensor + index, value);
}

template <typename scalar_t, typename index_t,
          typename std::enable_if_t<!std::is_same<scalar_t, __half>::value
                                    && !std::is_same<scalar_t, hip_bfloat16>::value> * = nullptr>
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
        AtomicAdd(tensor + index, value);
    }
}
} // namespace infini_train::common::dcu
