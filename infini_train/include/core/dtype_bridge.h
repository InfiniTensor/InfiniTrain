#pragma once

#include <cstdint>
#include <cstring>
#include <type_traits>

#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"

namespace infini_train::core {

/**
 * Dtype bridge
 *
 * Purpose:
 * - Define the backend-agnostic mapping protocol from framework scalar types
 *   (e.g. infini_train::FP16/BF16) to backend-native scalar types
 *   (e.g. __half / __nv_bfloat16 / vendor fp16/bf16 types).
 *
 * Design notes:
 * - This header MUST remain backend-agnostic.
 * - Framework public code should only depend on infini_train::FP16/BF16.
 * - Backend code provides specializations of NativeScalar<Dev, Scalar>.
 * - ScalarConvert provides optional value-level conversion helpers.
 */

// -----------------------------------------------------------------------------
// NativeScalar: framework scalar -> backend native scalar mapping
// -----------------------------------------------------------------------------
// Primary template intentionally undefined.
// Each backend specializes the scalar types it supports.
template <Device::DeviceType Dev, typename Scalar> struct NativeScalar;

template <Device::DeviceType Dev, typename Scalar> using NativeScalar_t = typename NativeScalar<Dev, Scalar>::type;

// Optional convenience alias for CUDA call sites.
// Keep only one copy here; backend files should NOT redefine it.
template <typename Scalar> using NativeScalarCUDA_t = NativeScalar_t<Device::DeviceType::kCUDA, Scalar>;

// -----------------------------------------------------------------------------
// Bitcast utilities
// -----------------------------------------------------------------------------
template <typename To, typename From> inline To Bitcast(const From &from) noexcept {
    static_assert(sizeof(To) == sizeof(From), "Bitcast requires same size");
    static_assert(std::is_trivially_copyable_v<To>, "Bitcast To must be trivially copyable");
    static_assert(std::is_trivially_copyable_v<From>, "Bitcast From must be trivially copyable");

    To to{};
    std::memcpy(&to, &from, sizeof(To));
    return to;
}

// -----------------------------------------------------------------------------
// HasNativeScalar: detect whether a NativeScalar specialization exists
// -----------------------------------------------------------------------------
template <Device::DeviceType Dev, typename Scalar, typename = void> struct HasNativeScalar : std::false_type {};

template <Device::DeviceType Dev, typename Scalar>
struct HasNativeScalar<Dev, Scalar, std::void_t<typename NativeScalar<Dev, Scalar>::type>> : std::true_type {};

template <Device::DeviceType Dev, typename Scalar>
inline constexpr bool HasNativeScalar_v = HasNativeScalar<Dev, Scalar>::value;

// -----------------------------------------------------------------------------
// ScalarConvert: framework scalar <-> backend native scalar conversion glue
// -----------------------------------------------------------------------------
// Primary template intentionally undefined by default.
// Backends may specialize this if simple bitcast is insufficient.
template <Device::DeviceType Dev, typename Scalar, typename Enable = void> struct ScalarConvert;

// Default FP16 conversion: preserve raw 16-bit bit pattern.
template <Device::DeviceType Dev> struct ScalarConvert<Dev, infini_train::FP16, void> {
    static_assert(HasNativeScalar_v<Dev, infini_train::FP16>,
                  "Missing NativeScalar specialization for FP16 on this backend");

    using Native = NativeScalar_t<Dev, infini_train::FP16>;

    static inline Native ToNative(infini_train::FP16 v) noexcept {
        static_assert(sizeof(Native) == sizeof(uint16_t), "Native FP16 must be 16-bit");
        return Bitcast<Native>(v.x);
    }

    static inline infini_train::FP16 FromNative(Native v) noexcept {
        infini_train::FP16 out{};
        static_assert(sizeof(Native) == sizeof(uint16_t), "Native FP16 must be 16-bit");
        out.x = Bitcast<uint16_t>(v);
        return out;
    }
};

// Default BF16 conversion: preserve raw 16-bit bit pattern.
template <Device::DeviceType Dev> struct ScalarConvert<Dev, infini_train::BF16, void> {
    static_assert(HasNativeScalar_v<Dev, infini_train::BF16>,
                  "Missing NativeScalar specialization for BF16 on this backend");

    using Native = NativeScalar_t<Dev, infini_train::BF16>;

    static inline Native ToNative(infini_train::BF16 v) noexcept {
        static_assert(sizeof(Native) == sizeof(uint16_t), "Native BF16 must be 16-bit");
        return Bitcast<Native>(v.x);
    }

    static inline infini_train::BF16 FromNative(Native v) noexcept {
        infini_train::BF16 out{};
        static_assert(sizeof(Native) == sizeof(uint16_t), "Native BF16 must be 16-bit");
        out.x = Bitcast<uint16_t>(v);
        return out;
    }
};

// -----------------------------------------------------------------------------
// Convenience wrappers
// -----------------------------------------------------------------------------
template <Device::DeviceType Dev, typename Scalar> inline NativeScalar_t<Dev, Scalar> ToNative(Scalar v) noexcept {
    return ScalarConvert<Dev, Scalar>::ToNative(v);
}

template <Device::DeviceType Dev, typename Scalar> inline Scalar FromNative(NativeScalar_t<Dev, Scalar> v) noexcept {
    return ScalarConvert<Dev, Scalar>::FromNative(v);
}

} // namespace infini_train::core
