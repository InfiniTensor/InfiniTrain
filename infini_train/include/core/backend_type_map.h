#pragma once

#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"

namespace infini_train::core {

/**
 * Backend type mapping: DataType -> backend-native dispatch type
 *
 * NativeScalar   — maps framework low-precision scalar types (FP16/BF16) to
 *                  backend-native scalar types (__half / __nv_bfloat16).
 *                  Primary template intentionally undefined.
 *                  Each backend specializes only the types it supports.
 *
 * BackendTypeMap — maps DataType to the C++ type used by kernels/dispatch.
 *                  Primary template intentionally undefined — there is NO
 *                  default fallback to the framework TypeMap<DType>.
 *
 *                  Backends must register dtypes explicitly:
 *                    - Standard types (int, float, double, ...):
 *                        call INFINI_REGISTER_STANDARD_BACKEND_TYPES(Dev)
 *                        once at file scope in the backend's dispatch header.
 *                    - Low-precision types (FP16, BF16):
 *                        specialize NativeScalar<Dev, infini_train::FP16/BF16>.
 *                        The generic partial specializations below then resolve
 *                        automatically via SFINAE-safe helper.
 *
 * If a backend does not register a dtype, HasMappedType_v returns false and
 * DispatchByTypeMap fires a clear static_assert at compile time.
 */

// -----------------------------------------------------------------------------
// NativeScalar: framework scalar -> backend native scalar
// Primary template intentionally undefined.
// -----------------------------------------------------------------------------
template <Device::DeviceType Dev, typename Scalar> struct NativeScalar;

template <Device::DeviceType Dev, typename Scalar> using NativeScalar_t = typename NativeScalar<Dev, Scalar>::type;

// -----------------------------------------------------------------------------
// BackendTypeMap: DataType -> backend dispatch type
// Primary template intentionally undefined — no TypeMap<DType> fallback.
// -----------------------------------------------------------------------------
template <Device::DeviceType Dev, DataType DType> struct BackendTypeMap;

// -----------------------------------------------------------------------------
// SFINAE-safe helper for low-precision type routing.
// When NativeScalar<Dev, Scalar> is undefined, this struct has no `type`
// member, making HasMappedType_v<..., kFLOAT16/kBFLOAT16> return false and
// triggering the static_assert in dispatch rather than an opaque hard error.
// -----------------------------------------------------------------------------
namespace detail {

template <Device::DeviceType Dev, typename Scalar, typename = void>
struct BackendLowPrecisionTypeHelper {}; // no `type` member when NativeScalar absent

template <Device::DeviceType Dev, typename Scalar>
struct BackendLowPrecisionTypeHelper<Dev, Scalar, std::void_t<typename NativeScalar<Dev, Scalar>::type>> {
    using type = typename NativeScalar<Dev, Scalar>::type;
};

} // namespace detail

// Low-precision partial specializations: generic over Dev, resolved via NativeScalar.
template <Device::DeviceType Dev>
struct BackendTypeMap<Dev, DataType::kFLOAT16> : detail::BackendLowPrecisionTypeHelper<Dev, infini_train::FP16> {};

template <Device::DeviceType Dev>
struct BackendTypeMap<Dev, DataType::kBFLOAT16> : detail::BackendLowPrecisionTypeHelper<Dev, infini_train::BF16> {};

} // namespace infini_train::core

// -----------------------------------------------------------------------------
// INFINI_REGISTER_STANDARD_BACKEND_TYPES(DEV)
//
// Explicitly registers the 10 standard (non-low-precision) dtypes for a backend
// device.  Invoke once at file scope (outside any namespace) in the backend's
// dispatch header, e.g.:
//
//   INFINI_REGISTER_STANDARD_BACKEND_TYPES(Device::DeviceType::kCUDA)
//
// FP16 and BF16 are NOT registered here — they are handled via NativeScalar.
// -----------------------------------------------------------------------------
#define INFINI_REGISTER_STANDARD_BACKEND_TYPES(DEV)                                                                    \
    namespace infini_train::core {                                                                                     \
    template <> struct BackendTypeMap<DEV, DataType::kUINT8> {                                                         \
        using type = uint8_t;                                                                                          \
    };                                                                                                                 \
    template <> struct BackendTypeMap<DEV, DataType::kINT8> {                                                          \
        using type = int8_t;                                                                                           \
    };                                                                                                                 \
    template <> struct BackendTypeMap<DEV, DataType::kUINT16> {                                                        \
        using type = uint16_t;                                                                                         \
    };                                                                                                                 \
    template <> struct BackendTypeMap<DEV, DataType::kINT16> {                                                         \
        using type = int16_t;                                                                                          \
    };                                                                                                                 \
    template <> struct BackendTypeMap<DEV, DataType::kUINT32> {                                                        \
        using type = uint32_t;                                                                                         \
    };                                                                                                                 \
    template <> struct BackendTypeMap<DEV, DataType::kINT32> {                                                         \
        using type = int32_t;                                                                                          \
    };                                                                                                                 \
    template <> struct BackendTypeMap<DEV, DataType::kUINT64> {                                                        \
        using type = uint64_t;                                                                                         \
    };                                                                                                                 \
    template <> struct BackendTypeMap<DEV, DataType::kINT64> {                                                         \
        using type = int64_t;                                                                                          \
    };                                                                                                                 \
    template <> struct BackendTypeMap<DEV, DataType::kFLOAT32> {                                                       \
        using type = float;                                                                                            \
    };                                                                                                                 \
    template <> struct BackendTypeMap<DEV, DataType::kFLOAT64> {                                                       \
        using type = double;                                                                                           \
    };                                                                                                                 \
    } /* namespace infini_train::core */
