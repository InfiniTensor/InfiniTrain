#pragma once

#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"

namespace infini_train::core {

/**
 * Backend type mapping: DataType -> backend-native dispatch type
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
 *                        directly specialize BackendTypeMap<Dev, kFLOAT16/kBFLOAT16>
 *                        in the backend's dispatch header (the native scalar type
 *                        differs per backend, e.g. __half on CUDA).
 *
 * If a backend does not register a dtype, HasMappedType_v returns false and
 * DispatchByTypeMap fires a clear static_assert at compile time.
 */

// -----------------------------------------------------------------------------
// BackendTypeMap: DataType -> backend dispatch type
// Primary template intentionally undefined — no TypeMap<DType> fallback.
// -----------------------------------------------------------------------------
template <Device::DeviceType Dev, DataType DType> struct BackendTypeMap;

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
// FP16 and BF16 are NOT registered here — backends must specialize
// BackendTypeMap<DEV, kFLOAT16/kBFLOAT16> directly with their native scalar
// type (e.g. __half / __nv_bfloat16 on CUDA).
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
