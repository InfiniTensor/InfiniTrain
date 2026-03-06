#pragma once

#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"

namespace infini_train::core {

/**
 * Backend type mapping: DataType -> backend dispatch type
 *
 * NativeScalar   — maps framework low-precision scalar types (FP16/BF16) to
 *                  backend-native scalar types (__half / __nv_bfloat16).
 *                  Each backend specializes only the types it needs.
 *
 * BackendTypeMap — maps DataType to the C++ type used by kernels/dispatch.
 *                  Falls back to the framework TypeMap for all types
 *                  except FP16/BF16, which are routed through NativeScalar.
 *
 * Value-level conversion between framework scalars and native scalars is out
 * of scope here; kernels use common::cuda::Cast<T> directly.
 */

// -----------------------------------------------------------------------------
// NativeScalar: framework scalar -> backend native scalar type mapping
// Primary template intentionally undefined.
// Each backend specializes the scalar types it supports.
// -----------------------------------------------------------------------------
template <Device::DeviceType Dev, typename Scalar> struct NativeScalar;

template <Device::DeviceType Dev, typename Scalar> using NativeScalar_t = typename NativeScalar<Dev, Scalar>::type;

// -----------------------------------------------------------------------------
// BackendTypeMap: DataType -> backend dispatch type
// Primary template falls back to the framework TypeMap.
// FP16/BF16 are overridden to resolve through NativeScalar.
// -----------------------------------------------------------------------------
template <Device::DeviceType Dev, DataType DType> struct BackendTypeMap : infini_train::TypeMap<DType> {};

template <Device::DeviceType Dev> struct BackendTypeMap<Dev, DataType::kFLOAT16> {
    using type = NativeScalar_t<Dev, infini_train::FP16>;
};

template <Device::DeviceType Dev> struct BackendTypeMap<Dev, DataType::kBFLOAT16> {
    using type = NativeScalar_t<Dev, infini_train::BF16>;
};

} // namespace infini_train::core
