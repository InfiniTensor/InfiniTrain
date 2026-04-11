#pragma once

#include <utility>
#include <vector>

#include "infini_train/include/core/backend_type_map.h"
#include "infini_train/include/dtype_dispatch.h"

// -----------------------------------------------------------------------------
// CPU low-precision BackendTypeMap specializations:
//   FP16 -> infini_train::FP16, BF16 -> infini_train::BF16
// CPU uses the framework wrapper types directly (host-side conversion).
// -----------------------------------------------------------------------------
namespace infini_train::core {
template <> struct BackendTypeMap<Device::DeviceType::kCPU, DataType::kFLOAT16> {
    using type = infini_train::FP16;
};

template <> struct BackendTypeMap<Device::DeviceType::kCPU, DataType::kBFLOAT16> {
    using type = infini_train::BF16;
};
} // namespace infini_train::core

// Register all standard (non-low-precision) dtypes for the CPU backend.
// FP16/BF16 are registered explicitly above.
INFINI_REGISTER_STANDARD_BACKEND_TYPES(infini_train::Device::DeviceType::kCPU)

namespace infini_train::core::cpu {

// -----------------------------------------------------------------------------
// CpuTypeMap: DataType -> CPU native scalar type
// Primary template intentionally undefined — no default fallback.
// Each dtype is explicitly registered below.
// -----------------------------------------------------------------------------
template <DataType DType> struct CpuTypeMap;

#define INFINI_REGISTER_CPU_TYPEMAP(DTYPE)                                                                             \
    template <>                                                                                                        \
    struct CpuTypeMap<DataType::DTYPE>                                                                                 \
        : infini_train::core::BackendTypeMap<Device::DeviceType::kCPU, DataType::DTYPE> {};

INFINI_REGISTER_CPU_TYPEMAP(kUINT8)
INFINI_REGISTER_CPU_TYPEMAP(kINT8)
INFINI_REGISTER_CPU_TYPEMAP(kUINT16)
INFINI_REGISTER_CPU_TYPEMAP(kINT16)
INFINI_REGISTER_CPU_TYPEMAP(kUINT32)
INFINI_REGISTER_CPU_TYPEMAP(kINT32)
INFINI_REGISTER_CPU_TYPEMAP(kUINT64)
INFINI_REGISTER_CPU_TYPEMAP(kINT64)
INFINI_REGISTER_CPU_TYPEMAP(kFLOAT32)
INFINI_REGISTER_CPU_TYPEMAP(kFLOAT64)
INFINI_REGISTER_CPU_TYPEMAP(kFLOAT16)
INFINI_REGISTER_CPU_TYPEMAP(kBFLOAT16)

#undef INFINI_REGISTER_CPU_TYPEMAP

// -----------------------------------------------------------------------------
// CPU dispatch helpers
// -----------------------------------------------------------------------------

template <DataType... AllowedDTypes, typename Functor, typename... Args>
auto DispatchCpuFunc(DataType dtype, Functor &&func, std::string_view context_identifier = "", Args &&...args) {
    return infini_train::DispatchByTypeMap<CpuTypeMap, AllowedDTypes...>(
        dtype, std::forward<Functor>(func), context_identifier, std::forward<Args>(args)...);
}

template <typename... AllowedTypeLists, typename Functor, typename... Args>
auto DispatchCpuFunc(const std::vector<DataType> &dtypes, Functor &&func, std::string_view context_identifier = "",
                     Args &&...args) {
    return infini_train::DispatchByTypeMap<CpuTypeMap, AllowedTypeLists...>(
        dtypes, std::forward<Functor>(func), context_identifier, std::forward<Args>(args)...);
}

} // namespace infini_train::core::cpu
