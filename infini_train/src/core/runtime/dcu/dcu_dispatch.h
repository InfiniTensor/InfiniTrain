#pragma once

#include <utility>
#include <vector>

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

#include "infini_train/include/core/backend_type_map.h"
#include "infini_train/include/dtype_dispatch.h"

// -----------------------------------------------------------------------------
// HIP low-precision BackendTypeMap specializations:
//   FP16 -> __half, BF16 -> hip_bfloat16
// -----------------------------------------------------------------------------
namespace infini_train::core {
template <> struct BackendTypeMap<Device::DeviceType::kDCU, DataType::kFLOAT16> {
    using type = __half;
};

template <> struct BackendTypeMap<Device::DeviceType::kDCU, DataType::kBFLOAT16> {
    using type = hip_bfloat16;
};
} // namespace infini_train::core

// Register all standard (non-low-precision) dtypes for the HIP backend.
// FP16/BF16 are registered explicitly above with their HIP-native scalar types.
INFINI_REGISTER_STANDARD_BACKEND_TYPES(infini_train::Device::DeviceType::kDCU)

namespace infini_train::core::dcu {

template <DataType DType> struct DcuTypeMap : BackendTypeMap<Device::DeviceType::kDCU, DType> {};

// -----------------------------------------------------------------------------
// HIP dispatch helpers
// -----------------------------------------------------------------------------

template <DataType... AllowedDTypes, typename Functor, typename... Args>
auto DispatchDcuFunc(DataType dtype, Functor &&func, std::string_view context_identifier = "", Args &&...args) {
    return infini_train::DispatchByTypeMap<DcuTypeMap, AllowedDTypes...>(
        dtype, std::forward<Functor>(func), context_identifier, std::forward<Args>(args)...);
}

template <typename... AllowedTypeLists, typename Functor, typename... Args>
auto DispatchDcuFunc(const std::vector<DataType> &dtypes, Functor &&func, std::string_view context_identifier = "",
                      Args &&...args) {
    return infini_train::DispatchByTypeMap<DcuTypeMap, AllowedTypeLists...>(
        dtypes, std::forward<Functor>(func), context_identifier, std::forward<Args>(args)...);
}

} // namespace infini_train::core::dcu
