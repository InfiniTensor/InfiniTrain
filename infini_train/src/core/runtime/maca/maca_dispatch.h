#pragma once

#include <utility>
#include <vector>

#include <common/maca_bfloat16.h>
#include <common/maca_fp16.h>

#include "infini_train/include/core/backend_type_map.h"
#include "infini_train/include/dtype_dispatch.h"

// -----------------------------------------------------------------------------
// MACA low-precision BackendTypeMap specializations:
//   FP16 -> __half, BF16 -> __maca_bfloat16
// -----------------------------------------------------------------------------
namespace infini_train::core {
template <> struct BackendTypeMap<Device::DeviceType::kMACA, DataType::kFLOAT16> {
    using type = __half;
};

template <> struct BackendTypeMap<Device::DeviceType::kMACA, DataType::kBFLOAT16> {
    using type = __maca_bfloat16;
};
} // namespace infini_train::core

// Register all standard (non-low-precision) dtypes for the MACA backend.
// FP16/BF16 are registered explicitly above with their MACA-native scalar types.
INFINI_REGISTER_STANDARD_BACKEND_TYPES(infini_train::Device::DeviceType::kMACA)

namespace infini_train::core::maca {

template <DataType DType> struct MacaTypeMap : BackendTypeMap<Device::DeviceType::kMACA, DType> {};

// -----------------------------------------------------------------------------
// MACA dispatch helpers
// -----------------------------------------------------------------------------

template <DataType... AllowedDTypes, typename Functor, typename... Args>
auto DispatchMacaFunc(DataType dtype, Functor &&func, std::string_view context_identifier = "", Args &&...args) {
    return infini_train::DispatchByTypeMap<MacaTypeMap, AllowedDTypes...>(
        dtype, std::forward<Functor>(func), context_identifier, std::forward<Args>(args)...);
}

template <typename... AllowedTypeLists, typename Functor, typename... Args>
auto DispatchMacaFunc(const std::vector<DataType> &dtypes, Functor &&func, std::string_view context_identifier = "",
                      Args &&...args) {
    return infini_train::DispatchByTypeMap<MacaTypeMap, AllowedTypeLists...>(
        dtypes, std::forward<Functor>(func), context_identifier, std::forward<Args>(args)...);
}

} // namespace infini_train::core::maca
