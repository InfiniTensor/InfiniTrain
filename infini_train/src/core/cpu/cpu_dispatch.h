#pragma once

#include "infini_train/include/dtype_dispatch.h"

namespace infini_train::core::cpu {

// CPU backend storage-oriented type map.
// For now, CPU uses the default framework scalar/storage mapping directly.
template <DataType DType> struct CpuTypeMap : DefaultScalarTypeMap<DType> {};

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
