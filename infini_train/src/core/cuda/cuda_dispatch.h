#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <utility>
#include <vector>

#include "infini_train/include/datatype.h"
#include "infini_train/include/dtype_dispatch.h"

namespace infini_train::core::cuda {

// CUDA backend type map.
// Reuse the default framework scalar mapping for all common scalar types,
// and override only the reduced floating types with CUDA native scalar types.
template <DataType DType> struct CudaTypeMap : DefaultScalarTypeMap<DType> {};

template <> struct CudaTypeMap<DataType::kFLOAT16> {
    using type = __half;
};

template <> struct CudaTypeMap<DataType::kBFLOAT16> {
    using type = __nv_bfloat16;
};

template <DataType DType> using CudaTypeMap_t = typename CudaTypeMap<DType>::type;

template <DataType... AllowedDTypes, typename Functor, typename... Args>
auto DispatchCudaFunc(DataType dtype, Functor &&func, std::string_view context_identifier = "", Args &&...args) {
    return infini_train::DispatchByTypeMap<CudaTypeMap, AllowedDTypes...>(
        dtype, std::forward<Functor>(func), context_identifier, std::forward<Args>(args)...);
}

template <typename... AllowedTypeLists, typename Functor, typename... Args>
auto DispatchCudaFunc(const std::vector<DataType> &dtypes, Functor &&func, std::string_view context_identifier = "",
                      Args &&...args) {
    return infini_train::DispatchByTypeMap<CudaTypeMap, AllowedTypeLists...>(
        dtypes, std::forward<Functor>(func), context_identifier, std::forward<Args>(args)...);
}

} // namespace infini_train::core::cuda
