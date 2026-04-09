#pragma once

#include <utility>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "infini_train/include/core/backend_type_map.h"
#include "infini_train/include/dtype_dispatch.h"

// -----------------------------------------------------------------------------
// CUDA low-precision BackendTypeMap specializations:
//   FP16 -> __half, BF16 -> __nv_bfloat16
// -----------------------------------------------------------------------------
namespace infini_train::core {
template <> struct BackendTypeMap<Device::DeviceType::kCUDA, DataType::kFLOAT16> {
    using type = __half;
};

template <> struct BackendTypeMap<Device::DeviceType::kCUDA, DataType::kBFLOAT16> {
    using type = __nv_bfloat16;
};
} // namespace infini_train::core

// Register all standard (non-low-precision) dtypes for the CUDA backend.
// FP16/BF16 are registered explicitly above with their CUDA-native scalar types.
INFINI_REGISTER_STANDARD_BACKEND_TYPES(infini_train::Device::DeviceType::kCUDA)

namespace infini_train::core::cuda {

template <DataType DType> struct CudaTypeMap : BackendTypeMap<Device::DeviceType::kCUDA, DType> {};

// -----------------------------------------------------------------------------
// CUDA dispatch helpers
// -----------------------------------------------------------------------------

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
