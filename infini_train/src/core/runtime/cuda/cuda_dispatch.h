#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <utility>
#include <vector>

#include "infini_train/include/core/backend_type_map.h"
#include "infini_train/include/dtype_dispatch.h"

namespace infini_train::core {
template <> struct NativeScalar<Device::DeviceType::kCUDA, infini_train::FP16> {
    using type = __half;
};

template <> struct NativeScalar<Device::DeviceType::kCUDA, infini_train::BF16> {
    using type = __nv_bfloat16;
};
} // namespace infini_train::core

namespace infini_train::core::cuda {

// -----------------------------------------------------------------------------
// CUDA backend native scalar specializations
// -----------------------------------------------------------------------------
// Map framework low-precision scalar/storage types to CUDA native scalar types.
// This keeps framework public code backend-agnostic while allowing CUDA kernels
// and dispatch to use native CUDA types directly.

// -----------------------------------------------------------------------------
// CUDA backend type map
// -----------------------------------------------------------------------------
// Reuse BackendTypeMap so that:
// - all non-low-precision dtypes fall back to framework TypeMap
// - FP16/BF16 are routed through NativeScalar<Device::kCUDA, ...>

template <DataType DType> struct CudaTypeMap : infini_train::core::BackendTypeMap<Device::DeviceType::kCUDA, DType> {};

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
