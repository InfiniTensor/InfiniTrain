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

// -----------------------------------------------------------------------------
// CudaTypeMap: DataType -> CUDA native scalar type
// Primary template intentionally undefined — no default fallback.
// Each dtype is explicitly registered below.
// -----------------------------------------------------------------------------
template <DataType DType> struct CudaTypeMap;

// Register all supported dtypes by delegating to BackendTypeMap<kCUDA, DType>.
// Standard types come from INFINI_REGISTER_STANDARD_BACKEND_TYPES above;
// FP16/BF16 come from the explicit BackendTypeMap specializations above.
#define INFINI_REGISTER_CUDA_TYPEMAP(DTYPE)                                                                            \
    template <>                                                                                                        \
    struct CudaTypeMap<DataType::DTYPE>                                                                                \
        : infini_train::core::BackendTypeMap<Device::DeviceType::kCUDA, DataType::DTYPE> {};

INFINI_REGISTER_CUDA_TYPEMAP(kUINT8)
INFINI_REGISTER_CUDA_TYPEMAP(kINT8)
INFINI_REGISTER_CUDA_TYPEMAP(kUINT16)
INFINI_REGISTER_CUDA_TYPEMAP(kINT16)
INFINI_REGISTER_CUDA_TYPEMAP(kUINT32)
INFINI_REGISTER_CUDA_TYPEMAP(kINT32)
INFINI_REGISTER_CUDA_TYPEMAP(kUINT64)
INFINI_REGISTER_CUDA_TYPEMAP(kINT64)
INFINI_REGISTER_CUDA_TYPEMAP(kFLOAT32)
INFINI_REGISTER_CUDA_TYPEMAP(kFLOAT64)
INFINI_REGISTER_CUDA_TYPEMAP(kFLOAT16)
INFINI_REGISTER_CUDA_TYPEMAP(kBFLOAT16)

#undef INFINI_REGISTER_CUDA_TYPEMAP

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
