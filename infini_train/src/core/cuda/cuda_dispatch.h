#pragma once

#include <format>
#include <type_traits>
#include <utility>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "glog/logging.h"

#include "infini_train/include/datatype.h"
#include "infini_train/include/dispatcher.h"

// Reuse LOG_UNSUPPORTED_DTYPE / DataTypeList / IsDataTypeInList_v / IsTypeInList
// from dispatcher.h

namespace infini_train::core::cuda {

// -----------------------------------------------------------------------------
// CUDA type map: runtime/framework DataType -> CUDA backend compile-time type
// -----------------------------------------------------------------------------
template <DataType DType> struct CudaTypeMap;

// Integral types
template <> struct CudaTypeMap<DataType::kUINT8> {
    using type = uint8_t;
};
template <> struct CudaTypeMap<DataType::kINT8> {
    using type = int8_t;
};
template <> struct CudaTypeMap<DataType::kUINT16> {
    using type = uint16_t;
};
template <> struct CudaTypeMap<DataType::kINT16> {
    using type = int16_t;
};
template <> struct CudaTypeMap<DataType::kUINT32> {
    using type = uint32_t;
};
template <> struct CudaTypeMap<DataType::kINT32> {
    using type = int32_t;
};
template <> struct CudaTypeMap<DataType::kUINT64> {
    using type = uint64_t;
};
template <> struct CudaTypeMap<DataType::kINT64> {
    using type = int64_t;
};

// Floating types
template <> struct CudaTypeMap<DataType::kFLOAT16> {
    using type = __half;
};
template <> struct CudaTypeMap<DataType::kBFLOAT16> {
    using type = __nv_bfloat16;
};
template <> struct CudaTypeMap<DataType::kFLOAT32> {
    using type = float;
};
template <> struct CudaTypeMap<DataType::kFLOAT64> {
    using type = double;
};

// -----------------------------------------------------------------------------
// Single-dtype CUDA dispatch
// -----------------------------------------------------------------------------
template <DataType... AllowedDTypes, typename Functor, typename... Args>
auto DispatchCudaFunc(DataType dtype, Functor &&func, std::string_view context_identifier = "", Args &&...args) {
    switch (dtype) {

#define CASE_FOR_TYPE(DType)                                                                                           \
    case DType: {                                                                                                      \
        if constexpr (infini_train::IsTypeInList<CudaTypeMap_t<DType>, CudaTypeMap_t<AllowedDTypes>...>) {             \
            return std::forward<Functor>(func).template operator()<CudaTypeMap_t<DType>>(std::forward<Args>(args)...); \
        } else {                                                                                                       \
            break;                                                                                                     \
        }                                                                                                              \
    }

        CASE_FOR_TYPE(DataType::kUINT8)
        CASE_FOR_TYPE(DataType::kINT8)
        CASE_FOR_TYPE(DataType::kUINT16)
        CASE_FOR_TYPE(DataType::kINT16)
        CASE_FOR_TYPE(DataType::kUINT32)
        CASE_FOR_TYPE(DataType::kINT32)
        CASE_FOR_TYPE(DataType::kUINT64)
        CASE_FOR_TYPE(DataType::kINT64)
        CASE_FOR_TYPE(DataType::kFLOAT32)
        CASE_FOR_TYPE(DataType::kFLOAT64)
        CASE_FOR_TYPE(DataType::kBFLOAT16)
        CASE_FOR_TYPE(DataType::kFLOAT16)

#undef CASE_FOR_TYPE
    }

    LOG_UNSUPPORTED_DTYPE(dtype, context_identifier);
    std::abort();
}

// -----------------------------------------------------------------------------
// Multi-dtype CUDA dispatch
// Similar to DtypeDispatcher in dispatcher.h, but resolves to CUDA native types.
// -----------------------------------------------------------------------------
namespace detail {

template <size_t Index, typename AllowedListTuple, typename... ResolvedTypes> struct CudaDtypeDispatcher {
    template <typename Functor, typename... Args>
    static auto call(const std::vector<DataType> &dtypes, Functor &&func, std::string_view context_identifier,
                     Args &&...args) {
        constexpr size_t kNumLists = std::tuple_size_v<AllowedListTuple>;

        if constexpr (Index == kNumLists) {
            return std::forward<Functor>(func).template operator()<ResolvedTypes...>(std::forward<Args>(args)...);
        } else {
            using CurrentList = std::tuple_element_t<Index, AllowedListTuple>;
            const DataType dtype = dtypes[Index];

            switch (dtype) {

#define CASE_FOR_TYPE(DType)                                                                                           \
    case DType:                                                                                                        \
        if constexpr (infini_train::IsDataTypeInList_v<DType, CurrentList>) {                                          \
            using T = CudaTypeMap_t<DType>;                                                                            \
            return CudaDtypeDispatcher<Index + 1, AllowedListTuple, ResolvedTypes..., T>::call(                        \
                dtypes, std::forward<Functor>(func), context_identifier, std::forward<Args>(args)...);                 \
        } else {                                                                                                       \
            break;                                                                                                     \
        }

                CASE_FOR_TYPE(DataType::kUINT8)
                CASE_FOR_TYPE(DataType::kINT8)
                CASE_FOR_TYPE(DataType::kUINT16)
                CASE_FOR_TYPE(DataType::kINT16)
                CASE_FOR_TYPE(DataType::kUINT32)
                CASE_FOR_TYPE(DataType::kINT32)
                CASE_FOR_TYPE(DataType::kUINT64)
                CASE_FOR_TYPE(DataType::kINT64)
                CASE_FOR_TYPE(DataType::kFLOAT32)
                CASE_FOR_TYPE(DataType::kFLOAT64)
                CASE_FOR_TYPE(DataType::kBFLOAT16)
                CASE_FOR_TYPE(DataType::kFLOAT16)

#undef CASE_FOR_TYPE
            }

            LOG_UNSUPPORTED_DTYPE(dtype, context_identifier);
            std::abort();
        }
    }
};

} // namespace detail

template <typename... AllowedTypeLists, typename Functor, typename... Args>
auto DispatchCudaFunc(const std::vector<DataType> &dtypes, Functor &&func, std::string_view context_identifier = "",
                      Args &&...args) {
    constexpr size_t kNumLists = sizeof...(AllowedTypeLists);
    if (dtypes.size() != kNumLists) {
        LOG(FATAL) << std::format("DispatchCudaFunc expects {} dtypes, but only got {} in {}", kNumLists, dtypes.size(),
                                  context_identifier);
        std::abort();
    }

    using AllowedListTuple = std::tuple<AllowedTypeLists...>;
    return detail::CudaDtypeDispatcher<0, AllowedListTuple>::call(dtypes, std::forward<Functor>(func),
                                                                  context_identifier, std::forward<Args>(args)...);
}

} // namespace infini_train::core::cuda
