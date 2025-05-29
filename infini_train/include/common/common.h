#pragma once

#include <cstdint>

#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))

namespace infini_train {
/**
 * Compile-time type mapping from DataType enum to concrete C++ types.
 *
 * - Primary template: Declared but undefined to enforce specialization
 * - Specializations: Explicit mappings (DataType::kFLOAT32 → float, etc)
 * - TypeMap_t alias: Direct access to mapped type (TypeMap_t<DataType::kINT32> → int32_t)
 *
 * Enables type-safe generic code where operations dispatch based on DataType tokens,
 * with zero runtime overhead. Extend by adding new specializations.
 */
template <DataType DType> struct TypeMap;
template <DataType DType> using TypeMap_t = typename TypeMap<DType>::type;

template <> struct TypeMap<DataType::kUINT8> {
    using type = uint8_t;
};
template <> struct TypeMap<DataType::kINT8> {
    using type = int8_t;
};
template <> struct TypeMap<DataType::kUINT16> {
    using type = uint16_t;
};
template <> struct TypeMap<DataType::kINT16> {
    using type = int16_t;
};
template <> struct TypeMap<DataType::kUINT32> {
    using type = uint32_t;
};
template <> struct TypeMap<DataType::kINT32> {
    using type = int32_t;
};
template <> struct TypeMap<DataType::kUINT64> {
    using type = uint64_t;
};
template <> struct TypeMap<DataType::kINT64> {
    using type = int64_t;
};
template <> struct TypeMap<DataType::kFLOAT32> {
    using type = float;
};
template <> struct TypeMap<DataType::kFLOAT64> {
    using type = double;
};

#ifdef USE_CUDA
// TypeMap for CUDA data types
template <> struct TypeMap<DataType::kBFLOAT16> {
    using type = nv_bfloat16;
};
template <> struct TypeMap<DataType::kFLOAT16> {
    using type = half;
};
#endif
} // namespace infini_train
