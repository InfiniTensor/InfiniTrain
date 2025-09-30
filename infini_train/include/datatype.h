#pragma once

#include <cstdint>
#include <type_traits>
#include <unordered_map>

#ifdef USE_CUDA
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#endif

namespace infini_train {
enum class DataType : int8_t {
    kUINT8,
    kINT8,
    kUINT16,
    kINT16,
    kUINT32,
    kINT32,
    kUINT64,
    kINT64,
    kBFLOAT16,
    kFLOAT16,
    kFLOAT32,
    kFLOAT64,
};

inline const std::unordered_map<DataType, size_t> kDataTypeToSize = {
    {DataType::kUINT8, 1},    {DataType::kINT8, 1},    {DataType::kUINT16, 2},  {DataType::kINT16, 2},
    {DataType::kUINT32, 4},   {DataType::kINT32, 4},   {DataType::kUINT64, 8},  {DataType::kINT64, 8},
    {DataType::kBFLOAT16, 2}, {DataType::kFLOAT16, 2}, {DataType::kFLOAT32, 4}, {DataType::kFLOAT64, 8},
};

inline const std::unordered_map<DataType, std::string> kDataTypeToDesc = {
    {DataType::kUINT8, "uint8"},   {DataType::kINT8, "int8"},     {DataType::kUINT16, "uint16"},
    {DataType::kINT16, "int16"},   {DataType::kUINT32, "uint32"}, {DataType::kINT32, "int32"},
    {DataType::kUINT64, "uint64"}, {DataType::kINT64, "int64"},   {DataType::kBFLOAT16, "bf16"},
    {DataType::kFLOAT16, "fp16"},  {DataType::kFLOAT32, "fp32"},  {DataType::kFLOAT64, "fp64"},
};

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
template <> struct TypeMap<DataType::kBFLOAT16> {
#ifdef USE_CUDA
    using type = nv_bfloat16;
#else
    using type = uint16_t;
#endif
};
template <> struct TypeMap<DataType::kFLOAT16> {
#ifdef USE_CUDA
    using type = half;
#else
    using type = uint16_t;
#endif
};
} // namespace infini_train
