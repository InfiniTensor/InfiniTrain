#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#ifdef USE_CUDA
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#endif
#ifdef USE_MACA
#include <common/maca_bfloat16.h>
#include <common/maca_fp16.h>
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

/**
 * Compile-time type mapping from C++ types to DataType enum.
 *
 * Example usage: DataTypeMap<int32_t>::value  // Returns DataType::kINT32
 * DataTypeMap_v for convenient access to the mapped value (e.g., DataTypeMap_v<int32_t>).
 */
template <typename T> struct DataTypeMap;
template <typename T> inline constexpr DataType DataTypeMap_v = DataTypeMap<T>::value;

// Macro to define TypeMap specializations and reverse mappings
#define DEFINE_DATA_TYPE_MAPPING(ENUM_VALUE, CPP_TYPE)                                                                 \
    template <> struct TypeMap<DataType::ENUM_VALUE> {                                                                 \
        using type = CPP_TYPE;                                                                                         \
    };                                                                                                                 \
    template <> struct DataTypeMap<CPP_TYPE> {                                                                         \
        static constexpr DataType value = DataType::ENUM_VALUE;                                                        \
    };

DEFINE_DATA_TYPE_MAPPING(kUINT8, uint8_t)
DEFINE_DATA_TYPE_MAPPING(kINT8, int8_t)
DEFINE_DATA_TYPE_MAPPING(kUINT16, uint16_t)
DEFINE_DATA_TYPE_MAPPING(kINT16, int16_t)
DEFINE_DATA_TYPE_MAPPING(kUINT32, uint32_t)
DEFINE_DATA_TYPE_MAPPING(kINT32, int32_t)
DEFINE_DATA_TYPE_MAPPING(kUINT64, uint64_t)
DEFINE_DATA_TYPE_MAPPING(kINT64, int64_t)
DEFINE_DATA_TYPE_MAPPING(kFLOAT32, float)
DEFINE_DATA_TYPE_MAPPING(kFLOAT64, double)

#if defined(USE_CUDA)
DEFINE_DATA_TYPE_MAPPING(kBFLOAT16, nv_bfloat16)
DEFINE_DATA_TYPE_MAPPING(kFLOAT16, half)
#elif defined(USE_MACA)
DEFINE_DATA_TYPE_MAPPING(kBFLOAT16, __maca_bfloat16)
DEFINE_DATA_TYPE_MAPPING(kFLOAT16, __half)
#else
// Non-CUDA fallbacks
template <> struct TypeMap<DataType::kBFLOAT16> {
    using type = uint16_t;
};
template <> struct TypeMap<DataType::kFLOAT16> {
    using type = uint16_t;
};

// TODO(lzm): currently for non-CUDA/CPU, there's an ambiguity of uint16_t mapping to both kUINT16 and
// kFLOAT16/kBFLOAT16. When CPU custom bfloat16/float16 types are defined, we should replace uint16_t with those types.
#endif
#undef DEFINE_DATA_TYPE_MAPPING

template <typename T> struct is_bfloat16 : std::false_type {};
#if defined(USE_CUDA)
template <> struct is_bfloat16<nv_bfloat16> : std::true_type {};
#elif defined(USE_MACA)
template <> struct is_bfloat16<__maca_bfloat16> : std::true_type {};
#endif

template <typename T> struct is_fp16 : std::false_type {};
#if defined(USE_CUDA)
template <> struct is_fp16<half> : std::true_type {};
#elif defined(USE_MACA)
template <> struct is_fp16<__half> : std::true_type {};
#endif

// Extends std::is_floating_point to support CUDA floating-point types.
template <typename T> struct is_floating_point_ext : std::is_floating_point<T> {};

// Extends std::is_arithmetic to support CUDA floating-point types.
template <typename T> struct is_arithmetic_ext : std::is_arithmetic<T> {};

// Specializations for CUDA types
#ifdef USE_CUDA
template <> struct is_floating_point_ext<__nv_bfloat16> : std::true_type {};
template <> struct is_arithmetic_ext<__nv_bfloat16> : std::true_type {};
template <> struct is_floating_point_ext<__half> : std::true_type {};
template <> struct is_arithmetic_ext<__half> : std::true_type {};
#endif

#ifdef USE_MACA
template <> struct is_floating_point_ext<__maca_bfloat16> : std::true_type {};
template <> struct is_arithmetic_ext<__maca_bfloat16> : std::true_type {};
template <> struct is_floating_point_ext<__half> : std::true_type {};
template <> struct is_arithmetic_ext<__half> : std::true_type {};
#endif

namespace {
template <typename T1, typename T2> struct LargerType {
    static constexpr size_t size1 = sizeof(T1);
    static constexpr size_t size2 = sizeof(T2);
    using type = std::conditional_t<(size1 >= size2), T1, T2>;
};

// Specializations of LargerType for the specific 16-bit FP combinations
#ifdef USE_CUDA
template <> struct LargerType<__nv_bfloat16, __half> {
    using type = float;
};

template <> struct LargerType<__half, __nv_bfloat16> {
    using type = float;
};
#endif

#ifdef USE_MACA
template <> struct LargerType<__maca_bfloat16, __half> {
    using type = float;
};

template <> struct LargerType<__half, __maca_bfloat16> {
    using type = float;
};
#endif

/**
 * @brief Finds the first type in a parameter pack that satisfies the given predicate. If no type matches,
 * returns the last type in the pack (base case).
 *
 * @tparam Predicate Template template parameter that takes one type and provides a static `value` member
 * @tparam Ts Parameter pack of types to check
 */
template <template <typename> class Predicate, typename... Ts> struct FirstMatchingType;

template <template <typename> class Predicate, typename T> struct FirstMatchingType<Predicate, T> {
    using type = T;
};

template <template <typename> class Predicate, typename T, typename... Ts>
struct FirstMatchingType<Predicate, T, Ts...> {
    using type = std::conditional_t<Predicate<T>::value, T, typename FirstMatchingType<Predicate, Ts...>::type>;
};

/**
 * @brief Recursively finds the widest type among those that satisfy a predicate. Types not satisfying the predicate are
 * ignored and don't affect the current maximum.
 *
 * @tparam Predicate Template template parameter that defines the type filter
 * @tparam CurrentMax The current widest type found so far
 * @tparam Ts Remaining types to process
 */
template <template <typename> class Predicate, typename CurrentMax, typename... Ts> struct WidestTypeImpl;

template <template <typename> class Predicate, typename CurrentMax> struct WidestTypeImpl<Predicate, CurrentMax> {
    using type = CurrentMax;
};

template <template <typename> class Predicate, typename CurrentMax, typename T, typename... Ts>
struct WidestTypeImpl<Predicate, CurrentMax, T, Ts...> {
    using new_max = std::conditional_t<Predicate<T>::value, typename LargerType<CurrentMax, T>::type, CurrentMax>;
    using type = typename WidestTypeImpl<Predicate, new_max, Ts...>::type;
};

template <template <typename> class Predicate, typename... Ts> struct MaxTypeBySizeWithPredicate {
    using first = typename FirstMatchingType<Predicate, Ts...>::type;
    using type = typename WidestTypeImpl<Predicate, first, Ts...>::type;
};
} // namespace

/**
 * @brief Finds the widest/largest type according to the dtype promotion logic in PyTorch among a pack of arithmetic
 * types.
 *
 * Selects the type with the widest size from the provided type list. Includes support for CUDA floating-point types
 * (__half, __nv_bfloat16) when compiled with CUDA.
 * - If floating-point types are present, selects the largest floating-point type;
 * - Otherwise selects the largest integral type.
 * - If multiple integral types have the same size, the precedence follows the list order (i.e., the first type that has
 * the widest size will be selected).
 *
 * @tparam Ts Pack of types to evaluate. Must all be arithmetic or CUDA floating-point types.
 * @throws static_assert If no types are provided or if any type is non-arithmetic.
 * @note For mixed 16-bit floating-point types (__half and __nv_bfloat16), promotes to float (32-bit).
 */
template <typename... Ts> struct WidestType {
    static_assert(sizeof...(Ts) > 0, "At least one type is required");
    static_assert((is_arithmetic_ext<Ts>::value && ...), "All types must be arithmetic or CUDA floating-point types");
    static constexpr bool has_float = (is_floating_point_ext<Ts>::value || ...);
    using type = typename std::conditional_t<has_float, MaxTypeBySizeWithPredicate<is_floating_point_ext, Ts...>,
                                             MaxTypeBySizeWithPredicate<std::is_integral, Ts...>>::type;
};

// Convenience alias for WidestType::type
template <typename... Ts> using WidestType_t = typename WidestType<Ts...>::type;
} // namespace infini_train
