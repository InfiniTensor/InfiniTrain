#pragma once

#include <cstdint>
#include <string>
#include <type_traits>
#include <unordered_map>

namespace infini_train {

// -----------------------------------------------------------------------------
// Framework scalar types (opaque 16-bit wrappers)
// -----------------------------------------------------------------------------
// FP16/BF16 are framework-level 16-bit storage wrappers.
// They are used for framework type identity / dtype mapping / metadata transport.
// They are NOT intended to provide arithmetic semantics in backend kernels.
//
struct FP16 {
    uint16_t x{0};
};

struct BF16 {
    uint16_t x{0};
};

// -----------------------------------------------------------------------------
// DataType enum and metadata tables
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Compile-time type mapping: DataType -> C++ type
// -----------------------------------------------------------------------------
// Primary template is declared but undefined to enforce specialization.
template <DataType DType> struct TypeMap;

template <DataType DType> using TypeMap_t = typename TypeMap<DType>::type;

// -----------------------------------------------------------------------------
// Compile-time type mapping: C++ type -> DataType
// -----------------------------------------------------------------------------
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
DEFINE_DATA_TYPE_MAPPING(kBFLOAT16, BF16)
DEFINE_DATA_TYPE_MAPPING(kFLOAT16, FP16)
DEFINE_DATA_TYPE_MAPPING(kFLOAT32, float)
DEFINE_DATA_TYPE_MAPPING(kFLOAT64, double)

#undef DEFINE_DATA_TYPE_MAPPING

// -----------------------------------------------------------------------------
// Type traits extensions (treat FP16/BF16 as arithmetic + floating-point)
// -----------------------------------------------------------------------------
template <typename T> struct is_floating_point_ext : std::is_floating_point<T> {};

template <typename T> struct is_arithmetic_ext : std::is_arithmetic<T> {};

// FP16/BF16 are framework floating-point types.
template <> struct is_floating_point_ext<BF16> : std::true_type {};
template <> struct is_arithmetic_ext<BF16> : std::true_type {};

template <> struct is_floating_point_ext<FP16> : std::true_type {};
template <> struct is_arithmetic_ext<FP16> : std::true_type {};

// -----------------------------------------------------------------------------
// Promotion helpers (WidestType)
// -----------------------------------------------------------------------------
namespace detail {

template <typename T1, typename T2> struct LargerType {
    static constexpr size_t size1 = sizeof(T1);
    static constexpr size_t size2 = sizeof(T2);
    using type = std::conditional_t<(size1 >= size2), T1, T2>;
};

// Special-case: mixed 16-bit floating types promote to float
template <> struct LargerType<BF16, FP16> {
    using type = float;
};
template <> struct LargerType<FP16, BF16> {
    using type = float;
};

/**
 * @brief Finds the first type in a parameter pack that satisfies the given predicate.
 * If no type matches, returns the last type in the pack (base case).
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
 * @brief Recursively finds the widest type among those that satisfy a predicate.
 * Types not satisfying the predicate are ignored and don't affect the current maximum.
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

} // namespace detail

/**
 * @brief Finds the widest/largest type according to a PyTorch-like dtype promotion rule among a pack of arithmetic
 * types.
 *
 * - If floating-point types are present, selects the largest floating-point type;
 * - Otherwise selects the largest integral type.
 * - If multiple integral types have the same size, precedence follows the list order.
 *
 * Note:
 * - FP16/BF16 are treated as floating-point.
 * - Mixed FP16 and BF16 promotes to float (32-bit).
 */
template <typename... Ts> struct WidestType {
    static_assert(sizeof...(Ts) > 0, "At least one type is required");
    static_assert((is_arithmetic_ext<Ts>::value && ...),
                  "All types must be arithmetic or framework floating-point types (FP16/BF16)");

    static constexpr bool has_float = (is_floating_point_ext<Ts>::value || ...);

    using type =
        typename std::conditional_t<has_float, detail::MaxTypeBySizeWithPredicate<is_floating_point_ext, Ts...>,
                                    detail::MaxTypeBySizeWithPredicate<std::is_integral, Ts...>>::type;
};

// Convenience alias
template <typename... Ts> using WidestType_t = typename WidestType<Ts...>::type;

} // namespace infini_train
