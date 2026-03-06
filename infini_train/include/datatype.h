#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <string>
#include <type_traits>
#include <unordered_map>

namespace infini_train {

// -----------------------------------------------------------------------------
// Framework scalar types (16-bit storage + fallback scalar semantics)
// -----------------------------------------------------------------------------
// FP16/BF16 are framework-level 16-bit scalar/storage types.
// They are used for:
//   - framework type identity
//   - baseline dtype mapping
//   - metadata / storage layout
//   - CPU/reference/fallback conversion paths
//
// They are NOT intended to define backend-native arithmetic semantics.
// Backend kernels should use backend-specific type maps, e.g.:
//   - CUDA: __half / __nv_bfloat16
//   - CPU : FP16 / BF16 / widened compute types (as needed)
// -----------------------------------------------------------------------------

namespace detail {

// ---------------------------
// BF16 helpers
// ---------------------------
inline constexpr uint16_t FloatToBf16Bits(float value) {
    const uint32_t bits = std::bit_cast<uint32_t>(value);
    const uint32_t lsb = (bits >> 16) & 1u;
    const uint32_t rounding_bias = 0x7fffu + lsb;
    return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

inline constexpr float Bf16BitsToFloat(uint16_t bits) {
    const uint32_t u32 = static_cast<uint32_t>(bits) << 16;
    return std::bit_cast<float>(u32);
}

// ---------------------------
// FP16 helpers
// Pure software IEEE-754 half <-> float conversion for framework fallback use.
// ---------------------------
inline constexpr uint16_t FloatToFp16Bits(float value) {
    const uint32_t bits = std::bit_cast<uint32_t>(value);

    const uint32_t sign = (bits >> 16) & 0x8000u;
    uint32_t mantissa = bits & 0x007fffffu;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xffu);

    // NaN / Inf
    if (exp == 0xff) {
        if (mantissa == 0) {
            return static_cast<uint16_t>(sign | 0x7c00u); // inf
        }
        return static_cast<uint16_t>(sign | 0x7e00u); // quiet NaN
    }

    // Zero / subnormal in float32
    if (exp == 0) {
        return static_cast<uint16_t>(sign);
    }

    // Convert exponent bias: fp32 bias 127 -> fp16 bias 15
    exp = exp - 127 + 15;

    // Overflow -> inf
    if (exp >= 0x1f) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }

    // Underflow -> subnormal / zero
    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }

        mantissa |= 0x00800000u;

        const int shift = 14 - exp;
        uint32_t half_mant = mantissa >> shift;

        const uint32_t remainder = mantissa & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (half_mant & 1u))) {
            ++half_mant;
        }

        return static_cast<uint16_t>(sign | half_mant);
    }

    // Normal fp16
    uint32_t half_exp = static_cast<uint32_t>(exp) << 10;
    uint32_t half_mant = mantissa >> 13;

    const uint32_t round_bits = mantissa & 0x1fffu;
    if (round_bits > 0x1000u || (round_bits == 0x1000u && (half_mant & 1u))) {
        ++half_mant;
        if (half_mant == 0x400u) {
            half_mant = 0;
            half_exp += 0x0400u;
            if (half_exp >= 0x7c00u) {
                return static_cast<uint16_t>(sign | 0x7c00u);
            }
        }
    }

    return static_cast<uint16_t>(sign | half_exp | half_mant);
}

inline constexpr float Fp16BitsToFloat(uint16_t bits) {
    const uint32_t sign = (static_cast<uint32_t>(bits & 0x8000u)) << 16;
    const uint32_t exp = (bits >> 10) & 0x1fu;
    const uint32_t mant = bits & 0x03ffu;

    uint32_t out = 0;

    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            uint32_t mantissa = mant;
            int32_t e = -14;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                --e;
            }
            mantissa &= 0x03ffu;
            const uint32_t exp32 = static_cast<uint32_t>(e + 127) << 23;
            const uint32_t mant32 = mantissa << 13;
            out = sign | exp32 | mant32;
        }
    } else if (exp == 0x1f) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        const uint32_t exp32 = static_cast<uint32_t>(static_cast<int32_t>(exp) - 15 + 127) << 23;
        const uint32_t mant32 = mant << 13;
        out = sign | exp32 | mant32;
    }

    return std::bit_cast<float>(out);
}

} // namespace detail

struct alignas(2) FP16 {
    uint16_t x{0};

    struct from_bits_t {};
    static constexpr from_bits_t from_bits() { return {}; }

    constexpr FP16() = default;
    constexpr FP16(uint16_t bits, from_bits_t) : x(bits) {}

    explicit constexpr FP16(float value) : x(detail::FloatToFp16Bits(value)) {}
    explicit constexpr FP16(double value) : FP16(static_cast<float>(value)) {}
    explicit constexpr FP16(int value) : FP16(static_cast<float>(value)) {}
    explicit constexpr FP16(int64_t value) : FP16(static_cast<float>(value)) {}

    explicit constexpr operator float() const { return detail::Fp16BitsToFloat(x); }
    explicit constexpr operator double() const { return static_cast<double>(static_cast<float>(*this)); }

    FP16 &operator++() {
        *this = FP16(static_cast<float>(*this) + 1.0f);
        return *this;
    }
};

struct alignas(2) BF16 {
    uint16_t x{0};

    struct from_bits_t {};
    static constexpr from_bits_t from_bits() { return {}; }

    constexpr BF16() = default;
    constexpr BF16(uint16_t bits, from_bits_t) : x(bits) {}

    explicit constexpr BF16(float value) : x(detail::FloatToBf16Bits(value)) {}
    explicit constexpr BF16(double value) : BF16(static_cast<float>(value)) {}
    explicit constexpr BF16(int value) : BF16(static_cast<float>(value)) {}
    explicit constexpr BF16(int64_t value) : BF16(static_cast<float>(value)) {}

    explicit constexpr operator float() const { return detail::Bf16BitsToFloat(x); }
    explicit constexpr operator double() const { return static_cast<double>(static_cast<float>(*this)); }

    BF16 &operator++() {
        *this = BF16(static_cast<float>(*this) + 1.0f);
        return *this;
    }
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
// Compile-time type mapping infrastructure
// -----------------------------------------------------------------------------
// Baseline framework scalar/storage mapping.
// This is the single source of truth for:
//   - framework DataType -> C++ type mapping
//   - CPU default type mapping
//   - backend type-map fallback for dtypes without backend-native overrides
template <DataType DType> struct TypeMap;

template <DataType DType> using TypeMap_t = typename TypeMap<DType>::type;

// -----------------------------------------------------------------------------
// Compile-time reverse mapping: framework C++ type -> DataType
// -----------------------------------------------------------------------------
template <typename T> struct DataTypeMap;

template <typename T> inline constexpr DataType DataTypeMap_v = DataTypeMap<T>::value;

// Macro to define baseline mapping + reverse mapping
#define DEFINE_DEFAULT_DATA_TYPE_MAPPING(ENUM_VALUE, CPP_TYPE)                                                         \
    template <> struct TypeMap<DataType::ENUM_VALUE> {                                                                 \
        using type = CPP_TYPE;                                                                                         \
    };                                                                                                                 \
    template <> struct DataTypeMap<CPP_TYPE> {                                                                         \
        static constexpr DataType value = DataType::ENUM_VALUE;                                                        \
    };

DEFINE_DEFAULT_DATA_TYPE_MAPPING(kUINT8, uint8_t)
DEFINE_DEFAULT_DATA_TYPE_MAPPING(kINT8, int8_t)
DEFINE_DEFAULT_DATA_TYPE_MAPPING(kUINT16, uint16_t)
DEFINE_DEFAULT_DATA_TYPE_MAPPING(kINT16, int16_t)
DEFINE_DEFAULT_DATA_TYPE_MAPPING(kUINT32, uint32_t)
DEFINE_DEFAULT_DATA_TYPE_MAPPING(kINT32, int32_t)
DEFINE_DEFAULT_DATA_TYPE_MAPPING(kUINT64, uint64_t)
DEFINE_DEFAULT_DATA_TYPE_MAPPING(kINT64, int64_t)
DEFINE_DEFAULT_DATA_TYPE_MAPPING(kBFLOAT16, BF16)
DEFINE_DEFAULT_DATA_TYPE_MAPPING(kFLOAT16, FP16)
DEFINE_DEFAULT_DATA_TYPE_MAPPING(kFLOAT32, float)
DEFINE_DEFAULT_DATA_TYPE_MAPPING(kFLOAT64, double)

#undef DEFINE_DEFAULT_DATA_TYPE_MAPPING

// -----------------------------------------------------------------------------
// Type traits extensions (framework fallback scalar semantics)
// -----------------------------------------------------------------------------
template <typename T> struct is_floating_point_ext : std::is_floating_point<T> {};

template <typename T> struct is_arithmetic_ext : std::is_arithmetic<T> {};

template <> struct is_floating_point_ext<BF16> : std::true_type {};
template <> struct is_arithmetic_ext<BF16> : std::true_type {};

template <> struct is_floating_point_ext<FP16> : std::true_type {};
template <> struct is_arithmetic_ext<FP16> : std::true_type {};

// -----------------------------------------------------------------------------
// Promotion helpers (framework-level WidestType)
// -----------------------------------------------------------------------------
namespace detail {

template <typename T1, typename T2> struct LargerType {
    static constexpr size_t size1 = sizeof(T1);
    static constexpr size_t size2 = sizeof(T2);
    using type = std::conditional_t<(size1 >= size2), T1, T2>;
};

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
