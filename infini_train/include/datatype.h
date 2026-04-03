#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <string>
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

constexpr size_t DTypeSize(DataType data_type) {
    switch (data_type) {
    case DataType::kUINT8:
        return 1;
    case DataType::kINT8:
        return 1;
    case DataType::kUINT16:
        return 2;
    case DataType::kINT16:
        return 2;
    case DataType::kUINT32:
        return 4;
    case DataType::kINT32:
        return 4;
    case DataType::kUINT64:
        return 8;
    case DataType::kINT64:
        return 8;
    case DataType::kBFLOAT16:
        return 2;
    case DataType::kFLOAT16:
        return 2;
    case DataType::kFLOAT32:
        return 4;
    case DataType::kFLOAT64:
        return 8;
    }
    return 0; // unreachable
}

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
DEFINE_DEFAULT_DATA_TYPE_MAPPING(kFLOAT32, float)
DEFINE_DEFAULT_DATA_TYPE_MAPPING(kFLOAT64, double)

#undef DEFINE_DEFAULT_DATA_TYPE_MAPPING

// ---------------------------------------------------------------------------
// Low-precision types: reverse mapping ONLY (DataTypeMap).
// TypeMap<kFLOAT16> / TypeMap<kBFLOAT16> are intentionally NOT defined here.
// Backend TypeMaps must explicitly provide these mappings; the default TypeMap
// will static_assert at compile time if dispatch reaches an unmapped dtype.
// ---------------------------------------------------------------------------
template <> struct DataTypeMap<FP16> {
    static constexpr DataType value = DataType::kFLOAT16;
};
template <> struct DataTypeMap<BF16> {
    static constexpr DataType value = DataType::kBFLOAT16;
};

// =============================================================================
// DataType-level promotion  (pure enum → enum, no concrete/backend types)
// =============================================================================
// Rules (priority order):
//   1. FP16 + BF16 → FLOAT32   (neither is a lossless superset of the other)
//   2. Any float dominates any integer → keep the float type
//   3. Same category (float-float or int-int) → wider byte size wins
// =============================================================================

/// Returns true for floating-point DataTypes (FP16, BF16, FP32, FP64).
constexpr bool IsFloatingPointDType(DataType dt) {
    return dt == DataType::kFLOAT16 || dt == DataType::kBFLOAT16 || dt == DataType::kFLOAT32
        || dt == DataType::kFLOAT64;
}

/// Binary DataType promotion.  Safe to call in both host and device code.
constexpr DataType PromoteDataTypes(DataType a, DataType b) {
    if (a == b) {
        return a;
    }

    // Rule 1: FP16 ↔ BF16 — no lossless path, promote to FP32
    if ((a == DataType::kFLOAT16 && b == DataType::kBFLOAT16)
        || (a == DataType::kBFLOAT16 && b == DataType::kFLOAT16)) {
        return DataType::kFLOAT32;
    }

    const bool a_fp = IsFloatingPointDType(a);
    const bool b_fp = IsFloatingPointDType(b);

    // Rule 2: float beats integer
    if (a_fp && !b_fp) {
        return a;
    }
    if (b_fp && !a_fp) {
        return b;
    }

    // Rule 3: same category — wider wins
    return DTypeSize(a) >= DTypeSize(b) ? a : b;
}

/// Compile-time binary promotion: DataTypePromotion<A, B>::value
template <DataType A, DataType B> struct DataTypePromotion {
    static constexpr DataType value = PromoteDataTypes(A, B);
};

/// Convenience variable template
template <DataType A, DataType B> inline constexpr DataType DataTypePromotion_v = DataTypePromotion<A, B>::value;

} // namespace infini_train
