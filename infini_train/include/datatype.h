#pragma once

#include <cstddef>
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
uint16_t FloatToBf16Bits(float value);
float Bf16BitsToFloat(uint16_t bits);

// ---------------------------
// FP16 helpers
// Pure software IEEE-754 half <-> float conversion for framework fallback use.
// ---------------------------
uint16_t FloatToFp16Bits(float value);
float Fp16BitsToFloat(uint16_t bits);

} // namespace detail

struct alignas(2) FP16 {
    uint16_t x{0};

    struct from_bits_t {};
    static constexpr from_bits_t from_bits() { return {}; }

    constexpr FP16() = default;
    constexpr FP16(uint16_t bits, from_bits_t) : x(bits) {}

    explicit FP16(float value);
    explicit FP16(double value);
    explicit FP16(int value);
    explicit FP16(int64_t value);

    explicit operator float() const;
    explicit operator double() const;

    FP16 &operator++();
};

struct alignas(2) BF16 {
    uint16_t x{0};

    struct from_bits_t {};
    static constexpr from_bits_t from_bits() { return {}; }

    constexpr BF16() = default;
    constexpr BF16(uint16_t bits, from_bits_t) : x(bits) {}

    explicit BF16(float value);
    explicit BF16(double value);
    explicit BF16(int value);
    explicit BF16(int64_t value);

    explicit operator float() const;
    explicit operator double() const;

    BF16 &operator++();
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

size_t DTypeSize(DataType data_type);

extern const std::unordered_map<DataType, std::string> kDataTypeToDesc;

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
bool IsFloatingPointDType(DataType dt);

/// Binary DataType promotion.
DataType PromoteDataTypes(DataType a, DataType b);

} // namespace infini_train
