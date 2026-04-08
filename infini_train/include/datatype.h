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
