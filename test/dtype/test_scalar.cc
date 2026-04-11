#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>

#include "glog/logging.h"

#include "infini_train/include/scalar.h"

using namespace infini_train;

// ============================================================================
// Test 1: Default Constructor
// ============================================================================
void TestDefaultConstructor() {
    std::cout << "\n=== Test 1: Default Constructor ===" << std::endl;

    Scalar default_scalar;
    CHECK_EQ(static_cast<int>(default_scalar.kind), static_cast<int>(Scalar::Kind::kInt64));
    CHECK_EQ(default_scalar.i, 0);

    std::cout << "Default constructor test passed!" << std::endl;
}

// ============================================================================
// Test 2: Bool Constructor
// ============================================================================
void TestBoolConstructor() {
    std::cout << "\n=== Test 2: Bool Constructor ===" << std::endl;

    Scalar scalar_true(true);
    CHECK_EQ(static_cast<int>(scalar_true.kind), static_cast<int>(Scalar::Kind::kBool));
    CHECK_EQ(scalar_true.u, static_cast<uint64_t>(1));

    Scalar scalar_false(false);
    CHECK_EQ(static_cast<int>(scalar_false.kind), static_cast<int>(Scalar::Kind::kBool));
    CHECK_EQ(scalar_false.u, static_cast<uint64_t>(0));

    std::cout << "Bool constructor test passed!" << std::endl;
}

// ============================================================================
// Test 3: Signed Integer Constructor
// ============================================================================
void TestSignedIntConstructor() {
    std::cout << "\n=== Test 3: Signed Integer Constructor ===" << std::endl;

    Scalar scalar_positive(42);
    CHECK_EQ(static_cast<int>(scalar_positive.kind), static_cast<int>(Scalar::Kind::kInt64));
    CHECK_EQ(scalar_positive.i, static_cast<int64_t>(42));

    Scalar scalar_negative(-7);
    CHECK_EQ(scalar_negative.i, static_cast<int64_t>(-7));

    int16_t short_val = -100;
    Scalar scalar_short(short_val);
    CHECK_EQ(static_cast<int>(scalar_short.kind), static_cast<int>(Scalar::Kind::kInt64));
    CHECK_EQ(scalar_short.i, static_cast<int64_t>(-100));

    std::cout << "Signed integer constructor test passed!" << std::endl;
}

// ============================================================================
// Test 4: Unsigned Integer Constructor
// ============================================================================
void TestUnsignedIntConstructor() {
    std::cout << "\n=== Test 4: Unsigned Integer Constructor ===" << std::endl;

    unsigned int uint_val = 99u;
    Scalar scalar_uint(uint_val);
    CHECK_EQ(static_cast<int>(scalar_uint.kind), static_cast<int>(Scalar::Kind::kUInt64));
    CHECK_EQ(scalar_uint.u, static_cast<uint64_t>(99));

    uint64_t uint64_max = std::numeric_limits<uint64_t>::max();
    Scalar scalar_max(uint64_max);
    CHECK_EQ(scalar_max.u, uint64_max);

    std::cout << "Unsigned integer constructor test passed!" << std::endl;
}

// ============================================================================
// Test 5: Float Constructor
// ============================================================================
void TestFloatConstructor() {
    std::cout << "\n=== Test 5: Float Constructor ===" << std::endl;

    Scalar scalar_float(3.14f);
    CHECK_EQ(static_cast<int>(scalar_float.kind), static_cast<int>(Scalar::Kind::kDouble));
    CHECK(std::abs(scalar_float.d - 3.14) < 1e-5) << "Expected ~3.14, got " << scalar_float.d;

    std::cout << "Float constructor test passed!" << std::endl;
}

// ============================================================================
// Test 6: Double Constructor
// ============================================================================
void TestDoubleConstructor() {
    std::cout << "\n=== Test 6: Double Constructor ===" << std::endl;

    Scalar scalar_double(2.718281828);
    CHECK_EQ(static_cast<int>(scalar_double.kind), static_cast<int>(Scalar::Kind::kDouble));
    CHECK(std::abs(scalar_double.d - 2.718281828) < 1e-12) << "Expected ~2.718281828, got " << scalar_double.d;

    std::cout << "Double constructor test passed!" << std::endl;
}

// ============================================================================
// Test 7: FP16 / BF16 Constructor
// ============================================================================
void TestHalfPrecisionConstructor() {
    std::cout << "\n=== Test 7: FP16 / BF16 Constructor ===" << std::endl;

    FP16 fp16_val(1.5f);
    Scalar scalar_from_fp16(fp16_val);
    CHECK_EQ(static_cast<int>(scalar_from_fp16.kind), static_cast<int>(Scalar::Kind::kDouble));
    CHECK(std::abs(scalar_from_fp16.d - 1.5) < 1e-3) << "FP16 constructor: expected ~1.5, got " << scalar_from_fp16.d;

    BF16 bf16_val(2.0f);
    Scalar scalar_from_bf16(bf16_val);
    CHECK_EQ(static_cast<int>(scalar_from_bf16.kind), static_cast<int>(Scalar::Kind::kDouble));
    CHECK(std::abs(scalar_from_bf16.d - 2.0) < 1e-2) << "BF16 constructor: expected ~2.0, got " << scalar_from_bf16.d;

    std::cout << "FP16 / BF16 constructor test passed!" << std::endl;
}

// ============================================================================
// Test 8: to<T>() Same-Type and Numeric Conversions
// ============================================================================
void TestToNumericConversions() {
    std::cout << "\n=== Test 8: to<T>() Same-Type and Numeric Conversions ===" << std::endl;

    // Same type
    Scalar scalar_int(10);
    CHECK_EQ(scalar_int.to<int64_t>(), static_cast<int64_t>(10));

    Scalar scalar_double(3.14);
    CHECK(std::abs(scalar_double.to<double>() - 3.14) < 1e-12) << "to<double> failed";

    // Int -> float
    Scalar scalar_positive(42);
    CHECK(std::abs(scalar_positive.to<float>() - 42.0f) < 1e-6) << "to<float> failed";
    CHECK(std::abs(scalar_positive.to<double>() - 42.0) < 1e-12) << "to<double> failed";

    // Float -> int (truncation)
    Scalar scalar_fractional(7.9);
    CHECK_EQ(scalar_fractional.to<int64_t>(), static_cast<int64_t>(7));
    CHECK_EQ(scalar_fractional.to<int32_t>(), static_cast<int32_t>(7));

    // Negative int -> float
    Scalar scalar_negative(-42);
    CHECK(std::abs(scalar_negative.to<float>() - (-42.0f)) < 1e-6) << "negative to<float> failed";

    std::cout << "Numeric conversion test passed!" << std::endl;
}

// ============================================================================
// Test 9: to<T>() Bool Conversions
// ============================================================================
void TestToBoolConversions() {
    std::cout << "\n=== Test 9: to<T>() Bool Conversions ===" << std::endl;

    // Bool -> int
    Scalar scalar_true(true);
    CHECK_EQ(scalar_true.to<int64_t>(), static_cast<int64_t>(1));

    Scalar scalar_false(false);
    CHECK_EQ(scalar_false.to<int64_t>(), static_cast<int64_t>(0));

    // Bool -> double
    CHECK(std::abs(scalar_true.to<double>() - 1.0) < 1e-12) << "true to<double> failed";
    CHECK(std::abs(scalar_false.to<double>() - 0.0) < 1e-12) << "false to<double> failed";

    // Int -> bool
    Scalar scalar_zero(0);
    CHECK_EQ(scalar_zero.to<bool>(), false);

    Scalar scalar_nonzero(5);
    CHECK_EQ(scalar_nonzero.to<bool>(), true);

    std::cout << "Bool conversion test passed!" << std::endl;
}

// ============================================================================
// Test 10: to<T>() Unsigned Conversions
// ============================================================================
void TestToUnsignedConversions() {
    std::cout << "\n=== Test 10: to<T>() Unsigned Conversions ===" << std::endl;

    // uint -> double
    uint64_t uint_val = 12345;
    Scalar scalar_uint(uint_val);
    CHECK(std::abs(scalar_uint.to<double>() - 12345.0) < 1e-6) << "uint to<double> failed";

    // double -> uint
    Scalar scalar_double(100.0);
    CHECK_EQ(scalar_double.to<uint64_t>(), static_cast<uint64_t>(100));

    std::cout << "Unsigned conversion test passed!" << std::endl;
}

// ============================================================================
// Test 11: to<T>() FP16 / BF16 Conversions
// ============================================================================
void TestToHalfPrecisionConversions() {
    std::cout << "\n=== Test 11: to<T>() FP16 / BF16 Conversions ===" << std::endl;

    // NOTE(dcj): These tests exercise scalar.to<FP16/BF16>(), which goes through
    // common::cpu::Cast and follows a double->float->bf16/fp16 two-step path.
    // This differs from what happens in CUDA kernels where dispatch resolves T
    // to __nv_bfloat16/__half and Cast falls through to a one-step static_cast
    // (double->bf16 directly). The two paths may produce different rounding
    // results. See the TODO in scalar.h for the planned fix.

    // Double -> FP16
    Scalar scalar_double(1.5);
    auto to_fp16 = scalar_double.to<FP16>();
    CHECK(std::abs(static_cast<float>(to_fp16) - 1.5f) < 1e-3) << "to<FP16> from double failed";

    // Int -> FP16
    Scalar scalar_int(3);
    auto int_to_fp16 = scalar_int.to<FP16>();
    CHECK(std::abs(static_cast<float>(int_to_fp16) - 3.0f) < 1e-3) << "to<FP16> from int failed";

    // Double -> BF16
    Scalar scalar_double2(2.0);
    auto to_bf16 = scalar_double2.to<BF16>();
    CHECK(std::abs(static_cast<float>(to_bf16) - 2.0f) < 1e-2) << "to<BF16> from double failed";

    // Int -> BF16
    Scalar scalar_int2(5);
    auto int_to_bf16 = scalar_int2.to<BF16>();
    CHECK(std::abs(static_cast<float>(int_to_bf16) - 5.0f) < 1e-1) << "to<BF16> from int failed";

    std::cout << "FP16 / BF16 conversion test passed!" << std::endl;
}

// ============================================================================
// Test 12: Edge Cases — Numeric Limits
// ============================================================================
void TestNumericLimits() {
    std::cout << "\n=== Test 12: Edge Cases — Numeric Limits ===" << std::endl;

    int64_t int64_max = std::numeric_limits<int64_t>::max();
    Scalar scalar_int64_max(int64_max);
    CHECK_EQ(scalar_int64_max.to<int64_t>(), int64_max);

    int64_t int64_min = std::numeric_limits<int64_t>::min();
    Scalar scalar_int64_min(int64_min);
    CHECK_EQ(scalar_int64_min.to<int64_t>(), int64_min);

    uint64_t uint64_max = std::numeric_limits<uint64_t>::max();
    Scalar scalar_uint64_max(uint64_max);
    CHECK_EQ(scalar_uint64_max.to<uint64_t>(), uint64_max);

    std::cout << "Numeric limits test passed!" << std::endl;
}

// ============================================================================
// Test 13: Edge Cases — Zero Values
// ============================================================================
void TestZeroValues() {
    std::cout << "\n=== Test 13: Edge Cases — Zero Values ===" << std::endl;

    Scalar scalar_int_zero(0);
    CHECK_EQ(scalar_int_zero.to<int64_t>(), static_cast<int64_t>(0));
    CHECK(std::abs(scalar_int_zero.to<double>()) < 1e-12) << "int zero to<double> failed";

    Scalar scalar_double_zero(0.0);
    CHECK(std::abs(scalar_double_zero.to<double>()) < 1e-12) << "double zero to<double> failed";
    CHECK_EQ(scalar_double_zero.to<int64_t>(), static_cast<int64_t>(0));

    std::cout << "Zero values test passed!" << std::endl;
}

// ============================================================================
// Test 14: FP16 / BF16 Roundtrip
// ============================================================================
void TestHalfPrecisionRoundtrip() {
    std::cout << "\n=== Test 14: FP16 / BF16 Roundtrip ===" << std::endl;

    FP16 original_fp16(0.5f);
    Scalar scalar_from_fp16(original_fp16);
    auto roundtrip_fp16 = scalar_from_fp16.to<FP16>();
    CHECK(std::abs(static_cast<float>(roundtrip_fp16) - 0.5f) < 1e-3) << "FP16 roundtrip failed";

    BF16 original_bf16(4.0f);
    Scalar scalar_from_bf16(original_bf16);
    auto roundtrip_bf16 = scalar_from_bf16.to<BF16>();
    CHECK(std::abs(static_cast<float>(roundtrip_bf16) - 4.0f) < 1e-2) << "BF16 roundtrip failed";

    std::cout << "FP16 / BF16 roundtrip test passed!" << std::endl;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char *argv[]) {
    google::InitGoogleLogging(argv[0]);

    std::cout << "========================================" << std::endl;
    std::cout << "       Scalar Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;

    TestDefaultConstructor();
    TestBoolConstructor();
    TestSignedIntConstructor();
    TestUnsignedIntConstructor();
    TestFloatConstructor();
    TestDoubleConstructor();
    TestHalfPrecisionConstructor();
    TestToNumericConversions();
    TestToBoolConversions();
    TestToUnsignedConversions();
    TestToHalfPrecisionConversions();
    TestNumericLimits();
    TestZeroValues();
    TestHalfPrecisionRoundtrip();

    std::cout << "\n========================================" << std::endl;
    std::cout << "    All Tests Completed Successfully" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
