#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>

#include "infini_train/include/scalar.h"
#include "test_utils.h"

using namespace infini_train;

class ScalarTest : public infini_train::test::InfiniTrainTest {};

TEST_P(ScalarTest, DefaultConstructor) {
    Scalar s;
    EXPECT_EQ(static_cast<int>(s.kind), static_cast<int>(Scalar::Kind::kInt64));
    EXPECT_EQ(s.i, 0);
}

TEST_P(ScalarTest, BoolConstructor) {
    Scalar t(true);
    EXPECT_EQ(static_cast<int>(t.kind), static_cast<int>(Scalar::Kind::kBool));
    EXPECT_EQ(t.u, static_cast<uint64_t>(1));

    Scalar f(false);
    EXPECT_EQ(static_cast<int>(f.kind), static_cast<int>(Scalar::Kind::kBool));
    EXPECT_EQ(f.u, static_cast<uint64_t>(0));
}

TEST_P(ScalarTest, SignedIntConstructor) {
    Scalar pos(42);
    EXPECT_EQ(static_cast<int>(pos.kind), static_cast<int>(Scalar::Kind::kInt64));
    EXPECT_EQ(pos.i, 42);

    Scalar neg(-7);
    EXPECT_EQ(neg.i, -7);

    int16_t short_val = -100;
    Scalar s(short_val);
    EXPECT_EQ(static_cast<int>(s.kind), static_cast<int>(Scalar::Kind::kInt64));
    EXPECT_EQ(s.i, -100);
}

TEST_P(ScalarTest, UnsignedIntConstructor) {
    unsigned int uint_val = 99u;
    Scalar s(uint_val);
    EXPECT_EQ(static_cast<int>(s.kind), static_cast<int>(Scalar::Kind::kUInt64));
    EXPECT_EQ(s.u, static_cast<uint64_t>(99));

    uint64_t uint64_max = std::numeric_limits<uint64_t>::max();
    Scalar smax(uint64_max);
    EXPECT_EQ(smax.u, uint64_max);
}

TEST_P(ScalarTest, FloatConstructor) {
    Scalar s(3.14f);
    EXPECT_EQ(static_cast<int>(s.kind), static_cast<int>(Scalar::Kind::kDouble));
    EXPECT_NEAR(s.d, 3.14, 1e-5);
}

TEST_P(ScalarTest, DoubleConstructor) {
    Scalar s(2.718281828);
    EXPECT_EQ(static_cast<int>(s.kind), static_cast<int>(Scalar::Kind::kDouble));
    EXPECT_NEAR(s.d, 2.718281828, 1e-12);
}

TEST_P(ScalarTest, HalfPrecisionConstructor) {
    FP16 fp16_val(1.5f);
    Scalar sfp16(fp16_val);
    EXPECT_EQ(static_cast<int>(sfp16.kind), static_cast<int>(Scalar::Kind::kDouble));
    EXPECT_NEAR(sfp16.d, 1.5, 1e-3);

    BF16 bf16_val(2.0f);
    Scalar sbf16(bf16_val);
    EXPECT_EQ(static_cast<int>(sbf16.kind), static_cast<int>(Scalar::Kind::kDouble));
    EXPECT_NEAR(sbf16.d, 2.0, 1e-2);
}

TEST_P(ScalarTest, ToNumericConversions) {
    EXPECT_EQ(Scalar(10).to<int64_t>(), 10);
    EXPECT_NEAR(Scalar(3.14).to<double>(), 3.14, 1e-12);
    EXPECT_NEAR(Scalar(42).to<float>(), 42.0f, 1e-6);
    EXPECT_NEAR(Scalar(42).to<double>(), 42.0, 1e-12);
    EXPECT_EQ(Scalar(7.9).to<int64_t>(), 7);
    EXPECT_EQ(Scalar(7.9).to<int32_t>(), 7);
    EXPECT_NEAR(Scalar(-42).to<float>(), -42.0f, 1e-6);
}

TEST_P(ScalarTest, ToBoolConversions) {
    EXPECT_EQ(Scalar(true).to<int64_t>(), 1);
    EXPECT_EQ(Scalar(false).to<int64_t>(), 0);
    EXPECT_NEAR(Scalar(true).to<double>(), 1.0, 1e-12);
    EXPECT_NEAR(Scalar(false).to<double>(), 0.0, 1e-12);
    EXPECT_EQ(Scalar(0).to<bool>(), false);
    EXPECT_EQ(Scalar(5).to<bool>(), true);
}

TEST_P(ScalarTest, ToUnsignedConversions) {
    uint64_t val = 12345;
    EXPECT_NEAR(Scalar(val).to<double>(), 12345.0, 1e-6);
    EXPECT_EQ(Scalar(100.0).to<uint64_t>(), static_cast<uint64_t>(100));
}

TEST_P(ScalarTest, ToHalfPrecisionConversions) {
    auto to_fp16 = Scalar(1.5).to<FP16>();
    EXPECT_NEAR(static_cast<float>(to_fp16), 1.5f, 1e-3);

    auto int_to_fp16 = Scalar(3).to<FP16>();
    EXPECT_NEAR(static_cast<float>(int_to_fp16), 3.0f, 1e-3);

    auto to_bf16 = Scalar(2.0).to<BF16>();
    EXPECT_NEAR(static_cast<float>(to_bf16), 2.0f, 1e-2);

    auto int_to_bf16 = Scalar(5).to<BF16>();
    EXPECT_NEAR(static_cast<float>(int_to_bf16), 5.0f, 1e-1);
}

TEST_P(ScalarTest, NumericLimits) {
    int64_t i64max = std::numeric_limits<int64_t>::max();
    EXPECT_EQ(Scalar(i64max).to<int64_t>(), i64max);

    int64_t i64min = std::numeric_limits<int64_t>::min();
    EXPECT_EQ(Scalar(i64min).to<int64_t>(), i64min);

    uint64_t u64max = std::numeric_limits<uint64_t>::max();
    EXPECT_EQ(Scalar(u64max).to<uint64_t>(), u64max);
}

TEST_P(ScalarTest, ZeroValues) {
    EXPECT_EQ(Scalar(0).to<int64_t>(), 0);
    EXPECT_NEAR(Scalar(0).to<double>(), 0.0, 1e-12);
    EXPECT_NEAR(Scalar(0.0).to<double>(), 0.0, 1e-12);
    EXPECT_EQ(Scalar(0.0).to<int64_t>(), 0);
}

TEST_P(ScalarTest, HalfPrecisionRoundtrip) {
    FP16 fp16(0.5f);
    auto rt_fp16 = Scalar(fp16).to<FP16>();
    EXPECT_NEAR(static_cast<float>(rt_fp16), 0.5f, 1e-3);

    BF16 bf16(4.0f);
    auto rt_bf16 = Scalar(bf16).to<BF16>();
    EXPECT_NEAR(static_cast<float>(rt_bf16), 4.0f, 1e-2);
}

INFINI_TRAIN_REGISTER_TEST(ScalarTest);
