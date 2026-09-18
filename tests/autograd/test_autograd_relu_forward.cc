#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autograd/activations.h"
#include "infini_train/include/nn/modules/activations.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

namespace {
uint32_t FloatBits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

// Reference for the finite-valued part: NaN and -0 must survive untouched,
// negatives collapse to +0.
float ExpectedRelu(float x) { return x < 0.0f ? 0.0f : x; }
} // namespace

class AutogradReLUForwardTest : public infini_train::test::InfiniTrainTest {};

// A single-case battery of NaN, -0, 0, +/-values, infinities and the fp32 extremes.
// NaN and -0 are asserted bit- or positionally because they have no ordering.
TEST_P(AutogradReLUForwardTest, ReLUForwardNaNZeroExtremes) {
    const std::vector<float> input_values{std::nanf(""),
                                          -0.0f,
                                          0.0f,
                                          1.5f,
                                          -2.5f,
                                          std::numeric_limits<float>::infinity(),
                                          -std::numeric_limits<float>::infinity(),
                                          1e-30f,
                                          -1e-30f,
                                          std::numeric_limits<float>::max(),
                                          -std::numeric_limits<float>::max()};
    auto input
        = std::make_shared<Tensor>(input_values.data(), std::vector<int64_t>{static_cast<int64_t>(input_values.size())},
                                   DataType::kFLOAT32, GetDevice());

    auto relu_fn = std::make_shared<autograd::ReLU>();
    auto result = relu_fn->Apply({input});
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0]->Dims(), input->Dims());

    const auto host_result = result[0]->To(Device());
    const float *out = static_cast<const float *>(host_result.DataPtr());

    for (size_t idx = 0; idx < input_values.size(); ++idx) {
        if (std::isnan(input_values[idx])) {
            EXPECT_TRUE(std::isnan(out[idx])) << "NaN position " << idx << " must stay NaN";
        } else if (std::signbit(input_values[idx]) && input_values[idx] == 0.0f) {
            EXPECT_EQ(out[idx], 0.0f) << "-0 position " << idx;
            EXPECT_TRUE(std::signbit(out[idx])) << "-0 signbit lost at position " << idx;
        } else {
            EXPECT_EQ(FloatBits(out[idx]), FloatBits(ExpectedRelu(input_values[idx])))
                << "finite bit mismatch at position " << idx;
        }
    }
}

TEST_P(AutogradReLUForwardTest, ReLUForwardModuleTwoD) {
    const std::vector<float> input_values{-1.0f, 0.0f, 2.0f, 3.5f, -4.25f, 0.5f};
    auto input
        = std::make_shared<Tensor>(input_values.data(), std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice());

    auto relu = std::make_shared<nn::ReLU>();
    auto result = (*relu)({input});
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0]->Dims(), (std::vector<int64_t>{2, 3}));
    test::ExpectTensorFloatEqual(result[0], {0.0f, 0.0f, 2.0f, 3.5f, 0.0f, 0.5f});
}

// ReLU is elementwise; the 4-D conv-output layout must keep its shape.
TEST_P(AutogradReLUForwardTest, ReLUForwardFourD) {
    std::vector<float> input_values;
    for (int idx = 0; idx < 2 * 2 * 4 * 4; ++idx) { input_values.push_back((idx * 7) % 13 / 3.0f - 2.0f); }
    auto input = std::make_shared<Tensor>(input_values.data(), std::vector<int64_t>{2, 2, 4, 4}, DataType::kFLOAT32,
                                          GetDevice());

    auto relu_fn = std::make_shared<autograd::ReLU>();
    auto result = relu_fn->Apply({input});
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0]->Dims(), (std::vector<int64_t>{2, 2, 4, 4}));

    const auto host_result = result[0]->To(Device());
    const float *out = static_cast<const float *>(host_result.DataPtr());
    for (size_t idx = 0; idx < input_values.size(); ++idx) {
        EXPECT_EQ(out[idx], ExpectedRelu(input_values[idx])) << "position " << idx;
    }
}

// An empty batch is a valid 0-element tensor: the kernel must not touch memory.
TEST_P(AutogradReLUForwardTest, ReLUForwardEmptyBatch) {
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{0, 3}, DataType::kFLOAT32, GetDevice());
    auto relu_fn = std::make_shared<autograd::ReLU>();
    auto result = relu_fn->Apply({input});
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{0, 3}));
    EXPECT_EQ(result[0]->NumElements(), 0);
}

// The op is declared FP32-only; other dtypes must be rejected loudly.
TEST_P(AutogradReLUForwardTest, ReLUForwardRejectsNonFloat) {
    ONLY_CPU();
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT16, GetDevice());
    EXPECT_DEATH(
        {
            auto relu_fn = std::make_shared<autograd::ReLU>();
            auto result = relu_fn->Apply({input});
            (void)result;
        },
        "");
}

INFINI_TRAIN_REGISTER_TEST(AutogradReLUForwardTest);
