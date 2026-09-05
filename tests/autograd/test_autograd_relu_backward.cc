#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autograd/activations.h"
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

// grad respect to input: NaN passes grad through, 0/-0/negative mask to 0, positive keeps grad.
float ExpectedReluGrad(float x, float grad) { return x <= 0.0f ? 0.0f : grad; }
} // namespace

class AutogradReLUBackwardTest : public infini_train::test::InfiniTrainTest {};

TEST_P(AutogradReLUBackwardTest, ReLUBackwardNaNTransparentAndZeroMask) {
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
    const std::vector<float> grad_values{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};

    auto input
        = std::make_shared<Tensor>(input_values.data(), std::vector<int64_t>{static_cast<int64_t>(input_values.size())},
                                   DataType::kFLOAT32, GetDevice());
    auto grad
        = std::make_shared<Tensor>(grad_values.data(), std::vector<int64_t>{static_cast<int64_t>(grad_values.size())},
                                   DataType::kFLOAT32, GetDevice());

    auto relu_fn = std::make_shared<autograd::ReLU>();
    auto result = relu_fn->Apply({input});
    ASSERT_EQ(result.size(), 1);
    auto grad_inputs = relu_fn->Backward({grad});
    ASSERT_EQ(grad_inputs.size(), 1);
    ASSERT_EQ(grad_inputs[0]->Dims(), input->Dims());

    const auto host_grad_input = grad_inputs[0]->To(Device());
    const float *out = static_cast<const float *>(host_grad_input.DataPtr());

    for (size_t idx = 0; idx < input_values.size(); ++idx) {
        const float expected = ExpectedReluGrad(input_values[idx], grad_values[idx]);
        if (std::isnan(input_values[idx])) {
            // The gradient must be passed through bit-for-bit, not zeroed.
            EXPECT_EQ(FloatBits(out[idx]), FloatBits(grad_values[idx])) << "NaN gradient must pass through at " << idx;
        } else {
            EXPECT_EQ(FloatBits(out[idx]), FloatBits(expected)) << "gradient bit mismatch at position " << idx;
        }
    }
}

// Two-dimensional case with a mixed-sign input, checking mask and passthrough jointly.
TEST_P(AutogradReLUBackwardTest, ReLUBackwardTwoD) {
    const std::vector<float> input_values{-1.0f, 2.0f, -0.0f, 4.0f, -5.0f, 6.0f};
    const std::vector<float> grad_values{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto input
        = std::make_shared<Tensor>(input_values.data(), std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice());
    auto grad
        = std::make_shared<Tensor>(grad_values.data(), std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice());

    auto relu_fn = std::make_shared<autograd::ReLU>();
    auto result = relu_fn->Apply({input});
    ASSERT_EQ(result.size(), 1);
    auto grad_inputs = relu_fn->Backward({grad});
    ASSERT_EQ(grad_inputs.size(), 1);
    ASSERT_EQ(grad_inputs[0]->Dims(), (std::vector<int64_t>{2, 3}));
    test::ExpectTensorFloatEqual(grad_inputs[0], {0.0f, 2.0f, 0.0f, 4.0f, 0.0f, 6.0f});
}

// Gradients entering at the masked (zero) positions must stay 0 regardless of the
// incoming grad value, and vice versa for the active positions.
TEST_P(AutogradReLUBackwardTest, ReLUBackwardGradValuesIndependentOfMask) {
    const std::vector<float> input_values{-3.0f, -0.5f, -0.0f, 0.0f, 1.0f, 7.0f};
    const std::vector<float> grad_values{-2.0f, 100.0f, -5.0f, 3.0f, -1.0f, 0.25f};
    auto input
        = std::make_shared<Tensor>(input_values.data(), std::vector<int64_t>{6}, DataType::kFLOAT32, GetDevice());
    auto grad = std::make_shared<Tensor>(grad_values.data(), std::vector<int64_t>{6}, DataType::kFLOAT32, GetDevice());

    auto relu_fn = std::make_shared<autograd::ReLU>();
    auto result = relu_fn->Apply({input});
    ASSERT_EQ(result.size(), 1);
    auto grad_inputs = relu_fn->Backward({grad});
    ASSERT_EQ(grad_inputs.size(), 1);

    const auto host_grad_input = grad_inputs[0]->To(Device());
    const float *out = static_cast<const float *>(host_grad_input.DataPtr());
    for (size_t idx = 0; idx < input_values.size(); ++idx) {
        EXPECT_EQ(out[idx], ExpectedReluGrad(input_values[idx], grad_values[idx])) << "position " << idx;
    }
}

INFINI_TRAIN_REGISTER_TEST(AutogradReLUBackwardTest);
