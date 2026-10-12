#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autograd/activations.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

namespace {
uint32_t FloatBits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

// Reference bits for the gradient w.r.t. the ReLU input, verified against torch 2.14 on CPU and
// CUDA: NaN inputs pass the incoming gradient through bit-for-bit, every other masked position
// (input <= 0, including -0.0) holds the literal +0.0 regardless of the sign or finiteness of
// that gradient. Note the distinction between +0.0 (0x00000000) and -0.0 (0x80000000).
uint32_t ExpectedReluGradBits(float x, float grad) {
    if (std::isnan(x)) {
        return FloatBits(grad);
    }
    return x <= 0.0f ? FloatBits(0.0f) : FloatBits(grad);
}
} // namespace

class AutogradReLUBackwardTest : public infini_train::test::InfiniTrainTest {};

// The gradients are deliberately negative, signed-zero, NaN and infinite at masked positions:
// scaling them by a 0/1 mask would yield -0.0 or NaN there instead of torch's +0.0.
TEST_P(AutogradReLUBackwardTest, ReLUBackwardNaNTransparentAndZeroMask) {
    const std::vector<float> input_values{std::nanf(""),
                                          -0.0f,
                                          0.0f,
                                          1.5f,
                                          -2.5f,
                                          std::numeric_limits<float>::infinity(),
                                          -std::numeric_limits<float>::infinity(),
                                          -1e-30f,
                                          1e-30f,
                                          std::numeric_limits<float>::max(),
                                          -std::numeric_limits<float>::max()};
    const std::vector<float> grad_values{-1.0f,
                                         -2.0f,
                                         -3.0f,
                                         -4.0f,
                                         std::numeric_limits<float>::infinity(),
                                         -std::numeric_limits<float>::infinity(),
                                         std::nanf(""),
                                         -0.0f,
                                         0.0f,
                                         1.0f,
                                         -1.0f};

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
        EXPECT_EQ(FloatBits(out[idx]), ExpectedReluGradBits(input_values[idx], grad_values[idx]))
            << "gradient bit mismatch at position " << idx << " (input " << input_values[idx] << ", grad "
            << grad_values[idx] << ")";
    }
}

// Two-dimensional case with a mixed-sign input and negative gradients at the masked positions,
// checking mask and passthrough jointly.
TEST_P(AutogradReLUBackwardTest, ReLUBackwardTwoD) {
    const std::vector<float> input_values{-1.0f, 2.0f, -0.0f, 4.0f, -5.0f, 6.0f};
    const std::vector<float> grad_values{-1.0f, 2.0f, -3.0f, 4.0f, -5.0f, 6.0f};
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

    // The masked positions must hold +0.0 bit-exactly; ExpectTensorFloatEqual cannot tell the
    // signed zeros apart, so pin them separately.
    const auto host_grad_input = grad_inputs[0]->To(Device());
    const float *out = static_cast<const float *>(host_grad_input.DataPtr());
    for (size_t idx = 0; idx < input_values.size(); ++idx) {
        EXPECT_EQ(FloatBits(out[idx]), ExpectedReluGradBits(input_values[idx], grad_values[idx])) << "position " << idx;
    }
}

INFINI_TRAIN_REGISTER_TEST(AutogradReLUBackwardTest);
