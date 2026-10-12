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

// The CPU kernel is FP32-only; other dtypes must be rejected loudly there. (The CUDA kernels
// follow the elementwise dtype dispatch, which additionally covers bfloat16.)
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

// bfloat16 is CUDA-only and runs through the elementwise dispatch in both directions. The values
// are exactly representable in bf16, and the comparison is bitwise on the fp32 view of the
// results: `ExpectTensorFloatEqual` compares with EXPECT_FLOAT_EQ, which cannot tell +0.0 from
// -0.0, so the masked positions below are pinned through their bit patterns instead.
TEST_P(AutogradReLUForwardTest, ReLUForwardBackwardBFloat16) {
    SKIP_CPU();
    const std::vector<float> input_values{-2.0f, -0.5f, 0.0f, 1.5f, 4.0f, -8.0f};
    // Negative gradients at the masked positions pin the +0.0 (0x00000000, not 0x80000000) value.
    const std::vector<float> grad_values{-1.0f, 1.0f, -2.0f, 2.0f, 3.0f, -3.0f};
    const std::vector<float> expected_forward{0.0f, 0.0f, 0.0f, 1.5f, 4.0f, 0.0f};
    const std::vector<float> expected_grad{0.0f, 0.0f, 0.0f, 2.0f, 3.0f, 0.0f};

    auto input_f32
        = std::make_shared<Tensor>(input_values.data(), std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice());
    auto input = std::make_shared<Tensor>(input_f32->To(DataType::kBFLOAT16));
    auto grad_f32
        = std::make_shared<Tensor>(grad_values.data(), std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice());
    auto grad = std::make_shared<Tensor>(grad_f32->To(DataType::kBFLOAT16));

    auto relu_fn = std::make_shared<autograd::ReLU>();
    auto result = relu_fn->Apply({input});
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0]->Dtype(), DataType::kBFLOAT16);
    ASSERT_EQ(result[0]->Dims(), (std::vector<int64_t>{2, 3}));
    const auto forward_f32 = std::make_shared<Tensor>(result[0]->To(DataType::kFLOAT32));
    const auto forward_host = forward_f32->To(Device());
    const float *forward_out = static_cast<const float *>(forward_host.DataPtr());
    for (size_t idx = 0; idx < expected_forward.size(); ++idx) {
        EXPECT_EQ(FloatBits(forward_out[idx]), FloatBits(expected_forward[idx])) << "forward bit mismatch at " << idx;
    }

    auto grad_inputs = relu_fn->Backward({grad});
    ASSERT_EQ(grad_inputs.size(), 1);
    ASSERT_EQ(grad_inputs[0]->Dtype(), DataType::kBFLOAT16);
    ASSERT_EQ(grad_inputs[0]->Dims(), (std::vector<int64_t>{2, 3}));
    const auto grad_f32_host = std::make_shared<Tensor>(grad_inputs[0]->To(DataType::kFLOAT32));
    const auto grad_host = grad_f32_host->To(Device());
    const float *grad_out = static_cast<const float *>(grad_host.DataPtr());
    for (size_t idx = 0; idx < expected_grad.size(); ++idx) {
        EXPECT_EQ(FloatBits(grad_out[idx]), FloatBits(expected_grad[idx])) << "gradient bit mismatch at " << idx;
    }
}

INFINI_TRAIN_REGISTER_TEST(AutogradReLUForwardTest);
