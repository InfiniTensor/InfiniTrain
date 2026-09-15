#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autograd/linear.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class AutogradLinearBackwardTest : public infini_train::test::InfiniTrainTest {};

TEST_P(AutogradLinearBackwardTest, LinearBackward) {
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    input->Fill(1.0f);
    auto weight = std::make_shared<Tensor>(std::vector<int64_t>{4, 3}, DataType::kFLOAT32, GetDevice(), true);
    weight->Fill(1.0f);
    auto bias = std::make_shared<Tensor>(std::vector<int64_t>{4}, DataType::kFLOAT32, GetDevice(), true);
    bias->Fill(0.0f);
    auto linear_fn = std::make_shared<autograd::Linear>();
    auto result = linear_fn->Apply({input, weight, bias});
    const float grad_values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    auto grad = std::make_shared<Tensor>(grad_values, std::vector<int64_t>{2, 4}, DataType::kFLOAT32, GetDevice());
    auto grad_inputs = linear_fn->Backward({grad});
    ASSERT_EQ(grad_inputs.size(), 3);
    ASSERT_NE(grad_inputs[2], nullptr);

    auto bias_grad_cpu = grad_inputs[2]->To(Device());
    core::GetDeviceGuardImpl(GetDevice().type())->SynchronizeDevice(GetDevice());
    const auto *bias_grad = static_cast<const float *>(bias_grad_cpu.DataPtr());
    const float expected_bias_grad[] = {6.0f, 8.0f, 10.0f, 12.0f};
    for (int idx = 0; idx < 4; ++idx) { EXPECT_FLOAT_EQ(bias_grad[idx], expected_bias_grad[idx]); }
}

TEST_P(AutogradLinearBackwardTest, LinearBackwardNoBias) {
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    input->Fill(1.0f);
    auto weight = std::make_shared<Tensor>(std::vector<int64_t>{4, 3}, DataType::kFLOAT32, GetDevice(), true);
    weight->Fill(1.0f);
    auto linear_fn = std::make_shared<autograd::Linear>();
    auto result = linear_fn->Apply({input, weight});
    auto grad = std::make_shared<Tensor>(std::vector<int64_t>{2, 4}, DataType::kFLOAT32, GetDevice(), true);
    grad->Fill(1.0f);
    auto grad_inputs = linear_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 2);
}

// Non-square (bs=4, out_features=3) with distinct values per element: the bias
// gradient must be the column sum of grad_output ([18, 22, 26] below), which a
// row-sum misread would report as [6, 22, 38]. Constant-valued grad_output
// cannot distinguish the two reductions, so values must vary.
TEST_P(AutogradLinearBackwardTest, LinearBackwardValues) {
    const std::vector<float> input_values{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    const std::vector<float> weight_values{1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
    const std::vector<float> grad_values{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};

    auto input
        = std::make_shared<Tensor>(input_values.data(), std::vector<int64_t>{4, 2}, DataType::kFLOAT32, GetDevice());
    input->set_requires_grad(true);
    auto weight
        = std::make_shared<Tensor>(weight_values.data(), std::vector<int64_t>{3, 2}, DataType::kFLOAT32, GetDevice());
    weight->set_requires_grad(true);
    auto bias = std::make_shared<Tensor>(std::vector<int64_t>{3}, DataType::kFLOAT32, GetDevice(), true);
    bias->Fill(0.0f);
    auto grad
        = std::make_shared<Tensor>(grad_values.data(), std::vector<int64_t>{4, 3}, DataType::kFLOAT32, GetDevice());

    auto linear_fn = std::make_shared<autograd::Linear>();
    linear_fn->Apply({input, weight, bias});
    auto grad_inputs = linear_fn->Backward({grad});

    ASSERT_EQ(grad_inputs.size(), 3);
    ASSERT_NE(grad_inputs[0], nullptr);
    ASSERT_NE(grad_inputs[1], nullptr);
    ASSERT_NE(grad_inputs[2], nullptr);

    // grad_bias = grad_output.sum(dim=0) = [0+3+6+9, 1+4+7+10, 2+5+8+11]
    EXPECT_EQ(grad_inputs[2]->Dims(), (std::vector<int64_t>{3}));
    test::ExpectTensorFloatEqual(grad_inputs[2], {18.0f, 22.0f, 26.0f});
    // grad_input = grad_output * weight
    EXPECT_EQ(grad_inputs[0]->Dims(), (std::vector<int64_t>{4, 2}));
    test::ExpectTensorFloatEqual(grad_inputs[0], {2.0f, 3.0f, 8.0f, 9.0f, 14.0f, 15.0f, 20.0f, 21.0f});
    // grad_weight = grad_output^T * input
    EXPECT_EQ(grad_inputs[1]->Dims(), (std::vector<int64_t>{3, 2}));
    test::ExpectTensorFloatEqual(grad_inputs[1], {102.0f, 120.0f, 118.0f, 140.0f, 134.0f, 160.0f});
}

// Larger non-square shape with a deterministic pseudo-random pattern, checked
// against a double-precision host reference of the sample-dimension sum.
TEST_P(AutogradLinearBackwardTest, LinearBackwardBiasValues) {
    constexpr int64_t bs = 16;
    constexpr int64_t out_features = 7;
    constexpr int64_t in_features = 3;

    std::vector<float> grad_values(bs * out_features);
    for (size_t i = 0; i < grad_values.size(); ++i) { grad_values[i] = static_cast<float>((i * 13 + 7) % 23) - 11.0f; }

    auto input = std::make_shared<Tensor>(std::vector<int64_t>{bs, in_features}, DataType::kFLOAT32, GetDevice(), true);
    input->Fill(1.0f);
    auto weight = std::make_shared<Tensor>(std::vector<int64_t>{out_features, in_features}, DataType::kFLOAT32,
                                           GetDevice(), true);
    weight->Fill(1.0f);
    auto bias = std::make_shared<Tensor>(std::vector<int64_t>{out_features}, DataType::kFLOAT32, GetDevice(), true);
    bias->Fill(0.0f);
    auto grad = std::make_shared<Tensor>(grad_values.data(), std::vector<int64_t>{bs, out_features}, DataType::kFLOAT32,
                                         GetDevice());

    auto linear_fn = std::make_shared<autograd::Linear>();
    linear_fn->Apply({input, weight, bias});
    auto grad_inputs = linear_fn->Backward({grad});

    ASSERT_EQ(grad_inputs.size(), 3);
    std::vector<double> expected_bias(out_features, 0.0);
    for (int64_t i = 0; i < bs; ++i) {
        for (int64_t j = 0; j < out_features; ++j) { expected_bias[j] += grad_values[i * out_features + j]; }
    }
    const std::vector<float> expected_bias_f32(expected_bias.begin(), expected_bias.end());
    test::ExpectTensorNear(grad_inputs[2], expected_bias_f32, 1e-4f);
}

// The bf16 branch accumulates in fp32 and returns a promoted fp32 grad_bias;
// bf16 kernels are CUDA-only (the CPU Linear path is fp32-only).
TEST_P(AutogradLinearBackwardTest, LinearBackwardBiasBFloat16) {
    ONLY_CUDA();
    const std::vector<float> grad_values{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};

    auto grad_f32
        = std::make_shared<Tensor>(grad_values.data(), std::vector<int64_t>{4, 3}, DataType::kFLOAT32, GetDevice());
    auto grad = std::make_shared<Tensor>(grad_f32->To(DataType::kBFLOAT16));

    auto input_f32 = std::make_shared<Tensor>(std::vector<int64_t>{4, 2}, DataType::kFLOAT32, GetDevice(), true);
    input_f32->Fill(1.0f);
    auto input = std::make_shared<Tensor>(input_f32->To(DataType::kBFLOAT16));
    input->set_requires_grad(true);
    auto weight_f32 = std::make_shared<Tensor>(std::vector<int64_t>{3, 2}, DataType::kFLOAT32, GetDevice(), true);
    weight_f32->Fill(1.0f);
    auto weight = std::make_shared<Tensor>(weight_f32->To(DataType::kBFLOAT16));
    weight->set_requires_grad(true);
    auto bias = std::make_shared<Tensor>(std::vector<int64_t>{3}, DataType::kBFLOAT16, GetDevice(), true);
    bias->Fill(0.0f);

    auto linear_fn = std::make_shared<autograd::Linear>();
    linear_fn->Apply({input, weight, bias});
    auto grad_inputs = linear_fn->Backward({grad});

    ASSERT_EQ(grad_inputs.size(), 3);
    // bf16 inputs 0..11 are exact; the fp32 accumulation must reproduce [18, 22, 26].
    EXPECT_EQ(grad_inputs[2]->Dtype(), DataType::kFLOAT32);
    test::ExpectTensorFloatEqual(grad_inputs[2], {18.0f, 22.0f, 26.0f});
}

INFINI_TRAIN_REGISTER_TEST(AutogradLinearBackwardTest);
