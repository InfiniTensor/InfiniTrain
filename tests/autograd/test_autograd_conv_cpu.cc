#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autograd/conv.h"
#include "infini_train/include/nn/modules/conv.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class AutogradConvCPUTest : public infini_train::test::InfiniTrainTest {};

TEST_P(AutogradConvCPUTest, Conv2dForwardNoBias) {
    ONLY_CPU();
    // input 3x3 = 1..9, weight [[1, 0], [0, -1]], s=1, p=0 -> 2x2 of -4
    const float input_values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    const float weight_values[] = {1.0f, 0.0f, 0.0f, -1.0f};
    auto input
        = std::make_shared<Tensor>(input_values, std::vector<int64_t>{1, 1, 3, 3}, DataType::kFLOAT32, GetDevice());
    auto weight
        = std::make_shared<Tensor>(weight_values, std::vector<int64_t>{1, 1, 2, 2}, DataType::kFLOAT32, GetDevice());
    auto conv_fn = std::make_shared<autograd::Conv2d>(1, 0);
    auto result = conv_fn->Apply({input, weight});
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{1, 1, 2, 2}));
    test::ExpectTensorFloatEqual(result[0], {-4.0f, -4.0f, -4.0f, -4.0f});
}

TEST_P(AutogradConvCPUTest, Conv2dForwardWithBias) {
    ONLY_CPU();
    const float input_values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    const float weight_values[] = {1.0f, 0.0f, 0.0f, -1.0f};
    const float bias_values[] = {1.0f};
    auto input
        = std::make_shared<Tensor>(input_values, std::vector<int64_t>{1, 1, 3, 3}, DataType::kFLOAT32, GetDevice());
    auto weight
        = std::make_shared<Tensor>(weight_values, std::vector<int64_t>{1, 1, 2, 2}, DataType::kFLOAT32, GetDevice());
    auto bias = std::make_shared<Tensor>(bias_values, std::vector<int64_t>{1}, DataType::kFLOAT32, GetDevice());
    auto conv_fn = std::make_shared<autograd::Conv2d>(1, 0);
    auto result = conv_fn->Apply({input, weight, bias});
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{1, 1, 2, 2}));
    test::ExpectTensorFloatEqual(result[0], {-3.0f, -3.0f, -3.0f, -3.0f});
}

TEST_P(AutogradConvCPUTest, Conv2dForwardStridePadding) {
    ONLY_CPU();
    // input 4x4 = 1..16, 3x3 ones, s=2, p=1 -> 2x2 {14, 30, 57, 99}
    std::vector<float> input_values(16);
    for (int i = 0; i < 16; ++i) { input_values[i] = static_cast<float>(i + 1); }
    auto input = std::make_shared<Tensor>(input_values.data(), std::vector<int64_t>{1, 1, 4, 4}, DataType::kFLOAT32,
                                          GetDevice());
    auto weight = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 3, 3}, DataType::kFLOAT32, GetDevice());
    weight->Fill(1.0f);
    auto conv_fn = std::make_shared<autograd::Conv2d>(2, 1);
    auto result = conv_fn->Apply({input, weight});
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{1, 1, 2, 2}));
    test::ExpectTensorFloatEqual(result[0], {14.0f, 30.0f, 57.0f, 99.0f});
}

TEST_P(AutogradConvCPUTest, Conv2dForwardNonSquareKernel) {
    ONLY_CPU();
    // input 3x4 = 1..12, 2x3 ones (Kh != Kw), s=1, p=0 -> 2x2 {24, 30, 48, 54}
    std::vector<float> input_values(12);
    for (int i = 0; i < 12; ++i) { input_values[i] = static_cast<float>(i + 1); }
    auto input = std::make_shared<Tensor>(input_values.data(), std::vector<int64_t>{1, 1, 3, 4}, DataType::kFLOAT32,
                                          GetDevice());
    auto weight = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 2, 3}, DataType::kFLOAT32, GetDevice());
    weight->Fill(1.0f);
    auto conv_fn = std::make_shared<autograd::Conv2d>(1, 0);
    auto result = conv_fn->Apply({input, weight});
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{1, 1, 2, 2}));
    test::ExpectTensorFloatEqual(result[0], {24.0f, 30.0f, 48.0f, 54.0f});
}

TEST_P(AutogradConvCPUTest, Conv2dBackward) {
    ONLY_CPU();
    const float input_values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    const float weight_values[] = {1.0f, 0.0f, 0.0f, -1.0f};
    auto input
        = std::make_shared<Tensor>(input_values, std::vector<int64_t>{1, 1, 3, 3}, DataType::kFLOAT32, GetDevice())
              ->RequiresGrad();
    auto weight
        = std::make_shared<Tensor>(weight_values, std::vector<int64_t>{1, 1, 2, 2}, DataType::kFLOAT32, GetDevice())
              ->RequiresGrad();
    auto bias = std::make_shared<Tensor>(std::vector<int64_t>{1}, DataType::kFLOAT32, GetDevice(), true);
    bias->Fill(0.0f);
    auto conv_fn = std::make_shared<autograd::Conv2d>(1, 0);
    auto result = conv_fn->Apply({input, weight, bias});
    auto grad = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 2, 2}, DataType::kFLOAT32, GetDevice());
    grad->Fill(1.0f);
    auto grad_inputs = conv_fn->Backward({grad});
    ASSERT_EQ(grad_inputs.size(), 3);
    ASSERT_NE(grad_inputs[0], nullptr);
    ASSERT_NE(grad_inputs[1], nullptr);
    ASSERT_NE(grad_inputs[2], nullptr);
    EXPECT_EQ(grad_inputs[0]->Dims(), (std::vector<int64_t>{1, 1, 3, 3}));
    EXPECT_EQ(grad_inputs[1]->Dims(), (std::vector<int64_t>{1, 1, 2, 2}));
    EXPECT_EQ(grad_inputs[2]->Dims(), (std::vector<int64_t>{1}));
    test::ExpectTensorFloatEqual(grad_inputs[0], {1.0f, 1.0f, 0.0f, 1.0f, 0.0f, -1.0f, 0.0f, -1.0f, -1.0f});
    test::ExpectTensorFloatEqual(grad_inputs[1], {12.0f, 16.0f, 24.0f, 28.0f});
    test::ExpectTensorFloatEqual(grad_inputs[2], {4.0f});
}

TEST_P(AutogradConvCPUTest, Conv2dBackwardNoBias) {
    ONLY_CPU();
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 3, 3}, DataType::kFLOAT32, GetDevice(), true);
    input->Fill(1.0f);
    auto weight = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 2, 2}, DataType::kFLOAT32, GetDevice(), true);
    weight->Fill(1.0f);
    auto conv_fn = std::make_shared<autograd::Conv2d>(1, 0);
    auto result = conv_fn->Apply({input, weight});
    auto grad = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 2, 2}, DataType::kFLOAT32, GetDevice());
    grad->Fill(1.0f);
    auto grad_inputs = conv_fn->Backward({grad});
    ASSERT_EQ(grad_inputs.size(), 2);
    ASSERT_NE(grad_inputs[0], nullptr);
    ASSERT_NE(grad_inputs[1], nullptr);
    test::ExpectTensorFloatEqual(grad_inputs[0], {1.0f, 2.0f, 1.0f, 2.0f, 4.0f, 2.0f, 1.0f, 2.0f, 1.0f});
    test::ExpectTensorFloatEqual(grad_inputs[1], {4.0f, 4.0f, 4.0f, 4.0f});
}

TEST_P(AutogradConvCPUTest, Conv2dModuleForward) {
    ONLY_CPU();
    nn::Conv2d conv(1, 2, 3, 1, 1, true, GetDevice());
    EXPECT_TRUE(conv.has_bias());
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 4, 4}, DataType::kFLOAT32, GetDevice());
    input->Fill(1.0f);
    auto result = conv.Forward({input});
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{1, 2, 4, 4}));
}

TEST_P(AutogradConvCPUTest, Conv2dModuleNoBias) {
    ONLY_CPU();
    nn::Conv2d conv(1, 1, 2, 2, 0, false, GetDevice());
    EXPECT_FALSE(conv.has_bias());
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 4, 4}, DataType::kFLOAT32, GetDevice());
    input->Fill(1.0f);
    auto result = conv.Forward({input});
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{1, 1, 2, 2}));
}

INFINI_TRAIN_REGISTER_TEST(AutogradConvCPUTest);
