#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autocast.h"
#include "infini_train/include/autograd/normalization.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class AutogradNormalizationForwardTest : public infini_train::test::InfiniTrainTest {};

TEST_P(AutogradNormalizationForwardTest, LayerNormForward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3, 4}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto weight = std::make_shared<Tensor>(std::vector<int64_t>{4}, DataType::kFLOAT32, GetDevice(), true);
    weight->Fill(1.0f);
    auto bias = std::make_shared<Tensor>(std::vector<int64_t>{4}, DataType::kFLOAT32, GetDevice(), true);
    bias->Fill(0.0f);
    auto layernorm_fn = std::make_shared<autograd::LayerNorm>(1e-5f);
    auto result = layernorm_fn->Apply({a, weight, bias});
    EXPECT_EQ(result.size(), 3);
    EXPECT_FALSE(result[1]->requires_grad());
    EXPECT_EQ(result[1]->grad_fn(), nullptr);
    EXPECT_FALSE(result[2]->requires_grad());
    EXPECT_EQ(result[2]->grad_fn(), nullptr);
}

TEST_P(AutogradNormalizationForwardTest, LayerNormZeroBias) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3, 4}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto weight = std::make_shared<Tensor>(std::vector<int64_t>{4}, DataType::kFLOAT32, GetDevice(), true);
    weight->Fill(1.0f);
    auto bias = std::make_shared<Tensor>(std::vector<int64_t>{4}, DataType::kFLOAT32, GetDevice(), true);
    bias->Fill(0.0f);
    auto layernorm_fn = std::make_shared<autograd::LayerNorm>(1e-5f);
    auto result = layernorm_fn->Apply({a, weight, bias});
    EXPECT_EQ(result.size(), 3);
    EXPECT_FALSE(result[1]->requires_grad());
    EXPECT_EQ(result[1]->grad_fn(), nullptr);
    EXPECT_FALSE(result[2]->requires_grad());
    EXPECT_EQ(result[2]->grad_fn(), nullptr);
}

TEST_P(AutogradNormalizationForwardTest, LayerNormThreeDim) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 1, 4}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto weight = std::make_shared<Tensor>(std::vector<int64_t>{4}, DataType::kFLOAT32, GetDevice(), true);
    weight->Fill(1.0f);
    auto bias = std::make_shared<Tensor>(std::vector<int64_t>{4}, DataType::kFLOAT32, GetDevice(), true);
    bias->Fill(0.0f);
    auto layernorm_fn = std::make_shared<autograd::LayerNorm>(1e-5f);
    auto result = layernorm_fn->Apply({a, weight, bias});
    EXPECT_EQ(result.size(), 3);
    EXPECT_FALSE(result[1]->requires_grad());
    EXPECT_EQ(result[1]->grad_fn(), nullptr);
    EXPECT_FALSE(result[2]->requires_grad());
    EXPECT_EQ(result[2]->grad_fn(), nullptr);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{2, 1, 4}));
}

TEST_P(AutogradNormalizationForwardTest, RMSNormForward) {
    const std::vector<int64_t> input_dims{1, 2, 4};
    std::vector<float> input_values{-1.5f, -0.5f, 0.5f, 1.5f, -3.0f, -1.0f, 1.0f, 3.0f};
    std::vector<float> weight_values{1.0f, 0.5f, -1.0f, 2.0f};
    auto input = std::make_shared<Tensor>(input_values.data(), input_dims, DataType::kFLOAT32, GetDevice());
    auto weight
        = std::make_shared<Tensor>(weight_values.data(), std::vector<int64_t>{4}, DataType::kFLOAT32, GetDevice());

    auto rmsnorm_fn = std::make_shared<autograd::RMSNorm>(1e-5f);
    auto result = rmsnorm_fn->Apply({input, weight});
    ASSERT_EQ(result.size(), 2);
    EXPECT_FALSE(result[1]->requires_grad());
    EXPECT_EQ(result[1]->grad_fn(), nullptr);
    test::ExpectTensorNear(
        result[0],
        {-1.34163547f, -0.22360590f, -0.44721180f, 2.68327093f, -1.34163940f, -0.22360657f, -0.44721314f, 2.68327880f},
        1e-5f);
    // rstd is the normalization statistic kept for the backward pass: 1/sqrt(mean(x^2) + eps) per row.
    test::ExpectTensorNear(result[1], {0.89442360f, 0.44721314f}, 1e-5f);
}

TEST_P(AutogradNormalizationForwardTest, RMSNormTwoDimInput) {
    // The fused op must stay rank-agnostic: flattened [rows, embed_dim] inputs work like the
    // composite path (Mean(-1)/Pow/Rsqrt/Mul), including the rstd shape (leading dims only).
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{8, 4}, DataType::kFLOAT32, GetDevice(), true);
    input->Fill(2.0f);
    auto weight = std::make_shared<Tensor>(std::vector<int64_t>{4}, DataType::kFLOAT32, GetDevice(), true);
    weight->Fill(1.0f);
    auto rmsnorm_fn = std::make_shared<autograd::RMSNorm>(1e-5f);
    auto result = rmsnorm_fn->Apply({input, weight});
    ASSERT_EQ(result.size(), 2);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{8, 4}));
    EXPECT_EQ(result[1]->Dims(), (std::vector<int64_t>{8}));
    test::ExpectTensorNear(result[0], 0.99999875f, 1e-5f);
    test::ExpectTensorNear(result[1], 0.49999937f, 1e-5f);
}

TEST_P(AutogradNormalizationForwardTest, RMSNormAutocastCastsInputToFP32) {
    SKIP_CPU();
    // RMSNorm is registered as a kFP32 op: under a bf16 autocast context the bf16 activation is
    // promoted to fp32 before the fused kernel runs, so the input/weight dtype CHECK holds and
    // both outputs come back as fp32.
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{1, 2, 4}, DataType::kBFLOAT16, GetDevice());
    input->Fill(2.0f);
    auto weight = std::make_shared<Tensor>(std::vector<int64_t>{4}, DataType::kFLOAT32, GetDevice());
    weight->Fill(1.0f);

    AutocastGuard guard(GetDevice().type(), DataType::kBFLOAT16);
    auto rmsnorm_fn = std::make_shared<autograd::RMSNorm>(1e-5f);
    auto result = rmsnorm_fn->Apply({input, weight});
    ASSERT_EQ(result.size(), 2);
    EXPECT_EQ(result[0]->Dtype(), DataType::kFLOAT32);
    EXPECT_EQ(result[1]->Dtype(), DataType::kFLOAT32);
    test::ExpectTensorNear(result[0], 0.99999875f, 1e-5f);
    test::ExpectTensorNear(result[1], 0.49999937f, 1e-5f);
}

INFINI_TRAIN_REGISTER_TEST(AutogradNormalizationForwardTest);
