#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autograd/normalization.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class AutogradNormalizationBackwardTest : public infini_train::test::InfiniTrainTest {};

TEST_P(AutogradNormalizationBackwardTest, LayerNormBackward) {
    const std::vector<int64_t> input_dims{1, 2, 4};
    std::vector<float> input_values{-1.5f, -0.5f, 0.5f, 1.5f, -3.0f, -1.0f, 1.0f, 3.0f};
    std::vector<float> weight_values{1.0f, 0.5f, -1.0f, 2.0f};
    std::vector<float> bias_values{0.5f, -0.5f, 1.0f, -1.0f};
    auto input = std::make_shared<Tensor>(input_values.data(), input_dims, DataType::kFLOAT32, GetDevice());
    auto weight
        = std::make_shared<Tensor>(weight_values.data(), std::vector<int64_t>{4}, DataType::kFLOAT32, GetDevice());
    auto bias = std::make_shared<Tensor>(bias_values.data(), std::vector<int64_t>{4}, DataType::kFLOAT32, GetDevice());

    auto layernorm_fn = std::make_shared<autograd::LayerNorm>(1e-5f);
    auto result = layernorm_fn->Apply({input, weight, bias});
    ASSERT_EQ(result.size(), 3);
    test::ExpectTensorNear(result[1], {0.0f, 0.0f}, 1e-5f);
    test::ExpectTensorNear(result[2], {0.89442366f, 0.44721317f}, 1e-5f);

    std::vector<float> grad_values{1.0f, 2.0f, 3.0f, 4.0f, 0.5f, -1.0f, 2.0f, -0.5f};
    auto grad = std::make_shared<Tensor>(grad_values.data(), input_dims, DataType::kFLOAT32, GetDevice());
    auto grad_inputs = layernorm_fn->Backward({grad});
    ASSERT_EQ(grad_inputs.size(), 3);
    test::ExpectTensorNear(
        grad_inputs[0],
        {1.60994446f, 0.08943637f, -5.00876665f, 3.30938578f, 0.15652537f, -0.02236041f, -0.42485276f, 0.29068780f},
        1e-5f);
    test::ExpectTensorNear(grad_inputs[1], {-2.01245522f, -0.44721049f, 2.23606181f, 4.69572210f}, 1e-5f);
    test::ExpectTensorNear(grad_inputs[2], {1.5f, 1.0f, 5.0f, 3.5f}, 1e-5f);
}

TEST_P(AutogradNormalizationBackwardTest, LayerNormBackwardZeroBias) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3, 4}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto weight = std::make_shared<Tensor>(std::vector<int64_t>{4}, DataType::kFLOAT32, GetDevice(), true);
    weight->Fill(1.0f);
    auto bias = std::make_shared<Tensor>(std::vector<int64_t>{4}, DataType::kFLOAT32, GetDevice(), true);
    bias->Fill(0.0f);
    auto layernorm_fn = std::make_shared<autograd::LayerNorm>(1e-5f);
    auto result = layernorm_fn->Apply({a, weight, bias});
    auto grad = std::make_shared<Tensor>(std::vector<int64_t>{2, 3, 4}, DataType::kFLOAT32, GetDevice(), true);
    grad->Fill(1.0f);
    auto grad_inputs = layernorm_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 3);
}

INFINI_TRAIN_REGISTER_TEST(AutogradNormalizationBackwardTest);
