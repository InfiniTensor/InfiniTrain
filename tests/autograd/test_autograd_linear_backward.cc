#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autograd/linear.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class AutogradLinearBackwardTest : public infini_train::test::InfiniTrainTest {};

TEST_P(AutogradLinearBackwardTest, LinearBackward) {
    const std::vector<float> input_data{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto input
        = std::make_shared<Tensor>(input_data.data(), std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice());
    input->set_requires_grad(true);
    const std::vector<float> weight_data{
        1.0f, 0.0f, -1.0f, 2.0f, 1.0f, 0.0f, 0.0f, 1.0f, 2.0f, -1.0f, 2.0f, 1.0f,
    };
    auto weight
        = std::make_shared<Tensor>(weight_data.data(), std::vector<int64_t>{4, 3}, DataType::kFLOAT32, GetDevice());
    weight->set_requires_grad(true);
    const std::vector<float> bias_data{0.0f, 0.0f, 0.0f, 0.0f};
    auto bias = std::make_shared<Tensor>(bias_data.data(), std::vector<int64_t>{4}, DataType::kFLOAT32, GetDevice());
    bias->set_requires_grad(true);
    auto linear_fn = std::make_shared<autograd::Linear>();
    auto result = linear_fn->Apply({input, weight, bias});
    const std::vector<float> grad_data{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    auto grad = std::make_shared<Tensor>(grad_data.data(), std::vector<int64_t>{2, 4}, DataType::kFLOAT32, GetDevice());
    auto grad_inputs = linear_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 3);
    test::ExpectTensorFloatEqual(grad_inputs[0], std::vector<float>{1.0f, 13.0f, 9.0f, 9.0f, 29.0f, 17.0f});
    test::ExpectTensorFloatEqual(grad_inputs[1], std::vector<float>{21.0f, 27.0f, 33.0f, 26.0f, 34.0f, 42.0f, 31.0f,
                                                                    41.0f, 51.0f, 36.0f, 48.0f, 60.0f});
    test::ExpectTensorFloatEqual(grad_inputs[2], std::vector<float>{6.0f, 8.0f, 10.0f, 12.0f});
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
    test::ExpectTensorFloatEqual(grad_inputs[0], 4.0f);
    test::ExpectTensorFloatEqual(grad_inputs[1], 2.0f);
}

INFINI_TRAIN_REGISTER_TEST(AutogradLinearBackwardTest);
