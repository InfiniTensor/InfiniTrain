#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autograd/softmax.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class AutogradSoftmaxBackwardTest : public infini_train::test::InfiniTrainTest {};

TEST_P(AutogradSoftmaxBackwardTest, SoftmaxBackward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto softmax_fn = std::make_shared<autograd::Softmax>(1);
    auto result = softmax_fn->Apply({a});
    std::vector<float> grad_values{1.0f, 2.0f, 4.0f, 4.0f, 2.0f, 1.0f};
    auto grad
        = std::make_shared<Tensor>(grad_values.data(), std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice());
    auto grad_inputs = softmax_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 1);
    test::ExpectTensorNear(grad_inputs[0],
                           {-4.0f / 9.0f, -1.0f / 9.0f, 5.0f / 9.0f, 5.0f / 9.0f, -1.0f / 9.0f, -4.0f / 9.0f}, 1e-6f);
}

TEST_P(AutogradSoftmaxBackwardTest, SoftmaxBackwardDim0) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{4, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto softmax_fn = std::make_shared<autograd::Softmax>(0);
    auto result = softmax_fn->Apply({a});
    std::vector<float> grad_values{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    auto grad
        = std::make_shared<Tensor>(grad_values.data(), std::vector<int64_t>{4, 3}, DataType::kFLOAT32, GetDevice());
    auto grad_inputs = softmax_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 1);
    test::ExpectTensorNear(
        grad_inputs[0],
        {-1.125f, -1.125f, -1.125f, -0.375f, -0.375f, -0.375f, 0.375f, 0.375f, 0.375f, 1.125f, 1.125f, 1.125f}, 1e-6f);
}

INFINI_TRAIN_REGISTER_TEST(AutogradSoftmaxBackwardTest);
