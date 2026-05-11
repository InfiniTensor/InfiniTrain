#include <vector>

#include "infini_train/include/autograd/softmax.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"
#include "tests/common/test_utils.h"
#include "gtest/gtest.h"

using namespace infini_train;

class AutogradSoftmaxBackwardTest : public infini_train::test::InfiniTrainTest {};

TEST_P(AutogradSoftmaxBackwardTest, SoftmaxBackward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto softmax_fn = std::make_shared<autograd::Softmax>(1);
    auto result = softmax_fn->Apply({a});
    auto grad = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    grad->Fill(1.0f);
    auto grad_inputs = softmax_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 1);
}

TEST_P(AutogradSoftmaxBackwardTest, SoftmaxBackwardDim0) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{4, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto softmax_fn = std::make_shared<autograd::Softmax>(0);
    auto result = softmax_fn->Apply({a});
    auto grad = std::make_shared<Tensor>(std::vector<int64_t>{4, 3}, DataType::kFLOAT32, GetDevice(), true);
    grad->Fill(1.0f);
    auto grad_inputs = softmax_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 1);
}

INFINI_TRAIN_REGISTER_TEST(AutogradSoftmaxBackwardTest);
