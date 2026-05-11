#include <vector>

#include "infini_train/include/autograd/softmax.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"
#include "tests/common/test_utils.h"
#include "gtest/gtest.h"

using namespace infini_train;

class AutogradSoftmaxForwardTest : public infini_train::test::InfiniTrainTest {};

TEST_P(AutogradSoftmaxForwardTest, SoftmaxForward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto softmax_fn = std::make_shared<autograd::Softmax>(1);
    auto result = softmax_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{2, 3}));
}

TEST_P(AutogradSoftmaxForwardTest, SoftmaxDim0) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{4, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto softmax_fn = std::make_shared<autograd::Softmax>(0);
    auto result = softmax_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{4, 3}));
}

TEST_P(AutogradSoftmaxForwardTest, SoftmaxLastDim) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3, 4}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto softmax_fn = std::make_shared<autograd::Softmax>(2);
    auto result = softmax_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{2, 3, 4}));
}

INFINI_TRAIN_REGISTER_TEST(AutogradSoftmaxForwardTest);
