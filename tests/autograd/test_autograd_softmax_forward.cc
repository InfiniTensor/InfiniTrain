#include <gtest/gtest.h>

#include <vector>

#include "infini_train/include/tensor.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/autograd/softmax.h"
#include "test_utils.h"

using namespace infini_train;

class AutogradSoftmaxForwardTest : public infini_train::test::AutogradTestBaseP {};

TEST_P(AutogradSoftmaxForwardTest, SoftmaxForward) {
    auto a = createTensor({2, 3}, 1.0f);
    auto softmax_fn = std::make_shared<autograd::Softmax>(1);
    auto result = softmax_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{2, 3}));
}

TEST_P(AutogradSoftmaxForwardTest, SoftmaxDim0) {
    auto a = createTensor({4, 3}, 1.0f);
    auto softmax_fn = std::make_shared<autograd::Softmax>(0);
    auto result = softmax_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{4, 3}));
}

TEST_P(AutogradSoftmaxForwardTest, SoftmaxLastDim) {
    auto a = createTensor({2, 3, 4}, 1.0f);
    auto softmax_fn = std::make_shared<autograd::Softmax>(2);
    auto result = softmax_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{2, 3, 4}));
}

INFINI_TRAIN_REGISTER_TEST(AutogradSoftmaxForwardTest);
