#include <gtest/gtest.h>

#include <vector>

#include "infini_train/include/tensor.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/autograd/reduction.h"
#include "test_utils.h"

using namespace infini_train;

class AutogradReductionForwardTest : public infini_train::test::AutogradTestBaseP {};

TEST_P(AutogradReductionForwardTest, SumForward) {
    auto a = createTensor({2, 3}, 1.0f);
    auto sum_fn = std::make_shared<autograd::Sum>(1, false);
    auto result = sum_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradReductionForwardTest, MeanForward) {
    auto a = createTensor({2, 3}, 1.0f);
    auto mean_fn = std::make_shared<autograd::Mean>(1, false);
    auto result = mean_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradReductionForwardTest, MaxForward) {
    auto a = createTensor({2, 3}, 1.0f);
    auto max_fn = std::make_shared<autograd::Max>(1, false);
    auto result = max_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradReductionForwardTest, MinForward) {
    auto a = createTensor({2, 3}, 1.0f);
    auto min_fn = std::make_shared<autograd::Min>(1, false);
    auto result = min_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradReductionForwardTest, SumKeepDim) {
    auto a = createTensor({2, 3}, 1.0f);
    auto sum_fn = std::make_shared<autograd::Sum>(1, true);
    auto result = sum_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradReductionForwardTest, MeanKeepDim) {
    auto a = createTensor({2, 3}, 1.0f);
    auto mean_fn = std::make_shared<autograd::Mean>(1, true);
    auto result = mean_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

INFINI_TRAIN_REGISTER_TEST(AutogradReductionForwardTest);
