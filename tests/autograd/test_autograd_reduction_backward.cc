#include <gtest/gtest.h>

#include <vector>

#include "infini_train/include/tensor.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/autograd/reduction.h"
#include "test_utils.h"

using namespace infini_train;

class AutogradReductionBackwardTest : public infini_train::test::AutogradTestBaseP {};

TEST_P(AutogradReductionBackwardTest, SumBackward) {
    auto a = createTensor({2, 3}, 1.0f);
    auto sum_fn = std::make_shared<autograd::Sum>(1, false);
    auto result = sum_fn->Apply({a});
    auto grad = createTensor({2}, 1.0f);
    auto grad_inputs = sum_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 1);
}

TEST_P(AutogradReductionBackwardTest, MeanBackward) {
    auto a = createTensor({2, 3}, 1.0f);
    auto mean_fn = std::make_shared<autograd::Mean>(1, false);
    auto result = mean_fn->Apply({a});
    auto grad = createTensor({2}, 1.0f);
    auto grad_inputs = mean_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 1);
}

TEST_P(AutogradReductionBackwardTest, MaxBackward) {
    auto a = createTensor({2, 3}, 1.0f);
    auto max_fn = std::make_shared<autograd::Max>(1, false);
    auto result = max_fn->Apply({a});
    auto grad = createTensor({2}, 1.0f);
    auto grad_inputs = max_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 1);
}

TEST_P(AutogradReductionBackwardTest, MinBackward) {
    auto a = createTensor({2, 3}, 1.0f);
    auto min_fn = std::make_shared<autograd::Min>(1, false);
    auto result = min_fn->Apply({a});
    auto grad = createTensor({2}, 1.0f);
    auto grad_inputs = min_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 1);
}

TEST_P(AutogradReductionBackwardTest, SumBackwardKeepDim) {
    auto a = createTensor({2, 3}, 1.0f);
    auto sum_fn = std::make_shared<autograd::Sum>(1, true);
    auto result = sum_fn->Apply({a});
    auto grad = createTensor({2, 1}, 1.0f);
    auto grad_inputs = sum_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 1);
}

TEST_P(AutogradReductionBackwardTest, MeanBackwardKeepDim) {
    auto a = createTensor({2, 3}, 1.0f);
    auto mean_fn = std::make_shared<autograd::Mean>(1, true);
    auto result = mean_fn->Apply({a});
    auto grad = createTensor({2, 1}, 1.0f);
    auto grad_inputs = mean_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 1);
}

INFINI_TRAIN_REGISTER_TEST(AutogradReductionBackwardTest);
