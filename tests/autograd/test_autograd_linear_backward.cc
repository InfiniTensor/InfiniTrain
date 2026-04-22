#include <gtest/gtest.h>

#include <vector>

#include "infini_train/include/autograd/linear.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"
#include "test_utils.h"

using namespace infini_train;

class AutogradLinearBackwardTest : public infini_train::test::AutogradTestBase {};

TEST_P(AutogradLinearBackwardTest, LinearBackward) {
    auto input = createTensor({2, 3}, 1.0f);
    auto weight = createTensor({4, 3}, 1.0f);
    auto bias = createTensor({4}, 0.0f);
    auto linear_fn = std::make_shared<autograd::Linear>();
    auto result = linear_fn->Apply({input, weight, bias});
    auto grad = createTensor({2, 4}, 1.0f);
    auto grad_inputs = linear_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 3);
}

TEST_P(AutogradLinearBackwardTest, LinearBackwardNoBias) {
    auto input = createTensor({2, 3}, 1.0f);
    auto weight = createTensor({4, 3}, 1.0f);
    auto linear_fn = std::make_shared<autograd::Linear>();
    auto result = linear_fn->Apply({input, weight});
    auto grad = createTensor({2, 4}, 1.0f);
    auto grad_inputs = linear_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 2);
}

INFINI_TRAIN_REGISTER_TEST(AutogradLinearBackwardTest);
