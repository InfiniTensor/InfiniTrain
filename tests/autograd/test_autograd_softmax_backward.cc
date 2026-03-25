#include <gtest/gtest.h>

#include <vector>

#include "infini_train/include/tensor.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/autograd/softmax.h"
#include "test_utils.h"

using namespace infini_train;

class AutogradSoftmaxBackwardTest : public infini_train::test::AutogradTestBase {};

TEST_F(AutogradSoftmaxBackwardTest, SoftmaxBackward) {
    auto a = createTensor({2, 3}, 1.0f);
    auto softmax_fn = std::make_shared<autograd::Softmax>(1);
    auto result = softmax_fn->Apply({a});
    auto grad = createTensor({2, 3}, 1.0f);
    auto grad_inputs = softmax_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 1);
}

TEST_F(AutogradSoftmaxBackwardTest, SoftmaxBackwardDim0) {
    auto a = createTensor({4, 3}, 1.0f);
    auto softmax_fn = std::make_shared<autograd::Softmax>(0);
    auto result = softmax_fn->Apply({a});
    auto grad = createTensor({4, 3}, 1.0f);
    auto grad_inputs = softmax_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 1);
}
