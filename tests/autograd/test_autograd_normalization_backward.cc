#include <gtest/gtest.h>

#include <vector>

#include "infini_train/include/tensor.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/autograd/normalization.h"
#include "test_utils.h"

using namespace infini_train;

class AutogradNormalizationBackwardTest : public infini_train::test::AutogradTestBase {};

TEST_F(AutogradNormalizationBackwardTest, LayerNormBackward) {
    auto a = createTensor({2, 3, 4}, 1.0f);
    auto weight = createTensor({4}, 1.0f);
    auto bias = createTensor({4}, 0.0f);
    auto layernorm_fn = std::make_shared<autograd::LayerNorm>(1e-5f);
    auto result = layernorm_fn->Apply({a, weight, bias});
    auto grad = createTensor({2, 3, 4}, 1.0f);
    auto grad_inputs = layernorm_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 3);
}

TEST_F(AutogradNormalizationBackwardTest, LayerNormBackwardNoBias) {
    auto a = createTensor({2, 3, 4}, 1.0f);
    auto weight = createTensor({4}, 1.0f);
    auto layernorm_fn = std::make_shared<autograd::LayerNorm>(1e-5f);
    auto result = layernorm_fn->Apply({a, weight});
    auto grad = createTensor({2, 3, 4}, 1.0f);
    auto grad_inputs = layernorm_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 2);
}
