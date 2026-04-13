#include <gtest/gtest.h>

#include <vector>

#include "infini_train/include/tensor.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/autograd/normalization.h"
#include "test_utils.h"

using namespace infini_train;

class AutogradNormalizationForwardTest : public infini_train::test::AutogradTestBaseP {};

TEST_P(AutogradNormalizationForwardTest, LayerNormForward) {
    auto a = createTensor({2, 3, 4}, 1.0f);
    auto weight = createTensor({4}, 1.0f);
    auto bias = createTensor({4}, 0.0f);
    auto layernorm_fn = std::make_shared<autograd::LayerNorm>(1e-5f);
    auto result = layernorm_fn->Apply({a, weight, bias});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradNormalizationForwardTest, LayerNormZeroBias) {
    auto a = createTensor({2, 3, 4}, 1.0f);
    auto weight = createTensor({4}, 1.0f);
    auto bias = createTensor({4}, 0.0f);
    auto layernorm_fn = std::make_shared<autograd::LayerNorm>(1e-5f);
    auto result = layernorm_fn->Apply({a, weight, bias});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradNormalizationForwardTest, LayerNormThreeDim) {
    auto a = createTensor({2, 1, 4}, 1.0f);
    auto weight = createTensor({4}, 1.0f);
    auto bias = createTensor({4}, 0.0f);
    auto layernorm_fn = std::make_shared<autograd::LayerNorm>(1e-5f);
    auto result = layernorm_fn->Apply({a, weight, bias});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{2, 1, 4}));
}

INFINI_TRAIN_REGISTER_TEST(AutogradNormalizationForwardTest);
