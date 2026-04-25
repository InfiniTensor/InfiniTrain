#include <gtest/gtest.h>

#include <vector>

#include "infini_train/include/autograd/linear.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"
#include "test_utils.h"

using namespace infini_train;

class AutogradLinearForwardTest : public infini_train::test::AutogradTestBase {};

TEST_P(AutogradLinearForwardTest, LinearForward) {
    auto input = createTensor({2, 3}, 1.0f);
    auto weight = createTensor({4, 3}, 1.0f);
    auto bias = createTensor({4}, 0.0f);
    auto linear_fn = std::make_shared<autograd::Linear>();
    auto result = linear_fn->Apply({input, weight, bias});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{2, 4}));
}

TEST_P(AutogradLinearForwardTest, LinearNoBias) {
    auto input = createTensor({2, 3}, 1.0f);
    auto weight = createTensor({4, 3}, 1.0f);
    auto linear_fn = std::make_shared<autograd::Linear>();
    auto result = linear_fn->Apply({input, weight});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{2, 4}));
}

TEST_P(AutogradLinearForwardTest, LinearBatch) {
    auto input = createTensor({32, 128}, 1.0f);
    auto weight = createTensor({64, 128}, 1.0f);
    auto bias = createTensor({64}, 0.0f);
    auto linear_fn = std::make_shared<autograd::Linear>();
    auto result = linear_fn->Apply({input, weight, bias});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{32, 64}));
}

INFINI_TRAIN_REGISTER_TEST(AutogradLinearForwardTest);
