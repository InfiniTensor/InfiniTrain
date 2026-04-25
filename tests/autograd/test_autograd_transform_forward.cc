#include <gtest/gtest.h>

#include <vector>

#include "infini_train/include/autograd/misc.h"
#include "infini_train/include/autograd/transform.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"
#include "test_utils.h"

using namespace infini_train;

class AutogradTransformForwardTest : public infini_train::test::AutogradTestBase {};

TEST_P(AutogradTransformForwardTest, TransposeForward) {
    auto a = createTensor({2, 3}, 1.0f);
    auto transpose_fn = std::make_shared<autograd::Transpose>(0, 1);
    auto result = transpose_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{3, 2}));
}

TEST_P(AutogradTransformForwardTest, SliceForward) {
    auto a = createTensor({4, 4}, 1.0f);
    auto slice_fn = std::make_shared<autograd::Slice>(std::vector<int64_t>{1, 1}, std::vector<int64_t>{3, 3},
                                                      std::vector<int64_t>{1, 1});
    auto result = slice_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradTransformForwardTest, SplitForward) {
    auto a = createTensor({4, 4}, 1.0f);
    auto split_fn = std::make_shared<autograd::Split>(2, 0);
    auto result = split_fn->Apply({a});
    EXPECT_EQ(result.size(), 2);
}

TEST_P(AutogradTransformForwardTest, ConcatForward) {
    auto a = createTensor({2, 2}, 1.0f);
    auto b = createTensor({2, 2}, 2.0f);
    auto concat_fn = std::make_shared<autograd::Concat>(0);
    auto result = concat_fn->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{4, 2}));
}

TEST_P(AutogradTransformForwardTest, StackForward) {
    auto a = createTensor({2, 3}, 1.0f);
    auto b = createTensor({2, 3}, 2.0f);
    auto stack_fn = std::make_shared<autograd::Stack>(0);
    auto result = stack_fn->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{2, 2, 3}));
}

TEST_P(AutogradTransformForwardTest, TrilForward) {
    auto a = createTensor({3, 3}, 1.0f);
    auto tril_fn = std::make_shared<autograd::Tril>(0);
    auto result = tril_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradTransformForwardTest, TriuForward) {
    auto a = createTensor({3, 3}, 1.0f);
    auto triu_fn = std::make_shared<autograd::Triu>(0);
    auto result = triu_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

INFINI_TRAIN_REGISTER_TEST(AutogradTransformForwardTest);
