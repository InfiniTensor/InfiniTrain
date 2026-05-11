#include <vector>

#include "infini_train/include/autograd/reduction.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"
#include "tests/common/test_utils.h"
#include "gtest/gtest.h"

using namespace infini_train;

class AutogradReductionForwardTest : public infini_train::test::InfiniTrainTest {};

TEST_P(AutogradReductionForwardTest, SumForward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto sum_fn = std::make_shared<autograd::Sum>(1, false);
    auto result = sum_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradReductionForwardTest, MeanForward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto mean_fn = std::make_shared<autograd::Mean>(1, false);
    auto result = mean_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradReductionForwardTest, MaxForward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto max_fn = std::make_shared<autograd::Max>(1, false);
    auto result = max_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradReductionForwardTest, MinForward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto min_fn = std::make_shared<autograd::Min>(1, false);
    auto result = min_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradReductionForwardTest, SumKeepDim) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto sum_fn = std::make_shared<autograd::Sum>(1, true);
    auto result = sum_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradReductionForwardTest, MeanKeepDim) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto mean_fn = std::make_shared<autograd::Mean>(1, true);
    auto result = mean_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

INFINI_TRAIN_REGISTER_TEST(AutogradReductionForwardTest);
