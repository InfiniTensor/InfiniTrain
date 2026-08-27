#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class OptimizerParameterNamesTest : public test::InfiniTrainTest {};

TEST_P(OptimizerParameterNamesTest, AdamStateDictUsesStableParameterNames) {
    auto first = std::make_shared<Tensor>(std::vector<int64_t>{2, 2}, DataType::kFLOAT32, GetDevice());
    auto second = std::make_shared<Tensor>(std::vector<int64_t>{3}, DataType::kFLOAT32, GetDevice());
    const NamedParameterList named_parameters{{"transformer.h.0.weight", first}, {"transformer.h.0.bias", second}};
    auto adam = optimizers::Adam::CreateNamed(0.001)(named_parameters);

    const auto state = adam->StateDict();
    EXPECT_TRUE(state.contains("adam.m.transformer.h.0.weight"));
    EXPECT_TRUE(state.contains("adam.v.transformer.h.0.weight"));
    EXPECT_TRUE(state.contains("adam.m.transformer.h.0.bias"));
    EXPECT_TRUE(state.contains("adam.v.transformer.h.0.bias"));
    EXPECT_TRUE(state.contains("adam.t"));

    auto restored = optimizers::Adam::CreateNamed(0.001)(named_parameters);
    restored->LoadStateDict(state);
    EXPECT_EQ(restored->StateDict().size(), state.size());
}

TEST_P(OptimizerParameterNamesTest, ConstructorMatchesNamesToOptimizerParameterOrder) {
    auto first = std::make_shared<Tensor>(std::vector<int64_t>{2, 2}, DataType::kFLOAT32, GetDevice());
    auto second = std::make_shared<Tensor>(std::vector<int64_t>{3}, DataType::kFLOAT32, GetDevice());
    const NamedParameterList named_parameters{{"second", second}, {"first", first}};

    auto adam = optimizers::Adam::CreateNamed(0.001)(named_parameters);
    const auto state = adam->StateDict();

    EXPECT_TRUE(state.contains("adam.m.second"));
    EXPECT_TRUE(state.contains("adam.v.second"));
    EXPECT_TRUE(state.contains("adam.m.first"));
    EXPECT_TRUE(state.contains("adam.v.first"));
}

TEST_P(OptimizerParameterNamesTest, PreservesNumericKeysWhenNamesAreNotSet) {
    auto parameter = std::make_shared<Tensor>(std::vector<int64_t>{2, 2}, DataType::kFLOAT32, GetDevice());
    auto adam = std::make_shared<optimizers::Adam>(std::vector<std::shared_ptr<Tensor>>{parameter}, 0.001);

    const auto state = adam->StateDict();
    EXPECT_TRUE(state.contains("adam.m.0"));
    EXPECT_TRUE(state.contains("adam.v.0"));
}

INFINI_TRAIN_REGISTER_TEST(OptimizerParameterNamesTest);
