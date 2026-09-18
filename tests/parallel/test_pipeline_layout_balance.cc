#include <gtest/gtest.h>

#include "infini_train/include/nn/parallel/pipeline_layout.h"

using namespace infini_train::nn::parallel;

TEST(PipelineLayoutBalanceTest, BalancesUniformCosts) {
    EXPECT_EQ(PipelineLayout::SuggestBalancedPartition(8, 4), (std::vector<int>{2, 2, 2, 2}));
}

TEST(PipelineLayoutBalanceTest, IsolatesHeavyLayer) {
    const auto partition = PipelineLayout::SuggestBalancedPartition(8, 4, {1, 1, 1, 1, 4, 1, 1, 1});
    ASSERT_EQ(partition.size(), 4u);
    EXPECT_EQ(partition[0], 3);
    EXPECT_EQ(partition[1], 1);
    EXPECT_EQ(partition[2], 1);
    EXPECT_EQ(partition[3], 3);
}

TEST(PipelineLayoutBalanceTest, RejectsInvalidCosts) {
    EXPECT_THROW(PipelineLayout::SuggestBalancedPartition(4, 2, {1, -1, 1, 1}), PipelineLayoutError);
    EXPECT_THROW(PipelineLayout::SuggestBalancedPartition(4, 2, {1, 1}), PipelineLayoutError);
}
