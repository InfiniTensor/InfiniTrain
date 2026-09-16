#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/nn/parallel/pp/pipeline_layout.h"

namespace infini_train::nn::parallel {
namespace {

TEST(PipelineLayoutTest, DefaultUniformPartition) {
    auto layout = PipelineLayout::Create(24, 4, 1);

    auto s0 = layout.GetStageInfo(0);
    EXPECT_TRUE(s0.is_first_stage);
    EXPECT_FALSE(s0.is_last_stage);
    ASSERT_EQ(s0.layer_ranges_per_chunk.size(), 1u);
    EXPECT_EQ(s0.layer_ranges_per_chunk[0], std::make_pair(0, 6));

    auto s3 = layout.GetStageInfo(3);
    EXPECT_FALSE(s3.is_first_stage);
    EXPECT_TRUE(s3.is_last_stage);
    ASSERT_EQ(s3.layer_ranges_per_chunk.size(), 1u);
    EXPECT_EQ(s3.layer_ranges_per_chunk[0], std::make_pair(18, 24));
}

TEST(PipelineLayoutTest, CustomPartition) {
    auto layout = PipelineLayout::Create(24, 4, 1, {4, 8, 6, 6});

    auto s0 = layout.GetStageInfo(0);
    auto s1 = layout.GetStageInfo(1);
    auto s2 = layout.GetStageInfo(2);
    auto s3 = layout.GetStageInfo(3);

    EXPECT_TRUE(s0.is_first_stage);
    EXPECT_FALSE(s0.is_last_stage);
    EXPECT_FALSE(s3.is_first_stage);
    EXPECT_TRUE(s3.is_last_stage);

    ASSERT_EQ(s0.layer_ranges_per_chunk.size(), 1u);
    ASSERT_EQ(s1.layer_ranges_per_chunk.size(), 1u);
    ASSERT_EQ(s2.layer_ranges_per_chunk.size(), 1u);
    ASSERT_EQ(s3.layer_ranges_per_chunk.size(), 1u);

    EXPECT_EQ(s0.layer_ranges_per_chunk[0], std::make_pair(0, 4));
    EXPECT_EQ(s1.layer_ranges_per_chunk[0], std::make_pair(4, 12));
    EXPECT_EQ(s2.layer_ranges_per_chunk[0], std::make_pair(12, 18));
    EXPECT_EQ(s3.layer_ranges_per_chunk[0], std::make_pair(18, 24));
}

TEST(PipelineLayoutTest, StageOfLayerAndOwnsLayer) {
    auto layout = PipelineLayout::Create(24, 4, 1, {4, 8, 6, 6});

    EXPECT_EQ(layout.StageOfLayer(0), 0);
    EXPECT_EQ(layout.StageOfLayer(5), 1);
    EXPECT_EQ(layout.StageOfLayer(17), 2);
    EXPECT_EQ(layout.StageOfLayer(23), 3);

    EXPECT_TRUE(layout.OwnsLayer(0, 3));
    EXPECT_FALSE(layout.OwnsLayer(0, 4));
}

TEST(PipelineLayoutTest, VirtualPipelineInterleaving) {
    auto layout = PipelineLayout::Create(8, 2, 2);

    auto s0 = layout.GetStageInfo(0);
    ASSERT_EQ(s0.layer_ranges_per_chunk.size(), 2u);
    EXPECT_EQ(s0.layer_ranges_per_chunk[0], std::make_pair(0, 2));
    EXPECT_EQ(s0.layer_ranges_per_chunk[1], std::make_pair(4, 6));

    auto s1 = layout.GetStageInfo(1);
    ASSERT_EQ(s1.layer_ranges_per_chunk.size(), 2u);
    EXPECT_EQ(s1.layer_ranges_per_chunk[0], std::make_pair(2, 4));
    EXPECT_EQ(s1.layer_ranges_per_chunk[1], std::make_pair(6, 8));
}

TEST(PipelineLayoutTest, ChunkMappingHelpers) {
    EXPECT_EQ(PipelineLayout::StageOfChunk(0, 4), 0);
    EXPECT_EQ(PipelineLayout::StageOfChunk(5, 4), 1);
    EXPECT_EQ(PipelineLayout::LocalChunkIndexOfChunk(5, 4), 1);
    EXPECT_EQ(PipelineLayout::LocalChunkIndexOfChunk(7, 4), 1);
}

TEST(PipelineLayoutTest, DescribeNonEmpty) {
    auto layout = PipelineLayout::Create(24, 4, 1, {4, 8, 6, 6});
    EXPECT_FALSE(layout.Describe().empty());
}

TEST(PipelineLayoutTest, ParsePartition) {
    EXPECT_TRUE(ParsePipelineLayerPartition("").empty());
    const std::vector<int> expected{4, 8, 6, 6};
    EXPECT_EQ(ParsePipelineLayerPartition("4,8,6,6"), expected);
}

TEST(PipelineLayoutTest, RejectsPartitionSumMismatch) {
    EXPECT_DEATH(PipelineLayout::Create(24, 4, 1, {4, 8, 6, 5}), "sums to");
}

TEST(PipelineLayoutTest, RejectsStageCountMismatch) {
    EXPECT_DEATH(PipelineLayout::Create(24, 4, 1, {6, 6, 6}), "entries");
}

TEST(PipelineLayoutTest, RejectsNonPositiveEntry) {
    EXPECT_DEATH(PipelineLayout::Create(24, 4, 1, {4, 0, 6, 14}), "must be positive");
}

TEST(PipelineLayoutTest, RejectsVirtualPipelineConflict) {
    EXPECT_DEATH(PipelineLayout::Create(24, 4, 2, {4, 8, 6, 6}), "incompatible");
}

TEST(PipelineLayoutTest, RejectsInvalidToken) {
    EXPECT_DEATH(ParsePipelineLayerPartition("4,8a,6,6"), "not a positive integer");
}

} // namespace
} // namespace infini_train::nn::parallel
