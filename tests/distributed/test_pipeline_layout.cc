#include <stdexcept>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/nn/parallel/pp/pipeline_parallel.h"
#include "infini_train/include/nn/parallel/pp/pipeline_schedule.h"

namespace infini_train::nn::parallel {
namespace {

TEST(PipelineLayoutTest, ParsesNonUniformContinuousPartition) {
    const auto layout = PipelineLayout::Parse(24, 4, "4, 8,6,6");

    EXPECT_EQ(layout.num_stages(), 4);
    EXPECT_EQ(layout.total_layers(), 24);
    EXPECT_EQ(layout.layer_ranges(0), (std::vector<std::pair<int, int>>{{0, 4}}));
    EXPECT_EQ(layout.layer_ranges(1), (std::vector<std::pair<int, int>>{{4, 12}}));
    EXPECT_EQ(layout.layer_ranges(2), (std::vector<std::pair<int, int>>{{12, 18}}));
    EXPECT_EQ(layout.layer_ranges(3), (std::vector<std::pair<int, int>>{{18, 24}}));

    for (int layer = 0; layer < 24; ++layer) {
        const int expected_stage = layer < 4 ? 0 : layer < 12 ? 1 : layer < 18 ? 2 : 3;
        EXPECT_EQ(layout.stage_for_layer(layer), expected_stage);
    }
}

TEST(PipelineLayoutTest, AssignsSpecialModulesToPipelineEndpoints) {
    const auto layout = PipelineLayout::Parse(6, 2, "2,4");

    EXPECT_TRUE(layout.owns_embedding(0));
    EXPECT_FALSE(layout.owns_final_norm(0));
    EXPECT_FALSE(layout.owns_lm_head(0));
    EXPECT_FALSE(layout.owns_embedding(1));
    EXPECT_TRUE(layout.owns_final_norm(1));
    EXPECT_TRUE(layout.owns_lm_head(1));
    EXPECT_NE(layout.ToString().find("stage 0: embedding layers[0,2)"), std::string::npos);
    EXPECT_NE(layout.ToString().find("stage 1: layers[2,6) final_norm lm_head"), std::string::npos);
}

TEST(PipelineLayoutTest, PreservesUniformAndVirtualPipelineDistribution) {
    const auto layout = PipelineLayout::Uniform(10, 2, 2);

    EXPECT_EQ(layout.chunks_per_stage(), 2);
    EXPECT_EQ(layout.layer_ranges(0), (std::vector<std::pair<int, int>>{{0, 3}, {6, 8}}));
    EXPECT_EQ(layout.layer_ranges(1), (std::vector<std::pair<int, int>>{{3, 6}, {8, 10}}));
}

TEST(PipelineLayoutTest, BalancesUserProvidedLayerCosts) {
    const auto layout = PipelineLayout::FromLayerCosts(6, 2, "10,1,1,1,1,1");

    EXPECT_EQ(layout.layer_ranges(0), (std::vector<std::pair<int, int>>{{0, 1}}));
    EXPECT_EQ(layout.layer_ranges(1), (std::vector<std::pair<int, int>>{{1, 6}}));
    EXPECT_EQ(layout.stage_for_layer(0), 0);
    EXPECT_EQ(layout.stage_for_layer(5), 1);
}

TEST(PipelineLayoutTest, SupportsArbitraryVirtualChunkOwnership) {
    const auto layout = PipelineLayout::FromChunkLayout(8, 2, "0:2,1:2,1:2,0:2");

    EXPECT_EQ(layout.chunks_per_stage(), 2);
    EXPECT_EQ(layout.num_global_chunks(), 4);
    EXPECT_EQ(layout.stage_for_chunk(2), 1);
    EXPECT_EQ(layout.local_chunk_index(2), 1);
    EXPECT_EQ(layout.stage_for_chunk(3), 0);
    EXPECT_EQ(layout.local_chunk_index(3), 1);
    EXPECT_TRUE(layout.owns_embedding(0));
    EXPECT_TRUE(layout.owns_final_norm(0));
    EXPECT_EQ(layout.layer_ranges(0), (std::vector<std::pair<int, int>>{{0, 2}, {6, 8}}));
    EXPECT_EQ(layout.layer_ranges(1), (std::vector<std::pair<int, int>>{{2, 4}, {4, 6}}));

    SetPipelineLayout(layout);
    const auto task = PipelineParallelScheduler::CreateTask(3, 0, 2, 2, 4, true);
    EXPECT_EQ(task.stage_id, 1);
    EXPECT_EQ(task.local_chunk_idx, 1);
    SetPipelineLayout(std::nullopt);
}

TEST(PipelineLayoutTest, ParsesMegatronRepetitionAndEmptyChunks) {
    const auto layout = PipelineLayout::FromMegatronLayout(8, 2, "Et*2||t*2|t*4NL");

    EXPECT_EQ(layout.chunks_per_stage(), 2);
    EXPECT_EQ(layout.num_global_chunks(), 4);
    EXPECT_EQ(layout.chunk_range(0), (std::pair<int, int>{0, 2}));
    EXPECT_EQ(layout.chunk_range(1), (std::pair<int, int>{2, 2}));
    EXPECT_EQ(layout.chunk_range(2), (std::pair<int, int>{2, 4}));
    EXPECT_EQ(layout.chunk_range(3), (std::pair<int, int>{4, 8}));
}

TEST(PipelineLayoutTest, RejectsInvalidAutomaticLayoutInputs) {
    EXPECT_THROW(PipelineLayout::FromLayerCosts(6, 2, "1,2,3"), std::invalid_argument);
    EXPECT_THROW(PipelineLayout::FromLayerCosts(3, 2, "1,0,2"), std::invalid_argument);
    EXPECT_THROW(PipelineLayout::FromLayerCosts(3, 2, "1,-1,2"), std::invalid_argument);
    EXPECT_THROW(PipelineLayout::FromLayerCosts(3, 2, "1,nope,2"), std::invalid_argument);
    EXPECT_THROW(PipelineLayout::FromLayerCosts(3, 2, "1,nan,2"), std::invalid_argument);
    EXPECT_THROW(PipelineLayout::FromLayerCosts(3, 2, "1,inf,2"), std::invalid_argument);
    EXPECT_THROW(PipelineLayout::FromLayerCosts(2, 2, "1.7e308,1.7e308"), std::invalid_argument);
    EXPECT_THROW(PipelineLayout::FromLayerCosts(3, 4, "1,1,1"), std::invalid_argument);
    EXPECT_THROW(PipelineLayout::FromLayerCosts(3, 2, "1,1,1", 2), std::invalid_argument);
    EXPECT_THROW(ResolvePipelineLayout(3, 2, 1, "1,2", "1,1,1"), std::invalid_argument);
    EXPECT_THROW(PipelineLayout::FromChunkLayout(4, 2, "0:2,1:1"), std::invalid_argument);
    EXPECT_THROW(PipelineLayout::FromChunkLayout(4, 2, "0:2,1:2,0:0"), std::invalid_argument);
    EXPECT_THROW(PipelineLayout::FromChunkLayout(4, 2, "0x:2,1:2"), std::invalid_argument);
    EXPECT_THROW(PipelineLayout::FromMegatronLayout(4, 2, ""), std::invalid_argument);
    EXPECT_THROW(PipelineLayout::FromMegatronLayout(4, 2, "Et*3|t*2L"), std::invalid_argument);
    EXPECT_THROW(PipelineLayout::FromMegatronLayout(4, 2, "Et*2|t*2N|L"), std::invalid_argument);
}

TEST(PipelineLayoutTest, RejectsInvalidPartitions) {
    EXPECT_THROW(PipelineLayout::Parse(24, 4, "4,8,6"), std::invalid_argument);
    EXPECT_THROW(PipelineLayout::Parse(24, 4, "4,8,6,5"), std::invalid_argument);
    EXPECT_THROW(PipelineLayout::Parse(24, 4, "4,-8,12,16"), std::invalid_argument);
    EXPECT_THROW(PipelineLayout::Parse(24, 4, "4,0,8,12"), std::invalid_argument);
    EXPECT_THROW(PipelineLayout::Parse(24, 4, "4,,8,12"), std::invalid_argument);
    EXPECT_THROW(PipelineLayout::Parse(24, 4, "4,8,6,6", 2), std::invalid_argument);
}

TEST(PipelineLayoutTest, RejectsOutOfRangeQueries) {
    const auto layout = PipelineLayout::Parse(4, 2, "1,3");
    EXPECT_THROW(layout.layer_ranges(2), std::out_of_range);
    EXPECT_THROW(layout.stage_for_layer(4), std::out_of_range);
}

} // namespace
} // namespace infini_train::nn::parallel
