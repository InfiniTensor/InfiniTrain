#include <gtest/gtest.h>

#include "infini_train/include/nn/parallel/pipeline_layout.h"

using namespace infini_train::nn::parallel;

TEST(PipelineLayoutMegatronStyleTest, ParsesRepeatedChunksAndSpecialModules) {
    auto layout = PipelineLayout::ParseMegatronStyleLayout("Ett|tt|tt|FH", 6, 4, 1);
    EXPECT_EQ(layout.num_layers(), 6);
    EXPECT_EQ(layout.stage(0).global_chunk_ids.size(), 1u);
    EXPECT_EQ(layout.stage(1).global_chunk_ids.size(), 1u);
    EXPECT_TRUE(layout.owns(SpecialModule::kEmbedding, 0));
    EXPECT_TRUE(layout.owns(SpecialModule::kFinalNorm, 3));
    EXPECT_TRUE(layout.owns(SpecialModule::kLMHead, 3));
}

TEST(PipelineLayoutMegatronStyleTest, ParsesVirtualPipelineChunks) {
    auto layout = PipelineLayout::ParseMegatronStyleLayout("tt,tt|tt,tt", 8, 2, 2);
    EXPECT_EQ(layout.vpp_size(), 2);
    EXPECT_EQ(layout.stage(0).global_chunk_ids.size(), 2u);
    EXPECT_EQ(layout.chunk_of_layer(0).local_chunk_id, 0);
    EXPECT_EQ(layout.chunk_of_layer(4).local_chunk_id, 1);
}

TEST(PipelineLayoutMegatronStyleTest, PreservesAsymmetricExplicitChunkOwnership) {
    // The two virtual chunks on each stage have different sizes.  Ownership
    // comes from the explicit stage/chunk description, not a fixed
    // equal-size or layer-count assumption.
    auto layout = PipelineLayout::ParseMegatronStyleLayout("t,tt|tt,t", 6, 2, 2);

    EXPECT_EQ(layout.chunk(0).stage_id, 0);
    EXPECT_EQ(layout.chunk(0).local_chunk_id, 0);
    EXPECT_EQ(layout.chunk(0).layers.size(), 1);
    EXPECT_EQ(layout.chunk(1).stage_id, 1);
    EXPECT_EQ(layout.chunk(1).local_chunk_id, 0);
    EXPECT_EQ(layout.chunk(1).layers.size(), 2);
    EXPECT_EQ(layout.chunk(2).stage_id, 0);
    EXPECT_EQ(layout.chunk(2).local_chunk_id, 1);
    EXPECT_EQ(layout.chunk(2).layers.size(), 2);
    EXPECT_EQ(layout.chunk(3).stage_id, 1);
    EXPECT_EQ(layout.chunk(3).local_chunk_id, 1);
    EXPECT_EQ(layout.chunk(3).layers.size(), 1);
}

TEST(PipelineLayoutMegatronStyleTest, BuildsExplicitChunkStageMap) {
    std::vector<ChunkLayout> chunks{
        {.global_chunk_id = 0, .stage_id = 1, .local_chunk_id = 0, .layers = {0, 2}},
        {.global_chunk_id = 1, .stage_id = 0, .local_chunk_id = 0, .layers = {2, 3}},
        {.global_chunk_id = 2, .stage_id = 1, .local_chunk_id = 1, .layers = {3, 5}},
        {.global_chunk_id = 3, .stage_id = 0, .local_chunk_id = 1, .layers = {5, 6}},
    };
    auto layout = PipelineLayout::BuildExplicit(6, 2, 2, chunks);

    EXPECT_EQ(layout.chunk(0).stage_id, 1);
    EXPECT_EQ(layout.chunk(1).stage_id, 0);
    EXPECT_EQ(layout.chunk(2).stage_id, 1);
    EXPECT_EQ(layout.chunk(3).stage_id, 0);
    EXPECT_EQ(layout.stage(0).global_chunk_ids, (std::vector<int>{1, 3}));
    EXPECT_EQ(layout.stage(1).global_chunk_ids, (std::vector<int>{0, 2}));
    EXPECT_EQ(layout.stage_of_layer(0), 1);
    EXPECT_EQ(layout.stage_of_layer(2), 0);
}

TEST(PipelineLayoutMegatronStyleTest, AllowsEmptyStageInExplicitLayout) {
    auto layout = PipelineLayout::ParseMegatronStyleLayout("Etttttt||ttttttFH", 12, 3, 1);

    EXPECT_EQ(layout.stage(0).global_chunk_ids.size(), 1u);
    EXPECT_EQ(layout.stage(1).global_chunk_ids.size(), 1u);
    EXPECT_EQ(layout.stage(2).global_chunk_ids.size(), 1u);
    EXPECT_EQ(layout.chunk(1).layers.size(), 0);
    EXPECT_NO_THROW(layout.ValidateForCurrentPipelineTransport());
}

TEST(PipelineLayoutMegatronStyleTest, RejectsIncompleteExplicitChunkStageMap) {
    std::vector<ChunkLayout> chunks{
        {.global_chunk_id = 0, .stage_id = 0, .local_chunk_id = 0, .layers = {0, 1}},
        {.global_chunk_id = 1, .stage_id = 1, .local_chunk_id = 0, .layers = {1, 2}},
        {.global_chunk_id = 2, .stage_id = 0, .local_chunk_id = 1, .layers = {2, 3}},
        // stage 1/local chunk 1 is intentionally missing
    };
    EXPECT_THROW(PipelineLayout::BuildExplicit(4, 2, 2, chunks), PipelineLayoutError);
}

TEST(PipelineLayoutMegatronStyleTest, ExpandsParenthesizedRepetition) {
    auto layout = PipelineLayout::ParseMegatronStyleLayout("(tt)*2|tt|tt", 8, 3, 1);
    EXPECT_EQ(layout.chunk_of_layer(0).layers.size(), 4);
    EXPECT_EQ(layout.stage_of_layer(4), 1);
}

TEST(PipelineLayoutMegatronStyleTest, RejectsMalformedExpressions) {
    EXPECT_THROW(PipelineLayout::ParseMegatronStyleLayout("(tt|tt", 4, 2), PipelineLayoutError);
    EXPECT_THROW(PipelineLayout::ParseMegatronStyleLayout("t,x|tt", 3, 2), PipelineLayoutError);
    EXPECT_THROW(PipelineLayout::ParseMegatronStyleLayout("tt|tt", 3, 2), PipelineLayoutError);
}
