#include <gtest/gtest.h>

#include "infini_train/include/nn/parallel/pipeline_layout.h"

using namespace infini_train::nn::parallel;

TEST(PipelineLayoutMegatronStyleTest, ParsesRepeatedChunksAndSpecialModules) {
    auto layout = PipelineLayout::ParseMegatronStyleLayout(
        "Ett|tt|tt|FH", 6, 4, 1);
    EXPECT_EQ(layout.num_layers(), 6);
    EXPECT_EQ(layout.stage(0).global_chunk_ids.size(), 1u);
    EXPECT_EQ(layout.stage(1).global_chunk_ids.size(), 1u);
    EXPECT_TRUE(layout.owns(SpecialModule::kEmbedding, 0));
    EXPECT_TRUE(layout.owns(SpecialModule::kFinalNorm, 3));
    EXPECT_TRUE(layout.owns(SpecialModule::kLMHead, 3));
}

TEST(PipelineLayoutMegatronStyleTest, ParsesVirtualPipelineChunks) {
    auto layout = PipelineLayout::ParseMegatronStyleLayout(
        "tt,tt|tt,tt", 8, 2, 2);
    EXPECT_EQ(layout.vpp_size(), 2);
    EXPECT_EQ(layout.stage(0).global_chunk_ids.size(), 2u);
    EXPECT_EQ(layout.chunk_of_layer(0).local_chunk_id, 0);
    EXPECT_EQ(layout.chunk_of_layer(4).local_chunk_id, 1);
}

TEST(PipelineLayoutMegatronStyleTest, ExpandsParenthesizedRepetition) {
    auto layout = PipelineLayout::ParseMegatronStyleLayout("(tt)*2|tt|tt", 8, 3, 1);
    EXPECT_EQ(layout.chunk_of_layer(0).layers.size(), 4);
    EXPECT_EQ(layout.stage_of_layer(4), 1);
}

TEST(PipelineLayoutMegatronStyleTest, RejectsMalformedExpressions) {
    EXPECT_THROW(PipelineLayout::ParseMegatronStyleLayout("(tt|tt", 4, 2),
                 PipelineLayoutError);
    EXPECT_THROW(PipelineLayout::ParseMegatronStyleLayout("t,x|tt", 3, 2),
                 PipelineLayoutError);
    EXPECT_THROW(PipelineLayout::ParseMegatronStyleLayout("tt|tt", 3, 2),
                 PipelineLayoutError);
}
