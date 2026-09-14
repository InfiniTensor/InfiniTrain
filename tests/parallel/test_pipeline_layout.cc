#include <functional>

#include "gtest/gtest.h"

#include "infini_train/include/nn/parallel/pipeline_layout.h"

namespace infini_train::nn::parallel {
namespace {

PipelineLayout BuildExampleLayout() {
    return PipelineLayout::BuildDefault(24, 4, 1);
}

}  // namespace

TEST(PipelineLayoutTest, BuildDefaultBasicProperties) {
    auto layout = BuildExampleLayout();

    EXPECT_EQ(layout.num_layers(), 24);
    EXPECT_EQ(layout.num_stages(), 4);
    EXPECT_EQ(layout.vpp_size(), 1);
}

TEST(PipelineLayoutTest, BuildDefaultChunkRanges) {
    auto layout = BuildExampleLayout();

    EXPECT_EQ(layout.chunk(0).layers.start, 0);
    EXPECT_EQ(layout.chunk(0).layers.end, 6);
    EXPECT_EQ(layout.chunk(1).layers.start, 6);
    EXPECT_EQ(layout.chunk(1).layers.end, 12);
    EXPECT_EQ(layout.chunk(2).layers.start, 12);
    EXPECT_EQ(layout.chunk(2).layers.end, 18);
    EXPECT_EQ(layout.chunk(3).layers.start, 18);
    EXPECT_EQ(layout.chunk(3).layers.end, 24);
}

TEST(PipelineLayoutTest, BuildDefaultVirtualPipelineUsesInterleavedChunkIds) {
    auto layout = PipelineLayout::BuildDefault(8, 2, 2);

    ASSERT_EQ(layout.chunks().size(), 4);
    EXPECT_EQ(layout.chunk(0).stage_id, 0);
    EXPECT_EQ(layout.chunk(0).local_chunk_id, 0);
    EXPECT_EQ(layout.chunk(1).stage_id, 1);
    EXPECT_EQ(layout.chunk(1).local_chunk_id, 0);
    EXPECT_EQ(layout.chunk(2).stage_id, 0);
    EXPECT_EQ(layout.chunk(2).local_chunk_id, 1);
    EXPECT_EQ(layout.chunk(3).stage_id, 1);
    EXPECT_EQ(layout.chunk(3).local_chunk_id, 1);
}

TEST(PipelineLayoutTest, BuildDefaultAllowsFewerLayersThanGlobalChunks) {
    auto layout = PipelineLayout::BuildDefault(2, 4, 1);

    EXPECT_EQ(layout.num_layers(), 2);
    EXPECT_EQ(layout.chunks().size(), 2);
    EXPECT_EQ(layout.chunk(0).layers.start, 0);
    EXPECT_EQ(layout.chunk(0).layers.end, 1);
    EXPECT_EQ(layout.chunk(1).layers.start, 1);
    EXPECT_EQ(layout.chunk(1).layers.end, 2);
    EXPECT_NO_THROW(layout.Validate());
    EXPECT_THROW(layout.ValidateForCurrentPipelineTransport(), PipelineLayoutError);
}

TEST(PipelineLayoutTest, StageOfLayer) {
    auto layout = BuildExampleLayout();

    EXPECT_EQ(layout.stage_of_layer(0), 0);
    EXPECT_EQ(layout.stage_of_layer(5), 0);
    EXPECT_EQ(layout.stage_of_layer(6), 1);
    EXPECT_EQ(layout.stage_of_layer(11), 1);
    EXPECT_EQ(layout.stage_of_layer(12), 2);
    EXPECT_EQ(layout.stage_of_layer(17), 2);
    EXPECT_EQ(layout.stage_of_layer(18), 3);
    EXPECT_EQ(layout.stage_of_layer(23), 3);
}

TEST(PipelineLayoutTest, ChunkOfLayer) {
    auto layout = BuildExampleLayout();

    EXPECT_EQ(layout.chunk_of_layer(0).global_chunk_id, 0);
    EXPECT_EQ(layout.chunk_of_layer(5).global_chunk_id, 0);
    EXPECT_EQ(layout.chunk_of_layer(6).global_chunk_id, 1);
    EXPECT_EQ(layout.chunk_of_layer(11).global_chunk_id, 1);
    EXPECT_EQ(layout.chunk_of_layer(23).global_chunk_id, 3);
}

TEST(PipelineLayoutTest, LocalLayerIndexForOwnedStage) {
    auto layout = BuildExampleLayout();

    EXPECT_EQ(layout.local_layer_index(0, 0), 0);
    EXPECT_EQ(layout.local_layer_index(0, 5), 5);
    EXPECT_EQ(layout.local_layer_index(1, 6), 0);
    EXPECT_EQ(layout.local_layer_index(1, 11), 5);
    EXPECT_EQ(layout.local_layer_index(3, 18), 0);
    EXPECT_EQ(layout.local_layer_index(3, 23), 5);
}

TEST(PipelineLayoutTest, SpecialModulePlacement) {
    auto layout = BuildExampleLayout();

    EXPECT_TRUE(layout.owns(SpecialModule::kEmbedding, 0));
    EXPECT_FALSE(layout.owns(SpecialModule::kEmbedding, 3));

    EXPECT_TRUE(layout.owns(SpecialModule::kFinalNorm, 3));
    EXPECT_FALSE(layout.owns(SpecialModule::kFinalNorm, 0));

    EXPECT_TRUE(layout.owns(SpecialModule::kLMHead, 3));
    EXPECT_FALSE(layout.owns(SpecialModule::kLMHead, 0));
}

TEST(PipelineLayoutTest, ExplicitBoundarySpecialModulePlacement) {
    SpecialModulePlacement placement{
        .embedding_stage = 0,
        .final_norm_stage = 3,
        .lm_head_stage = 3,
    };
    auto layout = PipelineLayout::BuildContiguous({4, 8, 6, 6}, placement);

    EXPECT_EQ(layout.special_modules().embedding_stage, 0);
    EXPECT_EQ(layout.special_modules().final_norm_stage, 3);
    EXPECT_EQ(layout.special_modules().lm_head_stage, 3);
    EXPECT_NO_THROW(layout.Validate());
}

TEST(PipelineLayoutTest, RejectsNonBoundarySpecialModulesByDefault) {
    SpecialModulePlacement placement{
        .embedding_stage = 1,
        .final_norm_stage = 3,
        .lm_head_stage = 3,
    };
    EXPECT_THROW(PipelineLayout::BuildContiguous({4, 8, 6, 6}, placement), PipelineLayoutError);
}

TEST(PipelineLayoutTest, ToStringContainsBasicInfo) {
    auto layout = BuildExampleLayout();
    const std::string text = layout.ToString();

    EXPECT_NE(text.find("num_layers: 24"), std::string::npos);
    EXPECT_NE(text.find("num_stages: 4"), std::string::npos);
    EXPECT_NE(text.find("stage 3:"), std::string::npos);
}

TEST(PipelineLayoutTest, ParseLayerPartitionValid) {
    EXPECT_EQ(PipelineLayout::ParseLayerPartition("4,8,6,6"), (std::vector<int>{4, 8, 6, 6}));
    EXPECT_EQ(PipelineLayout::ParseLayerPartition("24"), (std::vector<int>{24}));
    EXPECT_EQ(PipelineLayout::ParseLayerPartition("0,4"), (std::vector<int>{0, 4}));
}

TEST(PipelineLayoutTest, ParseLayerPartitionRejectsInvalidStrings) {
    EXPECT_THROW(PipelineLayout::ParseLayerPartition(""), PipelineLayoutError);
    EXPECT_THROW(PipelineLayout::ParseLayerPartition("4,,6"), PipelineLayoutError);
    EXPECT_THROW(PipelineLayout::ParseLayerPartition(",4,6"), PipelineLayoutError);
    EXPECT_THROW(PipelineLayout::ParseLayerPartition("4,6,"), PipelineLayoutError);
    EXPECT_THROW(PipelineLayout::ParseLayerPartition("4, 6"), PipelineLayoutError);
    EXPECT_THROW(PipelineLayout::ParseLayerPartition("4,a,6"), PipelineLayoutError);
    EXPECT_THROW(PipelineLayout::ParseLayerPartition("-1,6"), PipelineLayoutError);
    EXPECT_THROW(PipelineLayout::ParseLayerPartition("999999999999999999999"), PipelineLayoutError);
}

TEST(PipelineLayoutTest, BuildPipelineLayoutWithCustomPartition) {
    auto layout = PipelineLayout::BuildPipelineLayout(24, 4, 1, "4,8,6,6");

    EXPECT_EQ(layout.num_layers(), 24);
    EXPECT_EQ(layout.num_stages(), 4);
    EXPECT_EQ(layout.vpp_size(), 1);
    EXPECT_EQ(layout.chunk(0).layers.start, 0);
    EXPECT_EQ(layout.chunk(0).layers.end, 4);
    EXPECT_EQ(layout.chunk(1).layers.start, 4);
    EXPECT_EQ(layout.chunk(1).layers.end, 12);
    EXPECT_EQ(layout.chunk(3).layers.start, 18);
    EXPECT_EQ(layout.chunk(3).layers.end, 24);
}

TEST(PipelineLayoutTest, BuildPipelineLayoutEmptyPartitionUsesDefault) {
    auto layout = PipelineLayout::BuildPipelineLayout(24, 4, 1, "");

    EXPECT_EQ(layout.num_layers(), 24);
    EXPECT_EQ(layout.chunk(0).layers.start, 0);
    EXPECT_EQ(layout.chunk(0).layers.end, 6);
    EXPECT_EQ(layout.chunk(3).layers.start, 18);
    EXPECT_EQ(layout.chunk(3).layers.end, 24);
}

TEST(PipelineLayoutTest, BuildPipelineLayoutRejectsInvalidConfigurations) {
    EXPECT_THROW(PipelineLayout::BuildPipelineLayout(23, 4, 1, "4,8,6,6"), PipelineLayoutError);
    EXPECT_THROW(PipelineLayout::BuildPipelineLayout(24, 3, 1, "4,8,6,6"), PipelineLayoutError);
    EXPECT_THROW(PipelineLayout::BuildPipelineLayout(24, 4, 2, "4,8,6,6"), PipelineLayoutError);
}

}  // namespace infini_train::nn::parallel
