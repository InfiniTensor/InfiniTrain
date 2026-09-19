#include "gtest/gtest.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "infini_train/include/nn/parallel/pp/pipeline_layout.h"

namespace infini_train::nn::parallel {
namespace {

static_assert(!std::is_default_constructible_v<PipelineLayout>);

void ExpectRange(const LayerRange &range, LayerIndex expected_begin, LayerIndex expected_end) {
    EXPECT_EQ(range.begin, expected_begin);
    EXPECT_EQ(range.end, expected_end);
}

TEST(LayerRangeTest, UsesHalfOpenBounds) {
    const LayerRange range{4, 12};

    EXPECT_EQ(range.Size(), 8);
    EXPECT_FALSE(range.Contains(3));
    EXPECT_TRUE(range.Contains(4));
    EXPECT_TRUE(range.Contains(11));
    EXPECT_FALSE(range.Contains(12));
}

TEST(PipelineLayoutTest, CreatesUniformLayouts) {
    const PipelineLayout even = PipelineLayout::BuildUniformLayout(12, 2);
    EXPECT_EQ(even.GetNumLayers(), 12);
    EXPECT_EQ(even.GetNumStages(), 2);
    EXPECT_FALSE(even.IsCustom());

    const PipelineStageLayout &stage0 = even.GetStage(0);
    const PipelineStageLayout &stage1 = even.GetStage(1);
    ASSERT_EQ(stage0.chunks.size(), 1U);
    ASSERT_EQ(stage1.chunks.size(), 1U);
    EXPECT_EQ(stage0.stage_id, 0);
    EXPECT_EQ(stage1.stage_id, 1);
    EXPECT_EQ(stage0.chunks[0].local_chunk_idx, 0);
    EXPECT_EQ(stage1.chunks[0].local_chunk_idx, 0);
    ExpectRange(stage0.chunks[0].layer_range, 0, 6);
    ExpectRange(stage1.chunks[0].layer_range, 6, 12);
    EXPECT_TRUE(stage0.has_embedding);
    EXPECT_FALSE(stage0.has_final_norm);
    EXPECT_FALSE(stage0.has_lm_head);
    EXPECT_FALSE(stage1.has_embedding);
    EXPECT_TRUE(stage1.has_final_norm);
    EXPECT_TRUE(stage1.has_lm_head);

    const PipelineLayout uneven = PipelineLayout::BuildUniformLayout(13, 2);
    ASSERT_EQ(uneven.GetStage(0).chunks.size(), 1U);
    ASSERT_EQ(uneven.GetStage(1).chunks.size(), 1U);
    ExpectRange(uneven.GetStage(0).chunks[0].layer_range, 0, 7);
    ExpectRange(uneven.GetStage(1).chunks[0].layer_range, 7, 13);

    const PipelineLayout single = PipelineLayout::BuildUniformLayout(12, 1);
    const PipelineStageLayout &single_stage = single.GetStage(0);
    ASSERT_EQ(single_stage.chunks.size(), 1U);
    ExpectRange(single_stage.chunks[0].layer_range, 0, 12);
    EXPECT_TRUE(single_stage.has_embedding);
    EXPECT_TRUE(single_stage.has_final_norm);
    EXPECT_TRUE(single_stage.has_lm_head);
}

TEST(PipelineLayoutTest, OneLayerPerStage) {
    const auto layout = PipelineLayout::BuildUniformLayout(4, 4);
    for (int stage = 0; stage < 4; ++stage) {
        const auto &range = layout.GetStage(stage).chunks.at(0).layer_range;
        EXPECT_EQ(range.begin, stage);
        EXPECT_EQ(range.end, stage + 1);
        EXPECT_EQ(layout.GetLocalLayerIndex(stage, stage), 0);
    }
}

TEST(PipelineLayoutInvalidTest, RejectsInvalidUniformArguments) {
    EXPECT_THROW((void)PipelineLayout::BuildUniformLayout(-1, 2), std::invalid_argument);
    EXPECT_THROW((void)PipelineLayout::BuildUniformLayout(12, 0), std::invalid_argument);
    EXPECT_THROW((void)PipelineLayout::BuildUniformLayout(12, -1), std::invalid_argument);
    EXPECT_THROW((void)PipelineLayout::BuildUniformLayout(12, 2, 0), std::invalid_argument);
    EXPECT_THROW((void)PipelineLayout::BuildUniformLayout(12, 2, std::numeric_limits<int>::max()),
                 std::invalid_argument);
}

TEST(PipelineLayoutTest, CreatesCustomContinuousLayouts) {
    const std::vector<LayerIndex> counts{4, 8, 6, 6};
    const PipelineLayout layout = PipelineLayout::BuildCustomLayout(24, 4, counts);

    EXPECT_EQ(layout.GetNumLayers(), 24);
    EXPECT_EQ(layout.GetNumStages(), 4);
    EXPECT_TRUE(layout.IsCustom());

    const PipelineStageLayout &stage0 = layout.GetStage(0);
    const PipelineStageLayout &stage1 = layout.GetStage(1);
    const PipelineStageLayout &stage2 = layout.GetStage(2);
    const PipelineStageLayout &stage3 = layout.GetStage(3);
    ASSERT_EQ(stage0.chunks.size(), 1U);
    ASSERT_EQ(stage1.chunks.size(), 1U);
    ASSERT_EQ(stage2.chunks.size(), 1U);
    ASSERT_EQ(stage3.chunks.size(), 1U);
    EXPECT_EQ(stage0.stage_id, 0);
    EXPECT_EQ(stage1.stage_id, 1);
    EXPECT_EQ(stage2.stage_id, 2);
    EXPECT_EQ(stage3.stage_id, 3);
    EXPECT_EQ(stage0.chunks[0].local_chunk_idx, 0);
    EXPECT_EQ(stage1.chunks[0].local_chunk_idx, 0);
    EXPECT_EQ(stage2.chunks[0].local_chunk_idx, 0);
    EXPECT_EQ(stage3.chunks[0].local_chunk_idx, 0);
    ExpectRange(stage0.chunks[0].layer_range, 0, 4);
    ExpectRange(stage1.chunks[0].layer_range, 4, 12);
    ExpectRange(stage2.chunks[0].layer_range, 12, 18);
    ExpectRange(stage3.chunks[0].layer_range, 18, 24);

    EXPECT_TRUE(stage0.has_embedding);
    EXPECT_FALSE(stage1.has_embedding);
    EXPECT_FALSE(stage2.has_embedding);
    EXPECT_FALSE(stage3.has_embedding);
    EXPECT_FALSE(stage0.has_final_norm);
    EXPECT_FALSE(stage1.has_final_norm);
    EXPECT_FALSE(stage2.has_final_norm);
    EXPECT_TRUE(stage3.has_final_norm);
    EXPECT_FALSE(stage0.has_lm_head);
    EXPECT_FALSE(stage1.has_lm_head);
    EXPECT_FALSE(stage2.has_lm_head);
    EXPECT_TRUE(stage3.has_lm_head);

    const PipelineLayout gpt2_layout = [] {
        const std::vector<LayerIndex> gpt2_counts{7, 5};
        return PipelineLayout::BuildCustomLayout(12, 2, gpt2_counts);
    }();
    ASSERT_EQ(gpt2_layout.GetStage(0).chunks.size(), 1U);
    ASSERT_EQ(gpt2_layout.GetStage(1).chunks.size(), 1U);
    ExpectRange(gpt2_layout.GetStage(0).chunks[0].layer_range, 0, 7);
    ExpectRange(gpt2_layout.GetStage(1).chunks[0].layer_range, 7, 12);
}

TEST(PipelineLayoutInvalidTest, RejectsInvalidCustomArguments) {
    const std::vector<LayerIndex> empty_counts;
    const std::vector<LayerIndex> wrong_count{4, 4, 4};
    const std::vector<LayerIndex> zero_count{12, 0};
    const std::vector<LayerIndex> negative_count{13, -1};
    const std::vector<LayerIndex> valid_counts{7, 5};

    EXPECT_THROW((void)PipelineLayout::BuildCustomLayout(12, 2, empty_counts), std::invalid_argument);
    EXPECT_THROW((void)PipelineLayout::BuildCustomLayout(12, 2, wrong_count), std::invalid_argument);
    EXPECT_THROW((void)PipelineLayout::BuildCustomLayout(12, 2, zero_count), std::invalid_argument);
    EXPECT_THROW((void)PipelineLayout::BuildCustomLayout(12, 2, negative_count), std::invalid_argument);
    EXPECT_THROW((void)PipelineLayout::BuildCustomLayout(0, 2, valid_counts), std::invalid_argument);
    EXPECT_THROW((void)PipelineLayout::BuildCustomLayout(12, 0, valid_counts), std::invalid_argument);
    EXPECT_THROW((void)PipelineLayout::BuildCustomLayout(12, 2, valid_counts, 0), std::invalid_argument);
    EXPECT_THROW((void)PipelineLayout::BuildCustomLayout(12, 2, valid_counts, 2), std::invalid_argument);
    EXPECT_THROW((void)PipelineLayout::BuildCustomLayout(12, 2, valid_counts, std::numeric_limits<int>::max()),
                 std::invalid_argument);

    try {
        const std::vector<LayerIndex> counts{4, 7, 8};
        (void)PipelineLayout::BuildCustomLayout(20, 3, counts);
        FAIL() << "Expected std::invalid_argument";
    } catch (const std::invalid_argument &error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("received=[4,7,8]"), std::string::npos);
        EXPECT_NE(message.find("num_stages=3"), std::string::npos);
        EXPECT_NE(message.find("expected_sum=20"), std::string::npos);
        EXPECT_NE(message.find("actual_sum=19"), std::string::npos);
    }

    const std::vector<LayerIndex> overflowing_counts{std::numeric_limits<LayerIndex>::max(), 1};
    EXPECT_THROW((void)PipelineLayout::BuildCustomLayout(std::numeric_limits<LayerIndex>::max(), 2, overflowing_counts),
                 std::invalid_argument);
}

TEST(PipelineLayoutTest, MapsGlobalLayersToStagesAndLocalIndices) {
    const std::vector<LayerIndex> counts{7, 5};
    const PipelineLayout layout = PipelineLayout::BuildCustomLayout(12, 2, counts);

    EXPECT_EQ(layout.GetStageForLayer(0), 0);
    EXPECT_EQ(layout.GetStageForLayer(6), 0);
    EXPECT_EQ(layout.GetStageForLayer(7), 1);
    EXPECT_EQ(layout.GetStageForLayer(11), 1);
    EXPECT_EQ(layout.GetLocalLayerIndex(0, 0), 0);
    EXPECT_EQ(layout.GetLocalLayerIndex(0, 6), 6);
    EXPECT_EQ(layout.GetLocalLayerIndex(1, 7), 0);
    EXPECT_EQ(layout.GetLocalLayerIndex(1, 11), 4);
}

TEST(PipelineLayoutInvalidTest, RejectsOutOfRangeQueries) {
    const std::vector<LayerIndex> counts{7, 5};
    const PipelineLayout layout = PipelineLayout::BuildCustomLayout(12, 2, counts);

    EXPECT_THROW((void)layout.GetStage(-1), std::out_of_range);
    EXPECT_THROW((void)layout.GetStage(2), std::out_of_range);
    EXPECT_THROW((void)layout.GetChunk(-1), std::out_of_range);
    EXPECT_THROW((void)layout.GetChunk(layout.GetNumChunks()), std::out_of_range);
    EXPECT_THROW((void)layout.GetStageForLayer(-1), std::out_of_range);
    EXPECT_THROW((void)layout.GetStageForLayer(12), std::out_of_range);
    EXPECT_THROW((void)layout.GetLocalLayerIndex(0, -1), std::out_of_range);
    EXPECT_THROW((void)layout.GetLocalLayerIndex(0, 12), std::out_of_range);
    EXPECT_THROW((void)layout.GetLocalLayerIndex(0, 7), std::out_of_range);
}

// Explicit expected ranges/owners are independent of the factory implementation.
// Check every layer, including the transition into each stage's later chunks.
void ExpectChunks(const PipelineLayout &layout, const std::vector<LayerIndex> &bounds, const std::vector<int> &owners) {
    ASSERT_EQ(layout.GetNumStages(), 2);
    ASSERT_EQ(bounds.size(), owners.size() + 1);
    ASSERT_EQ(layout.GetNumChunks(), static_cast<int>(owners.size()));
    std::vector<int> local_chunks(layout.GetNumStages(), 0);
    std::vector<LayerIndex> local_layers(layout.GetNumStages(), 0);
    for (int gid = 0; gid < layout.GetNumChunks(); ++gid) {
        SCOPED_TRACE(gid);
        const int owner = owners[gid];
        const auto &chunk = layout.GetChunk(gid);
        EXPECT_EQ(chunk.global_chunk_id, gid);
        EXPECT_EQ(chunk.stage_id, owner);
        EXPECT_EQ(chunk.local_chunk_idx, local_chunks[owner]);
        ASSERT_GT(layout.GetStage(owner).chunks.size(), static_cast<std::size_t>(local_chunks[owner]));
        EXPECT_EQ(&chunk, &layout.GetStage(owner).chunks[local_chunks[owner]++]);
        ExpectRange(chunk.layer_range, bounds[gid], bounds[gid + 1]);
        for (LayerIndex layer = bounds[gid]; layer < bounds[gid + 1]; ++layer) {
            EXPECT_EQ(layout.GetStageForLayer(layer), owner);
            EXPECT_EQ(layout.GetLocalLayerIndex(owner, layer), local_layers[owner]++);
            EXPECT_THROW((void)layout.GetLocalLayerIndex(1 - owner, layer), std::out_of_range);
        }
    }
    EXPECT_EQ(bounds.back(), layout.GetNumLayers());
    for (int stage = 0; stage < layout.GetNumStages(); ++stage) {
        EXPECT_EQ(layout.GetStage(stage).chunks.size(), static_cast<std::size_t>(local_chunks[stage]));
    }
}

TEST(PipelineLayoutTest, InterleavesUniformAndCustomChunks) {
    const auto uniform = PipelineLayout::BuildUniformLayout(13, 2, 2);
    ExpectChunks(uniform, {0, 4, 7, 10, 13}, {0, 1, 0, 1});
    const std::vector<LayerIndex> counts{2, 4, 3, 3};
    const auto custom = PipelineLayout::BuildCustomLayout(12, 2, counts, 2);
    ExpectChunks(custom, {0, 2, 6, 9, 12}, {0, 1, 0, 1});
    for (const auto *layout : {&uniform, &custom}) {
        EXPECT_EQ(layout->GetMaxLocalChunks(), 2);
        EXPECT_EQ(layout->GetInputStage(), 0);
        EXPECT_EQ(layout->GetOutputStage(), 1);
    }
}

TEST(PipelineLayoutTest, SupportsExplicitOwnersAndUnequalLocalChunkCounts) {
    // Both endpoints move to stage 1; consecutive chunks and revisiting an owner
    // must work without assuming round-robin placement or equal local counts.
    const std::vector<PipelineChunkSpec> specs{{1, 2}, {0, 3}, {0, 4}, {1, 3}, {1, 1}};
    const auto layout = PipelineLayout::BuildChunkLayout(13, 2, specs);
    ExpectChunks(layout, {0, 2, 5, 9, 12, 13}, {1, 0, 0, 1, 1});
    EXPECT_TRUE(layout.IsCustom());
    EXPECT_EQ(layout.GetMaxLocalChunks(), 3);
    EXPECT_EQ(layout.GetInputStage(), 1);
    EXPECT_EQ(layout.GetOutputStage(), 1);
    for (int stage = 0; stage < 2; ++stage) {
        EXPECT_EQ(layout.GetStage(stage).has_embedding, stage == 1);
        EXPECT_EQ(layout.GetStage(stage).has_final_norm, stage == 1);
        EXPECT_EQ(layout.GetStage(stage).has_lm_head, stage == 1);
    }
}

TEST(PipelineLayoutTest, PreservesEmptyAndUnderfilledDefaultMetadata) {
    const auto empty = PipelineLayout::BuildUniformLayout(0, 2, 2);
    EXPECT_EQ(empty.GetNumChunks(), 0);
    EXPECT_EQ(empty.GetMaxLocalChunks(), 0);
    EXPECT_EQ(empty.GetInputStage(), 0);
    EXPECT_EQ(empty.GetOutputStage(), 1);
    EXPECT_THROW((void)empty.GetChunk(0), std::out_of_range);
    EXPECT_THROW((void)empty.GetStageForLayer(0), std::out_of_range);
    EXPECT_THROW((void)empty.GetLocalLayerIndex(0, 0), std::out_of_range);

    const auto partial = PipelineLayout::BuildUniformLayout(1, 2, 2);
    EXPECT_EQ(partial.GetNumChunks(), 1);
    EXPECT_EQ(partial.GetMaxLocalChunks(), 1);
    EXPECT_TRUE(partial.GetStage(1).chunks.empty());
    ExpectRange(partial.GetChunk(0).layer_range, 0, 1);
    EXPECT_EQ(partial.GetStageForLayer(0), 0);
    EXPECT_EQ(partial.GetLocalLayerIndex(0, 0), 0);
    EXPECT_THROW((void)partial.GetLocalLayerIndex(1, 0), std::out_of_range);
    EXPECT_THROW((void)partial.GetChunk(1), std::out_of_range);
    EXPECT_EQ(partial.GetInputStage(), 0);
    EXPECT_EQ(partial.GetOutputStage(), 1); // Legacy metadata, not empty-stage training.
}

TEST(PipelineLayoutInvalidTest, RejectsInvalidExplicitLayouts) {
    struct Case {
        const char *reason;
        std::vector<PipelineChunkSpec> specs;
    };
    const std::vector<Case> cases{
        {"empty list", {}},
        {"negative owner", {{-1, 6}, {1, 6}}},
        {"owner out of range", {{0, 6}, {2, 6}}},
        {"zero layer count", {{0, 0}, {1, 12}}},
        {"negative layer count", {{0, -1}, {1, 13}}},
        {"sum too small", {{0, 5}, {1, 6}}},
        {"sum too large", {{0, 7}, {1, 6}}},
        {"empty stage", {{0, 6}, {0, 6}}},
    };
    for (const auto &test : cases) {
        SCOPED_TRACE(test.reason);
        EXPECT_THROW((void)PipelineLayout::BuildChunkLayout(12, 2, test.specs), std::invalid_argument);
    }
    const auto max = std::numeric_limits<LayerIndex>::max();
    const std::vector<PipelineChunkSpec> overflow{{0, max}, {1, 1}};
    EXPECT_THROW((void)PipelineLayout::BuildChunkLayout(max, 2, overflow), std::invalid_argument);
    const std::vector<PipelineChunkSpec> valid{{0, 6}, {1, 6}};
    EXPECT_THROW((void)PipelineLayout::BuildChunkLayout(0, 2, valid), std::invalid_argument);
    EXPECT_THROW((void)PipelineLayout::BuildChunkLayout(12, 0, valid), std::invalid_argument);
}

} // namespace
} // namespace infini_train::nn::parallel
