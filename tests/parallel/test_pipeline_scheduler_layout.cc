#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/nn/parallel/pipeline_layout.h"
#include "infini_train/include/nn/parallel/pp/pipeline_schedule.h"

namespace infini_train::nn::parallel {

TEST(PipelineSchedulerLayoutTest, GPipeTasksUseLayoutOwnership) {
    const auto layout = PipelineLayout::BuildDefault(/*num_layers=*/12, /*num_stages=*/3, /*vpp_size=*/2);
    const auto schedule = PipelineParallelScheduler::GenerateGPipeSchedule(
        /*n=*/2, /*num_stages=*/3, /*vpp_size=*/2, layout);

    ASSERT_EQ(schedule.size(), 2U * 2U * 3U * 2U);
    for (const auto &task : schedule) {
        const auto &chunk = layout.chunk(task.global_chunk_id);
        EXPECT_EQ(task.stage_id, chunk.stage_id);
        EXPECT_EQ(task.local_chunk_idx, chunk.local_chunk_id);
    }
}

TEST(PipelineSchedulerLayoutTest, InterleavedTasksUseLayoutOwnership) {
    const auto layout = PipelineLayout::BuildDefault(/*num_layers=*/12, /*num_stages=*/3, /*vpp_size=*/2);
    const auto schedule = PipelineParallelScheduler::GenerateInterleaved1F1BSchedule(
        /*n=*/2, /*num_stages=*/3, /*vpp_size=*/2, layout);

    // Every microbatch traverses every global chunk once in each direction.
    ASSERT_EQ(schedule.size(), 2U * 3U * 2U * 2U);
    for (const auto &task : schedule) {
        const auto &chunk = layout.chunk(task.global_chunk_id);
        EXPECT_EQ(task.stage_id, chunk.stage_id);
        EXPECT_EQ(task.local_chunk_idx, chunk.local_chunk_id);
    }
}

TEST(PipelineSchedulerLayoutTest, CreateTaskSupportsContiguousCustomLayout) {
    const auto layout = PipelineLayout::BuildContiguous({2, 4, 3, 3});

    const auto task = PipelineParallelScheduler::CreateTask(
        /*step=*/7, /*mb=*/1, /*global_chunk=*/2, /*num_stages=*/4,
        /*total_chunks=*/4, /*is_forward=*/true, layout);

    EXPECT_EQ(task.global_chunk_id, 2);
    EXPECT_EQ(task.stage_id, 2);
    EXPECT_EQ(task.local_chunk_idx, 0);
    EXPECT_TRUE(task.is_forward);
}

TEST(PipelineSchedulerLayoutTest, ExplicitChunkStageMapDrivesSchedulerOwnership) {
    const std::vector<ChunkLayout> chunks{
        {.global_chunk_id = 0, .stage_id = 1, .local_chunk_id = 0, .layers = {0, 2}},
        {.global_chunk_id = 1, .stage_id = 0, .local_chunk_id = 0, .layers = {2, 3}},
        {.global_chunk_id = 2, .stage_id = 1, .local_chunk_id = 1, .layers = {3, 5}},
        {.global_chunk_id = 3, .stage_id = 0, .local_chunk_id = 1, .layers = {5, 6}},
    };
    const auto layout = PipelineLayout::BuildExplicit(6, 2, 2, chunks);
    const auto schedule = PipelineParallelScheduler::GenerateGPipeSchedule(1, 2, 2, layout);

    ASSERT_EQ(schedule.size(), 8U);
    for (const auto &task : schedule) {
        EXPECT_EQ(task.stage_id, layout.chunk(task.global_chunk_id).stage_id);
        EXPECT_EQ(task.local_chunk_idx, layout.chunk(task.global_chunk_id).local_chunk_id);
    }
}

TEST(PipelineSchedulerLayoutTest, RejectsTopologyMismatch) {
    const auto layout = PipelineLayout::BuildDefault(/*num_layers=*/12, /*num_stages=*/3, /*vpp_size=*/2);

    EXPECT_THROW(PipelineParallelScheduler::GenerateGPipeSchedule(/*n=*/2, /*num_stages=*/2, /*vpp_size=*/2, layout),
                 PipelineLayoutError);
}

} // namespace infini_train::nn::parallel
