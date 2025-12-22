// pipeline_parallel.cc
#include "infini_train/include/nn/parallel/pp/pipeline_parallel.h"

#include <cstdint>
#include <memory>

#include "infini_train/include/nn/modules/container.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/pp/pipeline_schedule.h"
#include "infini_train/include/nn/parallel/pp/pipeline_stage.h"
#include "infini_train/include/optimizer.h"

namespace infini_train::nn::parallel {
namespace {
constexpr char kModuleName[] = "module";
} // namespace

thread_local int pp_rank = 0;

void PipelineParallel::BuildPipelineStage(const std::shared_ptr<Module> &module,
                                          const std::shared_ptr<Optimizer> &optimizer,
                                          const std::vector<std::vector<int64_t>> &recv_shape, int device_id) {
    pipeline_stage_ = std::make_shared<PipelineStage>(module, rank_, num_stages_, recv_shape, optimizer, device_id);
}

void PipelineParallel::SetupSchedule(int num_micro_batches) {
    schedule_ = std::make_shared<PipelineSchedule>(pipeline_stage_, num_stages_, num_micro_batches);
}

float PipelineParallel::TrainStep(const std::vector<std::shared_ptr<Tensor>> &input,
                                  const std::vector<std::shared_ptr<Tensor>> &target,
                                  const std::shared_ptr<Module> &loss_fn, DataType dtype) {
    std::shared_ptr<Tensor> stage_input;
    std::shared_ptr<Tensor> stage_target = target[0];
    if (rank_ == 0) {
        stage_input = input[0];
    }

    return schedule_->Step(stage_input, stage_target, loss_fn, dtype);
}

StageInfo PipelineParallel::GetStageInfo(int total_layers, int pp_size, int chunks_per_stage) {
    int rank = pp_rank;
    bool is_first_stage = (pp_rank == 0);
    bool is_last_stage = (pp_rank == pp_size - 1);

    std::vector<std::pair<int, int>> layer_chunks;

    int layers_per_chunk = total_layers / (pp_size * chunks_per_stage);
    int remainder = total_layers % (pp_size * chunks_per_stage);

    for (int local_chunk_idx = 0; local_chunk_idx < chunks_per_stage; ++local_chunk_idx) {
        int global_chunk_idx = local_chunk_idx * pp_size + rank;

        if (global_chunk_idx * layers_per_chunk >= total_layers) {
            break;
        }

        int chunk_start = global_chunk_idx * layers_per_chunk;
        int chunk_end = chunk_start + layers_per_chunk;

        if (global_chunk_idx < remainder) {
            // Assign an additional layer to each of the first remainder chunks
            chunk_start = global_chunk_idx * (layers_per_chunk + 1);
            chunk_end = chunk_start + (layers_per_chunk + 1);
        } else {
            chunk_start = remainder * (layers_per_chunk + 1) + (global_chunk_idx - remainder) * layers_per_chunk;
            chunk_end = chunk_start + layers_per_chunk;
        }

        chunk_end = std::min(chunk_end, total_layers);
        if (chunk_start < chunk_end) {
            layer_chunks.push_back({chunk_start, chunk_end});
        }
    }

    return {is_first_stage, is_last_stage, layer_chunks};
}

PipelineParallel::PipelineParallel(const std::shared_ptr<Module> module, int num_stages, int num_micro_batches,
                                   const std::vector<std::vector<int64_t>> &recv_shape, int pp_rank,
                                   const std::shared_ptr<Optimizer> &optimizer, int device_id)
    : num_stages_(num_stages), rank_(pp_rank) {
    modules_[kModuleName] = std::move(module);

    BuildPipelineStage(module, optimizer, recv_shape, device_id);

    SetupSchedule(num_micro_batches);
}

} // namespace infini_train::nn::parallel
