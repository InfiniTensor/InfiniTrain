// pipeline_parallel.cc
#include "infini_train/include/nn/parallel/pp/pipeline_parallel.h"

#include <cstdint>
#include <memory>
#include <string>

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

void PipelineParallel::BuildPipelineStage(const std::shared_ptr<Optimizer> &optimizer,
                                          const std::vector<std::vector<int64_t>> &recv_shape, int device_id,
                                          std::vector<std::shared_ptr<Module>> &&chunks) {
    pipeline_stage_
        = std::make_shared<PipelineStage>(rank_, num_stages_, recv_shape, optimizer, device_id, std::move(chunks));
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

StageInfo PipelineParallel::GetStageInfo(int total_layers, int pp_size, int rank, int chunks_per_stage) {
    bool is_first_stage = (rank == 0);
    bool is_last_stage = (rank == pp_size - 1);

    std::vector<std::pair<int, int>> layer_ranges_per_chunk;

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
            layer_ranges_per_chunk.push_back({chunk_start, chunk_end});
        }
    }

    return {is_first_stage, is_last_stage, layer_ranges_per_chunk};
}

PipelineParallel::PipelineParallel(const std::shared_ptr<Module> module, int num_stages, int num_micro_batches,
                                   const std::vector<std::vector<int64_t>> &recv_shape, int pp_rank,
                                   const std::shared_ptr<Optimizer> &optimizer, int device_id, int chunk_size)
    : num_stages_(num_stages), rank_(pp_rank) {
    modules_[kModuleName] = std::move(module);

    int stage_id = pp_rank;
    int stage_size = num_stages;

    std::vector<std::shared_ptr<Module>> chunks;
    for (int chunk_id = 0; chunk_id < chunk_size; ++chunk_id) {
        std::vector<std::shared_ptr<Module>> chunk_parts;
        if (chunk_id == 0 && stage_id == 0) {
            chunk_parts.push_back(module->mutable_module(kPPFirstStageName));
        }
        chunk_parts.push_back(module->mutable_module(kPPChunkNamePrefix + std::to_string(chunk_id)));
        if (chunk_id == chunk_size - 1 && stage_id == stage_size - 1) {
            chunk_parts.push_back(module->mutable_module(kPPLastStageName));
        }
        chunks.push_back(std::make_shared<Sequential>(std::move(chunk_parts)));
    }

    BuildPipelineStage(optimizer, recv_shape, device_id, std::move(chunks));

    SetupSchedule(num_micro_batches);
}

std::vector<std::shared_ptr<Module>> *PipelineParallel::mutable_chunks() { return pipeline_stage_->mutable_chunks(); }
} // namespace infini_train::nn::parallel
