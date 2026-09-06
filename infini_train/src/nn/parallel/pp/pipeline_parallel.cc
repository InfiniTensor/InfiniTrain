// pipeline_parallel.cc
#include "infini_train/include/nn/parallel/pp/pipeline_parallel.h"

#include <cstdint>
#include <memory>
#include <string>

#include "infini_train/include/nn/modules/container.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/pp/pipeline_schedule.h"
#include "infini_train/include/nn/parallel/pp/pipeline_stage.h"

namespace infini_train::nn::parallel {
namespace {
constexpr char kModuleName[] = "module";
} // namespace

thread_local int pp_rank = 0;

void PipelineParallel::BuildPipelineStage(const std::vector<std::vector<int64_t>> &recv_shape, Device device,
                                          std::vector<std::shared_ptr<Module>> &&chunks) {
    pipeline_stage_ = std::make_shared<PipelineStage>(rank_, num_stages_, recv_shape, device, std::move(chunks));
}

void PipelineParallel::SetupSchedule(int num_micro_batches) {
    schedule_ = std::make_shared<PipelineSchedule>(pipeline_stage_, num_stages_, num_micro_batches);
}

float PipelineParallel::TrainStep(const std::vector<std::shared_ptr<Tensor>> &input,
                                  const std::vector<std::shared_ptr<Tensor>> &target,
                                  const std::shared_ptr<Optimizer> &optimizer, const std::shared_ptr<Module> &loss_fn,
                                  DataType dtype) {
    std::shared_ptr<Tensor> stage_input;
    std::shared_ptr<Tensor> stage_target = target[0];
    if (rank_ == 0) {
        stage_input = input[0];
    }

    return schedule_->Step(stage_input, stage_target, optimizer, loss_fn, dtype);
}

PipelineParallel::PipelineParallel(const std::shared_ptr<Module> module, int num_stages, int num_micro_batches,
                                   const std::vector<std::vector<int64_t>> &recv_shape, int pp_rank, Device device,
                                   const StageInfo &stage_info)
    : num_stages_(num_stages), rank_(pp_rank) {
    modules_[kModuleName] = std::move(module);

    const int chunk_size = static_cast<int>(stage_info.layer_ranges_per_chunk.size());

    std::vector<std::shared_ptr<Module>> chunks;
    for (int chunk_id = 0; chunk_id < chunk_size; ++chunk_id) {
        std::vector<std::shared_ptr<Module>> chunk_parts;
        if (chunk_id == 0 && stage_info.is_first_stage) {
            chunk_parts.push_back(module->mutable_module(kPPFirstStageName));
        }
        chunk_parts.push_back(module->mutable_module(kPPChunkNamePrefix + std::to_string(chunk_id)));
        if (chunk_id == chunk_size - 1 && stage_info.is_last_stage) {
            chunk_parts.push_back(module->mutable_module(kPPLastStageName));
        }
        chunks.push_back(std::make_shared<Sequential>(std::move(chunk_parts)));
    }

    BuildPipelineStage(recv_shape, device, std::move(chunks));

    SetupSchedule(num_micro_batches);
}

std::vector<std::shared_ptr<Module>> *PipelineParallel::mutable_chunks() { return pipeline_stage_->mutable_chunks(); }
} // namespace infini_train::nn::parallel
