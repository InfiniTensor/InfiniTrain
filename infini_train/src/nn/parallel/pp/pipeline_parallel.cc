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
                                          const std::vector<std::vector<int64_t>> &recv_shape) {
    pipeline_stage_ = std::make_shared<PipelineStage>(module, rank_, num_stages_, recv_shape, optimizer);
}

void PipelineParallel::SetupSchedule(int num_micro_batches) {
    schedule_ = std::make_shared<ScheduleGPipe>(pipeline_stage_, num_stages_, num_micro_batches, rank_);
}

float PipelineParallel::TrainStep(const std::vector<std::shared_ptr<Tensor>> &input,
                                  const std::vector<std::shared_ptr<Tensor>> &target,
                                  const std::shared_ptr<Module> &loss_fn) {
    std::shared_ptr<Tensor> stage_input;
    std::shared_ptr<Tensor> stage_target = target[0];
    if (rank_ == 0) {
        stage_input = input[0];
    }

    return schedule_->Step(stage_input, stage_target, loss_fn);
}

std::tuple<bool, bool, int, int> PipelineParallel::GetStageInfo(int total_layers, int pp_size) {
    int rank = pp_rank;
    bool is_first_stage = (pp_rank == 0);
    bool is_last_stage = (pp_rank == pp_size - 1);

    int layers_per_stage = total_layers / pp_size;
    int remainder = total_layers % pp_size;
    int start_layer, end_layer;
    if (pp_rank < remainder) {
        start_layer = pp_rank * (layers_per_stage + 1);
        end_layer = start_layer + layers_per_stage + 1;
    } else {
        start_layer = pp_rank * layers_per_stage + remainder;
        end_layer = start_layer + layers_per_stage;
    }

    return {is_first_stage, is_last_stage, start_layer, end_layer};
}

PipelineParallel::PipelineParallel(const std::shared_ptr<Module> module, int num_stages, int num_micro_batches,
                                   const std::vector<std::vector<int64_t>> &recv_shape, int rank,
                                   const std::shared_ptr<Optimizer> &optimizer)
    : num_stages_(num_stages), rank_(rank) {
    modules_[kModuleName] = std::move(module);

    BuildPipelineStage(module, optimizer, recv_shape);

    SetupSchedule(num_micro_batches);
}

} // namespace infini_train::nn::parallel
