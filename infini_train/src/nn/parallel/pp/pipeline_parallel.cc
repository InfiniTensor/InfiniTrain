// pipeline_parallel.cc
#include "infini_train/include/nn/parallel/pp/pipeline_parallel.h"

#include <cctype>
#include <memory>

#include "infini_train/include/nn/modules/container.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/pp/pipeline_schedule.h"
#include "infini_train/include/nn/parallel/pp/pipeline_stage.h"
#include "infini_train/include/optimizer.h"

namespace infini_train::nn::parallel {

std::vector<std::vector<std::shared_ptr<Module>>>
PipelineParallel::SplitLayersIntoStages(std::vector<std::shared_ptr<Module>> layers) {
    const int total_layers = layers.size();
    CHECK_GT(total_layers, 0) << "Model has no layers to split!";
    CHECK_GE(num_stages_, 1) << "num_stages must be >= 1";
    CHECK_LE(num_stages_, total_layers) << "num_stages (" << num_stages_ << ") cannot be greater than total layers ("
                                        << total_layers << ")";

    std::vector<std::vector<std::shared_ptr<Module>>> stages(num_stages_);
    int base_layers_per_stage = total_layers / num_stages_;
    int remainder = total_layers % num_stages_;
    int layer_idx = 0;

    for (int s = 0; s < num_stages_; ++s) {
        int layers_in_this_stage = base_layers_per_stage + (s < remainder ? 1 : 0);
        for (int i = 0; i < layers_in_this_stage; ++i) {
            auto layer = layers[layer_idx];
            stages[s].emplace_back(layer);
            layer_idx++;
        }
    }

    return stages;
}

std::vector<std::shared_ptr<Optimizer>>
PipelineParallel::CreateOptimizers(const std::vector<std::vector<std::shared_ptr<Module>>> &stage_layers,
                                   OptimizerFactory optimizer_factory) {
    std::vector<std::shared_ptr<Optimizer>> optims;
    optims.reserve(stage_layers.size());

    for (int s = 0; s < num_stages_; ++s) {
        std::vector<std::shared_ptr<Tensor>> params;
        for (const auto &layer : stage_layers[s]) {
            layer->To(devices_[s]);
            auto layer_params = layer->Parameters();
            params.insert(params.end(), layer_params.begin(), layer_params.end());
        }

        auto optim = optimizer_factory(params);
        CHECK(optim != nullptr) << "Optimizer factory returned null optimizer for stage " << s;
        optims.push_back(std::move(optim));
    }
    return optims;
}

void PipelineParallel::BuildPipelineStage(const std::vector<std::vector<std::shared_ptr<Module>>> &stage_layers,
                                          const std::vector<std::shared_ptr<Optimizer>> &optimizers,
                                          const std::vector<std::vector<int64_t>> &recv_shape) {
    pipeline_stage_
        = std::make_shared<PipelineStage>(stage_layers[rank_], rank_, num_stages_, recv_shape, optimizers[rank_]);
}

void PipelineParallel::SplitModel(const std::vector<std::vector<int64_t>> &recv_shape,
                                  OptimizerFactory optimizer_factory) {
    auto layers = original_model_->GetPipelineLayers();
    CHECK(!layers.empty()) << "SplitModel: GetPipelineLayers returned empty vector";

    auto stage_layer = SplitLayersIntoStages(layers);

    auto optimizer = CreateOptimizers(stage_layer, optimizer_factory);

    BuildPipelineStage(stage_layer, optimizer, recv_shape);
}

void PipelineParallel::SetupSchedule(int num_microbatches) {
    schedule_ = std::make_shared<ScheduleGPipe>(pipeline_stage_, num_stages_, num_microbatches, rank_);
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

PipelineParallel::PipelineParallel(const std::shared_ptr<Module> &model, int num_stages, int num_microbatches,
                                   const std::vector<std::vector<int64_t>> &recv_shape, int rank,
                                   OptimizerFactory optimizer_factory)
    : original_model_(model), devices_(DeviceManager::Instance()->GetAllAvailableDevices(DeviceType::kCUDA)),
      num_stages_(num_stages), rank_(rank) {
    CHECK(!devices_.empty()) << "Devices list is empty";

    SplitModel(recv_shape, optimizer_factory);

    SetupSchedule(num_microbatches);
}

} // namespace infini_train::nn::parallel
