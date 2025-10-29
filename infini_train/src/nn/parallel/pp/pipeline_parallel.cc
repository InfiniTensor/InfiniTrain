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
PipelineParallel::CreateOptimizers(const std::vector<std::vector<std::shared_ptr<Module>>> &stage_layers, float lr) {
    std::vector<std::shared_ptr<Optimizer>> optims;
    optims.reserve(stage_layers.size());

    for (int i = 0; i < num_stages_; ++i) {
        std::vector<std::shared_ptr<Tensor>> params;
        for (const auto &layer : stage_layers[i]) {
            layer->To(devices_[i]);
            auto layer_params = layer->Parameters();
            params.insert(params.end(), layer_params.begin(), layer_params.end());
        }
        optims.push_back(std::make_shared<optimizers::SGD>(params, lr));
    }
    return optims;
}

void PipelineParallel::BuildPipelineStages(const std::vector<std::vector<std::shared_ptr<Module>>> &stage_layers,
                                           const std::vector<std::shared_ptr<Optimizer>> &optimizers,
                                           const std::vector<std::vector<int64_t>> &recv_shape) {
    for (int i = 0; i < num_stages_; ++i) {
        auto stage = std::make_shared<PipelineStage>(stage_layers[i], i, num_stages_, recv_shape, optimizers[i]);
        pipeline_stages_.push_back(stage);
    }
}

void PipelineParallel::SplitModel(const std::vector<std::vector<int64_t>> &recv_shape, float lr) {
    auto layers = original_model_->GetPipelineLayers();
    printf("层数: %ld\n", layers.size());
    CHECK(!layers.empty()) << "SplitModel: GetPipelineLayers returned empty vector";
    

    auto stage_layer = SplitLayersIntoStages(layers);

    auto optimizer = CreateOptimizers(stage_layer, lr);

    BuildPipelineStages(stage_layer, optimizer, recv_shape);
}

void PipelineParallel::SetupSchedule(int num_microbatches) {

    schedule_ = std::make_shared<ScheduleGPipe>(pipeline_stages_[rank_], num_stages_, num_microbatches, rank_);  
}

PipelineParallel::PipelineParallel(const std::shared_ptr<Module> &model, int num_gpus, int num_microbatches,
                                   const std::vector<std::vector<int64_t>> &recv_shape, float learning_rate, int rank)
    : original_model_(model), devices_(DeviceManager::Instance()->GetAllAvailableDevices(DeviceType::kCUDA)),
      num_stages_(num_gpus), rank_(rank) {
    CHECK(!devices_.empty()) << "Devices list is empty";

    SplitModel(recv_shape, learning_rate);

    SetupSchedule(num_microbatches);
}

float PipelineParallel::TrainStep(const std::vector<std::shared_ptr<Tensor>> &input,
                                  const std::vector<std::shared_ptr<Tensor>> &target,
                                  const std::shared_ptr<Module> &loss_fn) {
    std::shared_ptr<Tensor> stage_input;
    std::shared_ptr<Tensor> stage_target = target[0];
    if (rank_ == 0) {
        stage_input = input[0];
    }

    auto stage_loss = schedule_->Step(stage_input, stage_target, loss_fn);
    float lossf = 0.0f;
    if (rank_ == num_stages_ - 1) {
        lossf = stage_loss;
    }

    return lossf;
}

} // namespace infini_train::nn::parallel
