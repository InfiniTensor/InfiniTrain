// pipeline_parallel.cc
#include "infini_train/include/nn/parallel/pp/pipeline_parallel.h"

#include <memory>

#include "infini_train/include/nn/modules/container.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/pp/pipeline_schedule.h"
#include "infini_train/include/nn/parallel/pp/pipeline_stage.h"
#include "infini_train/include/optimizer.h"
namespace infini_train::nn::pipeline {
namespace {

std::vector<std::vector<std::shared_ptr<Module>>> SplitLayersIntoStages(std::vector<std::shared_ptr<Module>> layers,
                                                                        int num_stages) {
    // printf("SplitLayersIntoStages Enter!\n");
    const int total_layers = layers.size();
    CHECK_GT(total_layers, 0) << "Model has no layers to split!";
    CHECK_GE(num_stages, 1) << "num_stages must be >= 1";
    CHECK_LE(num_stages, total_layers) << "num_stages (" << num_stages << ") cannot be greater than total layers ("
                                       << total_layers << ")";

    std::vector<std::vector<std::shared_ptr<Module>>> stages(num_stages);

    int base_layers_per_stage = total_layers / num_stages;
    int remainder = total_layers % num_stages;
    printf("base_layers_per_stage: %d, remainder: %d num_stages: %d layers层数:%d\n", base_layers_per_stage, remainder,
           num_stages, total_layers);

    int layer_idx = 0;
    for (int s = 0; s < num_stages; ++s) {
        int layers_in_this_stage = base_layers_per_stage + (s < remainder ? 1 : 0);
        for (int i = 0; i < layers_in_this_stage; i++) {
            auto layer = layers[layer_idx];
            stages[s].emplace_back(layer);
            layer_idx++;
        }
    }

    // printf("SplitLayersIntoStages   OK!\n");
    return stages;
}

} // namespace
PipelineParallel::PipelineParallel(const std::shared_ptr<Module> &model, int num_gpus, int num_microbatches,
                                   const int batch_size, const int seq_len, const int hidden_size, float learning_rate)
    : original_model_(model), devices_(DeviceManager::Instance()->GetAllAvailableDevices(DeviceType::kCUDA)),
      num_stages_(num_gpus) {
    CHECK(!devices_.empty()) << "Devices list is empty";

    printf("PipelineParallel entry!! devices 个数: %ld\n", devices_.size());
    SplitModel(batch_size, seq_len, hidden_size, learning_rate);
    SetupSchedules(num_microbatches);
}

void PipelineParallel::SplitModel(const int batch_size, const int seq_len, const int hidden_size, float learning_rate) {
    auto layers = original_model_->GetPipelineLayers();
    if (layers.empty()) {
        LOG(INFO) << "SplitModel: GetPipelineLayers returned empty vector!";
    }
    auto stage_layers = SplitLayersIntoStages(layers, num_stages_);

    std::vector<std::shared_ptr<Optimizer>> optims;
    for (int i = 0; i < num_stages_; i++) {
        std::vector<std::shared_ptr<Tensor>> stage_params;
        for (auto layer : stage_layers[i]) {
            layer->To(devices_[i]);
            auto layer_params = layer->Parameters();
            stage_params.insert(stage_params.end(), layer_params.begin(), layer_params.end());
        }
        optims.push_back(std::make_shared<optimizers::SGD>(stage_params, learning_rate));
    }

    ActivationShape recv_shape{.batch_size = batch_size, .seq_len = seq_len, .hidden_size = hidden_size};

    for (int s = 0; s < num_stages_; ++s) {
        printf("before PipelineStage !!!!\n");
        auto stage = std::make_shared<PipelineStage>(stage_layers[s], s, num_stages_, recv_shape, optims[s]);
        printf("after PipelineStage");
        pipeline_stages_.push_back(stage);
    }
}

void PipelineParallel::SetupSchedules(int num_microbatches) {
    printf("SetupSchedules enter %ld %d\n", pipeline_stages_.size(), num_stages_);
    for (int stage_idx = 0; stage_idx < num_stages_; ++stage_idx) {
        auto schedule
            = std::make_shared<ScheduleGPipe>(pipeline_stages_[stage_idx], num_stages_, num_microbatches, stage_idx);
        schedules_.push_back(schedule);
    }
}

float PipelineParallel::TrainStep(const std::vector<std::shared_ptr<Tensor>> &input,
                                  const std::vector<std::shared_ptr<Tensor>> &target,
                                  const std::shared_ptr<Module> &loss_fn) {
    // printf("[TrainStep] num_stages_=%d  input.size=%ld  target.size=%ld\n",
    //        num_stages_, input[0]->SizeInBytes(), target.size());
    std::vector<float> local_losses(num_stages_);
    std::vector<std::thread> stage_threads;
    stage_threads.reserve(num_stages_);

    // printf("[TrainStep] schedules_.size()=%zu\n", schedules_.size());
    for (int s = 0; s < num_stages_; ++s) { printf("  schedules_[%d] = %p\n", s, (void *)schedules_[s].get()); }
    for (int si = 0; si < num_stages_; ++si) {
        auto schedule = schedules_[si];

        // printf("[TrainStep] launch thread for stage %d\n", si);
        stage_threads.emplace_back([si, schedule, input, target, loss_fn, &local_losses, this]() {
            devices_[si]->SetDevice();

            std::shared_ptr<Tensor> stage_input;
            std::shared_ptr<Tensor> stage_target = target[0];
            if (si == 0) {
                stage_input = input[0];
                printf("[stage 0] use global input\n");
            } else {
                printf("[stage %d] input will be received from prev stage\n", si);
            }

            auto stage_losses = schedule->Step(stage_input, stage_target, loss_fn);
            // printf("[stage %d] Step returned %f losses\n", si, stage_losses);

            local_losses[si] = stage_losses;
        });
    }

    for (auto &t : stage_threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    float total_loss = 0.0f;
    for (int s = 0; s < num_stages_; ++s) {
        printf("[TrainStep] stage %d collected %lf losses\n", s, local_losses[s]);
        total_loss += local_losses[s];
    }

    // printf("[TrainStep] total_count=%d  total_loss=%.6f\n", total_count, total_loss);

    return total_loss / num_stages_;
}

} // namespace infini_train::nn::pipeline