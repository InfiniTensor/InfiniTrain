// pipeline_parallel.h
#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/tensor.h"

#include "infini_train/include/nn/parallel/pp/pipeline_schedule.h"
#include "infini_train/include/nn/parallel/pp/pipeline_stage.h"

namespace infini_train::nn::pipeline {

class PipelineParallel : public Module {
public:
    PipelineParallel(const std::shared_ptr<Module> &model, int num_gpus, int num_microbatches,
                     const std::vector<std::vector<int64_t>> &recv_shape, float learning_rate);

    float TrainStep(const std::vector<std::shared_ptr<Tensor>> &input,
                    const std::vector<std::shared_ptr<Tensor>> &target, const std::shared_ptr<Module> &loss_fn);

private:
    int num_stages_;
    int rank_;
    std::vector<const Device *> devices_;
    std::shared_ptr<Module> original_model_;
    std::vector<std::shared_ptr<PipelineStage>> pipeline_stages_;
    std::vector<std::shared_ptr<PipelineSchedule>> schedules_;

    std::vector<std::vector<std::shared_ptr<Module>>>
    SplitLayersIntoStages(std::vector<std::shared_ptr<Module>> layers);

    void SplitModel(const std::vector<std::vector<int64_t>> &recv_shape, float learning_rate);

    std::vector<std::shared_ptr<Optimizer>>
    CreateOptimizers(const std::vector<std::vector<std::shared_ptr<Module>>> &stage_layers, float lr);

    void BuildPipelineStages(const std::vector<std::vector<std::shared_ptr<Module>>> &stage_layers,
                             const std::vector<std::shared_ptr<Optimizer>> &optimizers,
                             const std::vector<std::vector<int64_t>> &recv_shape);

    void SetupSchedules(int num_microbatches);
};

} // namespace infini_train::nn::pipeline