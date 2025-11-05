// pipeline_parallel.h
#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/tensor.h"

#include "infini_train/include/nn/parallel/pp/pipeline_schedule.h"
#include "infini_train/include/nn/parallel/pp/pipeline_stage.h"

namespace infini_train::nn::parallel {

using OptimizerFactory = std::function<std::shared_ptr<Optimizer>(const std::vector<std::shared_ptr<Tensor>> &params)>;

class PipelineParallel : public Module {
public:
    PipelineParallel(const std::shared_ptr<Module> &model, int num_stages, int num_microbatches,
                     const std::vector<std::vector<int64_t>> &recv_shape, int rank, OptimizerFactory optimizer_factory);

    float TrainStep(const std::vector<std::shared_ptr<Tensor>> &input,
                    const std::vector<std::shared_ptr<Tensor>> &target, const std::shared_ptr<Module> &loss_fn);

private:
    int num_stages_;
    int rank_;
    std::vector<const Device *> devices_;
    std::shared_ptr<Module> original_model_;
    std::shared_ptr<PipelineStage> pipeline_stage_;
    std::shared_ptr<PipelineSchedule> schedule_;

    std::vector<std::vector<std::shared_ptr<Module>>>
    SplitLayersIntoStages(std::vector<std::shared_ptr<Module>> layers);

    void SplitModel(const std::vector<std::vector<int64_t>> &recv_shape, OptimizerFactory optimizer_factory);

    std::vector<std::shared_ptr<Optimizer>>
    CreateOptimizers(const std::vector<std::vector<std::shared_ptr<Module>>> &stage_layers,
                     OptimizerFactory optimizer_factory);

    void BuildPipelineStage(const std::vector<std::vector<std::shared_ptr<Module>>> &stage_layers,
                            const std::vector<std::shared_ptr<Optimizer>> &optimizers,
                            const std::vector<std::vector<int64_t>> &recv_shape);

    void SetupSchedule(int num_microbatches);
};

} // namespace infini_train::nn::parallel
