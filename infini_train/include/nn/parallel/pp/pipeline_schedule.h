#pragma once
#include <vector>
#include <memory>

#include "infini_train/include/nn/parallel/pp/pipeline_stage.h"

namespace infini_train::nn::pipeline {

class PipelineSchedule {
public:
    PipelineSchedule(
        std::shared_ptr<PipelineStage> stage, int num_stages, int num_microbatches, int stage_index) :
        stage_(std::move(stage)), num_stages_(num_stages), num_microbatches_(num_microbatches),
        stage_index_(stage_index) {}

    virtual ~PipelineSchedule() = default;

    void Step(
        const std::vector<std::shared_ptr<Tensor>>& input,
        const std::vector<std::shared_ptr<Tensor>>& target,
        const std::shared_ptr<Module>& loss_fn,
        std::vector<std::shared_ptr<Tensor>>* losses = nullptr
    );

    int NumMicrobatches() const {
        return num_microbatches_;
    }

    std::shared_ptr<PipelineStage> stage() const {
        return stage_;
    }

protected:
    std::vector<std::vector<std::shared_ptr<Tensor>>> SplitTensor(
        const std::vector<std::shared_ptr<Tensor>>& full_inputs);

    virtual void StepMicrobatches(
        const std::vector<std::vector<std::shared_ptr<Tensor>>>& arg_mbs,
        const std::vector<std::shared_ptr<Tensor>>& target_mbs,
        const std::shared_ptr<Module>& loss_fn,
        std::vector<std::shared_ptr<Tensor>>* losses) = 0;

    std::shared_ptr<PipelineStage> stage_;
    int num_stages_;
    int num_microbatches_;
    int stage_index_;
};

class Schedule1F1B : public PipelineSchedule {
public:
    Schedule1F1B(std::shared_ptr<PipelineStage> stage, int num_microbatches) : PipelineSchedule(stage, num_microbatches){};

     void StepMicrobatches(
        const std::vector<std::vector<std::shared_ptr<Tensor>>>& arg_mbs,
        const std::vector<std::shared_ptr<Tensor>>& target_mbs,
        const std::shared_ptr<Module>& loss_fn,
        std::vector<std::shared_ptr<Tensor>>* losses) override;
};

} // namespace infini_train::nn::pipeline