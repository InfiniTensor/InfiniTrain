#pragma once
#include <memory>
#include <vector>

#include "infini_train/include/nn/parallel/pp/pipeline_stage.h"

namespace infini_train::nn::pipeline {

class PipelineSchedule {
public:
    PipelineSchedule(std::shared_ptr<PipelineStage> stage, int num_stages, int num_microbatches, int stage_index)
        : stage_(std::move(stage)), num_stages_(num_stages), num_microbatches_(num_microbatches),
          stage_index_(stage_index) {}

    virtual ~PipelineSchedule() = default;

    float Step(std::shared_ptr<Tensor>input, std::shared_ptr<Tensor> target,
               const std::shared_ptr<Module> &loss_fn);

    virtual float StepMicrobatches(const std::vector<std::shared_ptr<Tensor>> &arg_mbs,
                                   const std::vector<std::shared_ptr<Tensor>> &target_mbs,
                                   const std::shared_ptr<Module> &loss_fn)
        = 0;

    void OptimizerStep();

    int NumMicrobatches() const { return num_microbatches_; }

    std::shared_ptr<PipelineStage> stage() const { return stage_; }

protected:
    std::shared_ptr<PipelineStage> stage_;
    int stage_index_;
    int num_stages_;

private:
    std::vector<std::shared_ptr<Tensor>> SplitTensor(std::shared_ptr<Tensor> full_inputs);

    int num_microbatches_;
};

class Schedule1F1B : public PipelineSchedule {
public:
    Schedule1F1B(std::shared_ptr<PipelineStage> stage, int num_stages, int num_microbatches, int stage_index)
        : PipelineSchedule(std::move(stage), num_stages, num_microbatches, stage_index){};

    float StepMicrobatches(const std::vector<std::shared_ptr<Tensor>> &arg_mbs,
                           const std::vector<std::shared_ptr<Tensor>> &target_mbs,
                           const std::shared_ptr<Module> &loss_fn) override;
};

class ScheduleGPipe : public PipelineSchedule {
public:
    ScheduleGPipe(std::shared_ptr<PipelineStage> stage, int num_stages, int num_microbatches, int stage_index)
        : PipelineSchedule(std::move(stage), num_stages, num_microbatches, stage_index){};

    float StepMicrobatches(const std::vector<std::shared_ptr<Tensor>> &arg_mbs,
                           const std::vector<std::shared_ptr<Tensor>> &target_mbs,
                           const std::shared_ptr<Module> &loss_fn) override;
};

} // namespace infini_train::nn::pipeline