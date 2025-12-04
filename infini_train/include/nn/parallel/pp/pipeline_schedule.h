#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/datatype.h"

namespace infini_train {
class Tensor;
namespace nn {
class Module;
}
} // namespace infini_train

namespace infini_train::nn::parallel {

class PipelineStage;

class PipelineSchedule {
public:
    PipelineSchedule(std::shared_ptr<PipelineStage> stage, int num_stages, int num_micro_batches, int stage_index)
        : stage_(std::move(stage)), num_micro_batches_(num_micro_batches), stage_index_(stage_index) {}

    virtual ~PipelineSchedule() = default;

    float Step(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target,
               const std::shared_ptr<nn::Module> &loss_fn, DataType dtype);

    virtual float StepMicroBatches(const std::vector<std::shared_ptr<Tensor>> &arg_mbs,
                                   const std::vector<std::shared_ptr<Tensor>> &target_mbs,
                                   const std::shared_ptr<nn::Module> &loss_fn, DataType dtype)
        = 0;

    std::vector<std::shared_ptr<Tensor>> ReceiveFromPrev();
    std::vector<std::shared_ptr<Tensor>> SendToNext(const std::vector<std::shared_ptr<Tensor>> &tensors);

protected:
    int num_micro_batches_ = -1;
    int stage_index_ = -1;
    std::shared_ptr<PipelineStage> stage_ = nullptr;
};

class ScheduleGPipe : public PipelineSchedule {
public:
    ScheduleGPipe(std::shared_ptr<PipelineStage> stage, int num_stages, int num_micro_batches, int stage_index)
        : PipelineSchedule(std::move(stage), num_stages, num_micro_batches, stage_index){};

    float StepMicroBatches(const std::vector<std::shared_ptr<Tensor>> &arg_mbs,
                           const std::vector<std::shared_ptr<Tensor>> &target_mbs,
                           const std::shared_ptr<nn::Module> &loss_fn, DataType dtype) override;
};

class Schedule1F1B : public PipelineSchedule {
public:
    Schedule1F1B(std::shared_ptr<PipelineStage> stage, int num_stages, int num_micro_batches, int stage_index)
        : PipelineSchedule(std::move(stage), num_stages, num_micro_batches, stage_index){};

    float StepMicroBatches(const std::vector<std::shared_ptr<Tensor>> &arg_mbs,
                           const std::vector<std::shared_ptr<Tensor>> &target_mbs,
                           const std::shared_ptr<nn::Module> &loss_fn, DataType dtype) override;
};

} // namespace infini_train::nn::parallel
