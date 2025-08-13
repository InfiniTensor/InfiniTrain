// pipeline_parallel.h
#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/tensor.h"
#include "infini_train/include/device.h"

#include "infini_train/include/nn/parallel/pp/pipeline_schedule.h"
#include "infini_train/include/nn/parallel/pp/pipeline_stage.h"

namespace infini_train::nn::pipeline {

class PipelineParallel : public Module {
public:
    /**
     * @param full_model
     * @param num_stages stage 数
     * @param devices 设备列表（每个 stage 对应一个 device）
     * @param global_batch_size 全局batch大小
     * @param micro_batch_size micro_batch大小
     */
    PipelineParallel(
        const std::shared_ptr<Module>& model,
        int num_stages,
        const std::vector<Device*>& devices,
        int global_batch_size,
        int seq_len,
        int vocab_size,
        int micro_batch_size,
        int device_type
    );

    float TrainStep(
        const std::vector<std::shared_ptr<Tensor>>& input_tensors,
        const std::vector<std::shared_ptr<Tensor>>& targets,
        const std::shared_ptr<Module> &loss_fn) override;

    std::vector<std::shared_ptr<Tensor>> Parameters() const override;

    void To(const Device *device) override;

    void To(DataType dtype) override;

protected:
    std::vector<std::shared_ptr<PipelineStage>> stages_;
    std::vector<std::shared_ptr<PipelineSchedule>> schedules_;

private:
    std::shared_ptr<Module> full_model_;
    int num_stages_;
    int global_batch_size_;
    int micro_batch_size_;
    int seq_len_;
    int vocab_size_;
    DeviceType device_type_;
    std::vector<Device*> devices_;

    // 每个 stage 的线程和调度器
    std::vector<std::thread> stage_threads_;
    // std::vector<std::promise<float>> loss_promises_;  // TODO(jym):用于收集 loss

    std::vector<std::shared_ptr<Module>> SplitModel(std::shared_ptr<Module> model);
};

} // namespace infini_train::nn::pipeline