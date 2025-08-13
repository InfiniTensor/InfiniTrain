// pipeline_parallel.cpp
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/pp/pipeline_parallel.h"
#include "infini_train/include/nn/parallel/pp/pipeline_stage.h"
#include "infini_train/include/nn/parallel/pp/pipeline_schedule.h"
#include "infini_train/include/nn/sequential.h"

namespace infini_train::nn::pipeline {

PipelineParallel::PipelineParallel(
    const std::shared_ptr<Module>& model,
    int num_stages,
    const std::vector<Device*>& devices,
    int global_batch_size,
    int seq_len,
    int vocab_size,
    int micro_batch_size,
    DeviceType device_type)
    : full_model_(std::move(model)),
    num_stages_(num_stages),
    devices_(devices),
    seq_len_(seq_len),
    vocab_size_(vocab_size),
    micro_batch_size_(micro_batch_size),
    device_type_(device_type)
{
    CHECK(num_stages_ > 1) << "Use regular model for single stage.";
    CHECK(full_model_ != nullptr);

    // 分割模型
    auto stage_modules = SplitModel(full_model_);

    // 设置容量为num_stages_个
    schedules_.reserve(num_stages_);
    // loss_promises_.resize(num_stages_);

    for (int si = 0; si < num_stages_; ++si) {
        auto schedule = std::make_unique<PipelineSchedule>(
            stage_modules[si].get(),     // 该 stage 的子模型
            si,                          // stage index
            num_stages_,
            micro_batch_size_,
            seq_len_,
            vocab_size_,
            DeviceManager::Instance()->GetDevice(device_type_)
        );
        schedules_.push_back(std::move(schedule));
    }
}

//TODO(jym):实现切分逻辑
std::vector<std::shared_ptr<PipelineStage>> PipelineParallel::SplitModel() {
    std::vector<std::shared_ptr<PipelineStage>> stages;

    for (int r = 0; r < num_stages_; ++r) {

        auto stage = std::make_shared<PipelineStage>(, , ,);
        stages.push_back(stage);

    }
    return stages;
}

//TODO(jym):训练逻辑
float PipelineParallel::TrainStep(
    const std::vector<std::shared_ptr<Tensor>>& inputs,
    const std::vector<std::shared_ptr<Tensor>>& labels,
    const std::shared_ptr<Module>& loss_fn)
{
    // 用于收集最后 stage 的 loss
    // std::future<float> loss_future = loss_promises_[num_stages_ - 1].get_future();

    // 启动每个 stage 的线程
    for (int si = 0; si < num_stages_; ++si) {
        auto& schedule = schedules_[si];
        // auto& promise = loss_promises_[si];

        stage_threads_.emplace_back([si, &schedule, &inputs, &labels, loss_fn, this]() {
            try {
                // 只有第一个 stage 使用输入数据
                std::vector<std::shared_ptr<Tensor>> stage_inputs;
                if (si == 0) {
                    stage_inputs = inputs;
                }

                // 执行完整一次调度
                auto outputs = schedule->Step(stage_inputs, labels, loss_fn);

            }
        });
    }

    // 等待所有线程完成
    for (auto& t : stage_threads_) {
        if (t.joinable()) {
            t.join();
        }
    }
    stage_threads_.clear();

    // 获取 loss
    float lossf = 0.0f;
    return lossf;
}

std::vector<Parameter> PipelineParallel::Parameters() const {
    // 可返回第一个 stage 的参数，或所有 stage 的拼接
    if (!schedules_.empty()) {
        return schedules_[0]->GetModule()->Parameters();
    }
    return {};
}

void PipelineParallel::To(const runtime::Device* device) {
    for (auto& s : schedules_) {
        s->GetModule()->To(device);
    }
}

void PipelineParallel::To(DataType dtype) {
    for (auto& s : schedules_) {
        s->GetModule()->To(dtype);
    }
}

} // namespace infini_train::nn::pipeline