#include "infini_train/include/nn/parallel/pp/pipeline_stage.h"

#include "glog/logging.h"

#include <memory>

#include "infini_train/include/dispatcher.h"
#include "infini_train/src/nn/parallel/pp/pipeline_stage_function.h"

namespace infini_train::nn::pipeline {

PipelineStage::PipelineStage(
    const std::shared_ptr<Module>& module,
    int stage_index,
    int num_stages,
    std::shared_ptr<Optimizer> opt) :
    stage_index_(stage_index),
    num_stages_(num_stages),
    module_(std::move(module)),
    prev_rank_(stage_index > 0 ? stage_index - 1 : -1),
    next_rank_(stage_index < num_stages - 1 ? stage_index + 1 : -1),
    optimizer_(opt) {}

std::vector<std::shared_ptr<Tensor>> PipelineStage::ForwardOneChunk(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    //设置 requires_grad（不是第一个stage的输入需要梯度）
    std::vector<std::shared_ptr<Tensor>> current = inputs;

    forward_outputs_.clear();

    for (const auto& layer : layers_) {
        auto outputs = layer->Forward(current);

        // 缓存每层输出，用于反向传播（如果需要梯度）
        if (layer->RequiresGrad()) {
            forward_outputs_.push_back(outputs[0]);  // 假设单输出
        }

        current = outputs;
    }

    return current;
}

std::shared_ptr<Tensor> PipelineStage::BackwardOneChunk(const std::shared_ptr<Tensor>& output_grad) {
    std::shared_ptr<Tensor> grad = output_grad;

    // 从最后一层开始，反向
    for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
        const auto& layer = layers_[i];

        std::shared_ptr<Tensor> output = forward_outputs_[i];

        auto input_grads = layer->Backward(grad);  //TODO(jym): 每一层的反向

        grad = input_grads.empty() ? nullptr : input_grads[0];
    }

    return grad;
}

void PipelineStage::ScaleGrads(float scale_factor) {
    for (auto& param : Parameters()) {
        param->Mul_(scale_factor);
    }
}

} // namespace infini_train::nn::pipeline