#pragma once

// #include "p2p_comm.h"
#include <vector>
#include <memory>

#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::pipeline {

class PipelineStage {
public:
    /**
     *  module 当前 stage 表示的子网络
     *  stage_index 当前 stage 的编号（0 ~ num_stages-1）
     *  num_stages 总 stage 数
     *  device 当前设备
     */
    PipelineStage(const std::shared_ptr<Module>& module, int stage_index, int num_stages, std::shared_ptr<Optimizer> opt);

    std::vector<std::shared_ptr<Tensor>> ForwardOneChunk(const std::vector<std::shared_ptr<Tensor>>& inputs);

    std::shared_ptr<Tensor> BackwardOneChunk(const std::shared_ptr<Tensor>& output_grad);

    bool IsFirstStage() const { return stage_index_ == 0; }c

    bool IsLastStage() const { return stage_index_ == num_stages_ - 1; }

    // 缩放梯度（用于 micro-batch 累加）
    void ScaleGrads(float scale_factor);

protected:
    int stage_index_;
    int num_stages_;
    std::shared_ptr<nn::Module> module_;
    std::shared_ptr<Optimizer> optimizer_;
    int prev_rank_;
    int next_rank_;

    std::vector<std::shared_ptr<Module>> layers_;
    std::vector<std::shared_ptr<Tensor>> forward_outputs_;
};

} // namespace infini_train::nn::pipeline