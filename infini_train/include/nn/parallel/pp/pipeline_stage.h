#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::parallel {

class PipelineStage {
public:
    PipelineStage(const std::vector<std::shared_ptr<Module>> &layers, int stage_index, int num_stages,
                  const std::vector<std::vector<int64_t>> &recvShape, std::shared_ptr<Optimizer> optim);

    std::vector<std::shared_ptr<Tensor>> ForwardOneChunk(const std::vector<std::shared_ptr<Tensor>> &inputs);

    bool IsFirstStage() const { return stage_index_ == 0; }
    bool IsLastStage() const { return stage_index_ == num_stages_ - 1; }
    int stage_index() const { return stage_index_; }
    int prev_rank() const { return prev_rank_; }
    int next_rank() const { return next_rank_; }
    int num_stages() const { return num_stages_; }
    const Device *device() const { return device_; }
    std::vector<std::vector<int64_t>> recv_shape() const { return recv_shape_; }
    std::shared_ptr<Optimizer> optimizer() { return optim_; }

private:
    int stage_index_;
    int num_stages_;
    int prev_rank_;
    int next_rank_;
    const Device *device_ = nullptr;
    std::vector<std::vector<int64_t>> recv_shape_;
    std::vector<std::shared_ptr<Module>> layers_;
    std::shared_ptr<Optimizer> optim_;
};

} // namespace infini_train::nn::parallel
