#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::pipeline {
struct ActivationShape {
    int batch_size;
    int seq_len;
    int hidden_size;
};

template<typename T>
std::string VectorToString(const std::vector<T>& vec) {
    if (vec.empty()) return "[]";
    std::string result = "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        result += std::to_string(vec[i]);
        if (i < vec.size() - 1) {
            result += ",";
        }
    }
    result += "]";
    return result;
}

void PrintTensorSummary(const std::shared_ptr<Tensor>& tensor, const std::string& tag);

class PipelineStage {
public:
    PipelineStage(std::vector<std::shared_ptr<Module>> &layers, int stage_index, int num_stages,
                  const ActivationShape &recvShape, std::shared_ptr<Optimizer> optim);

    std::vector<std::shared_ptr<Tensor>> ForwardOneChunk(const std::vector<std::shared_ptr<Tensor>> &inputs);

    bool IsFirstStage() const { return stage_index_ == 0; }
    bool IsLastStage() const { return stage_index_ == num_stages_ - 1; }

    int stage_index() const { return stage_index_; }
    int prev_rank() const { return prev_rank_; }
    int next_rank() const { return next_rank_; }
    int num_stages() const { return num_stages_; }
    const Device *device() const { return device_; }
    ActivationShape recv_shape() const { return recv_shape_; }
    std::shared_ptr<Optimizer> optimizer() { return optim_; }
    std::vector<std::shared_ptr<Module>> layers_;

private:
    int stage_index_;
    int num_stages_;
    int prev_rank_;
    int next_rank_;
    ActivationShape recv_shape_;
    const Device *device_ = nullptr;
    std::vector<const Device *> devices_;
    // std::vector<std::shared_ptr<Module>> layers_;
    std::vector<std::shared_ptr<Tensor>> forward_outputs_;
    std::shared_ptr<Optimizer> optim_;
};

} // namespace infini_train::nn::pipeline