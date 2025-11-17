#pragma once

#include <memory>
#include <vector>

namespace infini_train {
class Tensor;
class Device;
class Optimizer;
namespace nn {
class Module;
}
} // namespace infini_train

namespace infini_train::nn::parallel {

class PipelineStage {
public:
    PipelineStage(const std::shared_ptr<nn::Module> &model, int stage_index, int num_stages,
                  const std::vector<std::vector<int64_t>> &recv_shape, std::shared_ptr<Optimizer> optimizer);

    std::vector<std::shared_ptr<Tensor>> ForwardOneChunk(const std::vector<std::shared_ptr<Tensor>> &inputs);

    bool IsFirstStage() const { return stage_index_ == 0; }
    bool IsLastStage() const { return stage_index_ == num_stages_ - 1; }
    int stage_index() const { return stage_index_; }
    int prev_rank() const { return prev_rank_; }
    int next_rank() const { return next_rank_; }
    int num_stages() const { return num_stages_; }
    const Device *device() const { return device_; }
    const std::vector<std::vector<int64_t>> &recv_shape() const { return recv_shape_; }
    std::shared_ptr<Optimizer> optimizer() { return optimizer_; }

private:
    int stage_index_ = -1;
    int num_stages_ = -1;
    int prev_rank_ = -1;
    int next_rank_ = -1;
    const Device *device_ = nullptr;
    std::shared_ptr<nn::Module> model_ = nullptr;
    std::shared_ptr<Optimizer> optimizer_ = nullptr;
    std::vector<std::vector<int64_t>> recv_shape_;
};

} // namespace infini_train::nn::parallel
