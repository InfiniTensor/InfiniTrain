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
    PipelineStage(int stage_index, int num_stages, const std::vector<std::vector<int64_t>> &recv_shape, int device_id,
                  std::vector<std::shared_ptr<Module>> &&chunks);

    std::vector<std::shared_ptr<Tensor>> ForwardOneChunk(const std::vector<std::shared_ptr<Tensor>> &inputs,
                                                         int local_chunk_idx = 0);

    bool IsFirstStage() const;
    bool IsLastStage() const;

    int stage_index() const;
    int prev_rank() const;
    int next_rank() const;
    int num_stages() const;

    const Device *device() const;
    const std::vector<std::vector<int64_t>> &recv_shape() const;
    const std::vector<std::shared_ptr<Module>> &chunks();
    std::vector<std::shared_ptr<Module>> *mutable_chunks();

private:
    int stage_index_ = -1;
    int num_stages_ = -1;
    int prev_rank_ = -1;
    int next_rank_ = -1;
    const Device *device_ = nullptr;
    std::vector<std::shared_ptr<Module>> chunks_;
    std::vector<std::vector<int64_t>> recv_shape_;
};

} // namespace infini_train::nn::parallel
