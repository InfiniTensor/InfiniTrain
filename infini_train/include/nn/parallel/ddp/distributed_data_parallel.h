#pragma once

#include <memory>

#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/ddp/param_and_grad_buffer.h"
#include "infini_train/include/nn/parallel/ddp/reducer.h"

namespace infini_train {
class Tensor;
class Device;
namespace nn::parallel {
class DistributedDataParallelConfig;
} // namespace nn::parallel
} // namespace infini_train

namespace infini_train::nn::parallel {

class DistributedDataParallel : public nn::Module {
public:
    DistributedDataParallel(std::shared_ptr<nn::Module> module, int thread_rank,
                            DistributedDataParallelConfig ddp_config);

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

    DistributedDataParallelConfig ddp_config() const { return ddp_config_; }

    const std::vector<std::shared_ptr<ParamAndGradBuffer>> &param_grad_buffers() const { return param_grad_buffers_; }

    const std::vector<std::shared_ptr<ParamAndGradBucketGroup>> &bucket_groups() const { return bucket_groups_; }

private:
    void BuildParamAndGradBuffers();
    void RegisterBackwardHooks();
    void OnGradReady(const std::shared_ptr<Tensor> &param);

private:
    std::shared_ptr<Reducer> reducer_ = nullptr;

    DistributedDataParallelConfig ddp_config_;
    const ProcessGroup *ddp_pg_ = nullptr;

    std::vector<std::shared_ptr<ParamAndGradBuffer>> param_grad_buffers_;
    std::vector<std::shared_ptr<ParamAndGradBucketGroup>> bucket_groups_;
    std::unordered_map<Tensor *, std::shared_ptr<ParamAndGradBucketGroup>> param_to_bucket_group_;
};

} // namespace infini_train::nn::parallel
