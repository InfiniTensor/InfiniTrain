#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "infini_train/include/optimizer.h"

namespace infini_train::nn {
class Module;
namespace parallel {
class ParamAndGradBuffer;
class ParamAndGradBucketGroup;
} // namespace parallel
} // namespace infini_train::nn

namespace infini_train::nn::parallel {

class DistributedOptimizer final : public infini_train::Optimizer {
public:
    DistributedOptimizer(OptimizerCreator base_optimizer_creator,
                         const std::vector<std::shared_ptr<Tensor>> &full_params,
                         const std::vector<std::shared_ptr<Module>> &model_chunks, size_t ddp_world_size,
                         size_t ddp_rank);

    void Step() override;

    void ZeroGrad(bool set_to_none = true) override;

    void StartGradSync();
    void FinishGradSync();

    void StartParamSync(bool force_sync = false);
    void FinishParamSync(bool skip_next_bucket_dispatch = false);

private:
    void BuildShardParamsAndBindGrads();

private:
    // Inherit from DDP model
    std::vector<std::shared_ptr<ParamAndGradBuffer>> param_grad_buffers_;
    std::vector<std::shared_ptr<ParamAndGradBucketGroup>> bucket_groups_;

    // DP info
    size_t ddp_world_size_;
    size_t ddp_rank_;

    // shard params
    std::vector<std::shared_ptr<Tensor>> shard_params_;

    // Base optimizer (SGD, Adam and etc.)
    std::shared_ptr<Optimizer> base_optimizer_;
};

} // namespace infini_train::nn::parallel
