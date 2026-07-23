#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "infini_train/include/datatype.h"
#include "infini_train/include/nn/modules/transformer/moe/moe_utils.h"
#include "infini_train/include/nn/modules/transformer/transformer_config.h"

namespace infini_train {
class Tensor;
} // namespace infini_train

namespace infini_train::nn::parallel {
class ProcessGroup;
}

namespace infini_train::nn::moe {

class MoETokenDispatcher {
public:
    virtual ~MoETokenDispatcher() = default;

    virtual std::vector<std::shared_ptr<Tensor>> DispatchPreprocess(const std::shared_ptr<Tensor> &tokens,
                                                                    const std::shared_ptr<Tensor> &routing_map,
                                                                    const std::shared_ptr<Tensor> &probs)
        = 0;
    virtual std::vector<std::shared_ptr<Tensor>> TokenDispatch(const std::shared_ptr<Tensor> &hidden_states,
                                                               const std::shared_ptr<Tensor> &probs)
        = 0;
    virtual PermutationResult DispatchPostprocess(const std::shared_ptr<Tensor> &hidden_states,
                                                  const std::shared_ptr<Tensor> &probs)
        = 0;
    virtual std::shared_ptr<Tensor> CombinePreprocess(const std::shared_ptr<Tensor> &hidden_states) const = 0;
    virtual std::shared_ptr<Tensor> TokenCombine(const std::shared_ptr<Tensor> &hidden_states) const = 0;
    virtual std::shared_ptr<Tensor> CombinePostprocess(const std::shared_ptr<Tensor> &hidden_states) const = 0;

protected:
    explicit MoETokenDispatcher(const TransformerConfig &config);

    TransformerConfig config_;
};

class MoEAllGatherTokenDispatcher : public MoETokenDispatcher {
public:
    MoEAllGatherTokenDispatcher(int64_t num_local_experts, const std::vector<int64_t> &local_expert_ids,
                                const TransformerConfig &config, const parallel::ProcessGroup *process_group);

    std::vector<std::shared_ptr<Tensor>> DispatchPreprocess(const std::shared_ptr<Tensor> &tokens,
                                                            const std::shared_ptr<Tensor> &routing_map,
                                                            const std::shared_ptr<Tensor> &probs) override;
    std::vector<std::shared_ptr<Tensor>> TokenDispatch(const std::shared_ptr<Tensor> &hidden_states,
                                                       const std::shared_ptr<Tensor> &probs) override;
    PermutationResult DispatchPostprocess(const std::shared_ptr<Tensor> &hidden_states,
                                          const std::shared_ptr<Tensor> &probs) override;
    std::shared_ptr<Tensor> CombinePreprocess(const std::shared_ptr<Tensor> &hidden_states) const override;
    std::shared_ptr<Tensor> TokenCombine(const std::shared_ptr<Tensor> &hidden_states) const override;
    std::shared_ptr<Tensor> CombinePostprocess(const std::shared_ptr<Tensor> &hidden_states) const override;

private:
    // Global expert IDs owned by this rank, in local expert order.
    std::vector<int64_t> local_expert_ids_;
    const parallel::ProcessGroup *process_group_ = nullptr;
    PermutationMetadata permutation_metadata_;
    std::vector<int64_t> hidden_shape_;
    std::vector<int64_t> hidden_shape_before_permute_;
    std::shared_ptr<Tensor> routing_map_;
    std::shared_ptr<Tensor> empty_route_hidden_states_;
    std::shared_ptr<Tensor> empty_route_probs_;
    DataType local_probs_dtype_;
};

} // namespace infini_train::nn::moe
