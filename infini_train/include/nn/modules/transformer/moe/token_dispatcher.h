#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "infini_train/include/nn/modules/transformer/moe/moe_utils.h"
#include "infini_train/include/nn/modules/transformer/transformer_config.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::moe {

class MoETokenDispatcher {
public:
    virtual ~MoETokenDispatcher() = default;

    const PermutationResult &Dispatch(const std::shared_ptr<Tensor> &tokens, const std::shared_ptr<Tensor> &routing_map,
                                      const std::shared_ptr<Tensor> &probs);
    std::shared_ptr<Tensor> Combine(const std::shared_ptr<Tensor> &hidden_states) const;

protected:
    explicit MoETokenDispatcher(const TransformerConfig &config);

    virtual std::vector<std::shared_ptr<Tensor>> DispatchPreprocess(const std::shared_ptr<Tensor> &tokens,
                                                                    const std::shared_ptr<Tensor> &routing_map,
                                                                    const std::shared_ptr<Tensor> &probs)
        = 0;
    virtual std::vector<std::shared_ptr<Tensor>> TokenDispatch(const std::shared_ptr<Tensor> &hidden_states,
                                                               const std::shared_ptr<Tensor> &probs) const
        = 0;
    virtual const PermutationResult &DispatchPostprocess(const std::shared_ptr<Tensor> &hidden_states,
                                                         const std::shared_ptr<Tensor> &probs)
        = 0;
    virtual std::shared_ptr<Tensor> CombinePreprocess(const std::shared_ptr<Tensor> &hidden_states) const = 0;
    virtual std::shared_ptr<Tensor> TokenCombine(const std::shared_ptr<Tensor> &hidden_states) const = 0;
    virtual std::shared_ptr<Tensor> CombinePostprocess(const std::shared_ptr<Tensor> &hidden_states) const = 0;

    TransformerConfig config_;
    PermutationResult dispatch_;
    std::vector<int64_t> hidden_dims_;
    std::shared_ptr<Tensor> routing_map_;
    std::shared_ptr<Tensor> local_map_;
    std::shared_ptr<Tensor> local_probs_;
    int64_t num_tokens_ = 0;
    int64_t hidden_size_ = 0;
};

class MoEAllGatherTokenDispatcher : public MoETokenDispatcher {
public:
    MoEAllGatherTokenDispatcher(int64_t num_local_experts, const TransformerConfig &config);

private:
    std::vector<std::shared_ptr<Tensor>> DispatchPreprocess(const std::shared_ptr<Tensor> &tokens,
                                                            const std::shared_ptr<Tensor> &routing_map,
                                                            const std::shared_ptr<Tensor> &probs) override;
    std::vector<std::shared_ptr<Tensor>> TokenDispatch(const std::shared_ptr<Tensor> &hidden_states,
                                                       const std::shared_ptr<Tensor> &probs) const override;
    const PermutationResult &DispatchPostprocess(const std::shared_ptr<Tensor> &hidden_states,
                                                 const std::shared_ptr<Tensor> &probs) override;
    std::shared_ptr<Tensor> CombinePreprocess(const std::shared_ptr<Tensor> &hidden_states) const override;
    std::shared_ptr<Tensor> TokenCombine(const std::shared_ptr<Tensor> &hidden_states) const override;
    std::shared_ptr<Tensor> CombinePostprocess(const std::shared_ptr<Tensor> &hidden_states) const override;

    int64_t num_local_experts_ = 0;
};

} // namespace infini_train::nn::moe
