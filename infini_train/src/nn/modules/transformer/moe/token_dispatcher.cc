#include "infini_train/include/nn/modules/transformer/moe/token_dispatcher.h"

#include <memory>
#include <vector>

#include "glog/logging.h"

namespace infini_train::nn::moe {

MoETokenDispatcher::MoETokenDispatcher(const TransformerConfig &config) : config_(config) {}

const PermutationResult &MoETokenDispatcher::Dispatch(const std::shared_ptr<Tensor> &tokens,
                                                      const std::shared_ptr<Tensor> &routing_map,
                                                      const std::shared_ptr<Tensor> &probs) {
    auto preprocessed = DispatchPreprocess(tokens, routing_map, probs);
    auto dispatched = TokenDispatch(preprocessed[0], preprocessed[1]);
    return DispatchPostprocess(dispatched[0], dispatched[1]);
}

std::shared_ptr<Tensor> MoETokenDispatcher::Combine(const std::shared_ptr<Tensor> &hidden_states) const {
    auto preprocessed = CombinePreprocess(hidden_states);
    auto combined = TokenCombine(preprocessed);
    return CombinePostprocess(combined);
}

MoEAllGatherTokenDispatcher::MoEAllGatherTokenDispatcher(int64_t num_local_experts, const TransformerConfig &config)
    : MoETokenDispatcher(config), num_local_experts_(num_local_experts) {
    CHECK_GT(num_local_experts_, 0);
}

std::vector<std::shared_ptr<Tensor>>
MoEAllGatherTokenDispatcher::DispatchPreprocess(const std::shared_ptr<Tensor> &tokens,
                                                const std::shared_ptr<Tensor> &routing_map,
                                                const std::shared_ptr<Tensor> &probs) {
    CHECK(probs->Dims() == routing_map->Dims());
    CHECK(routing_map->Dtype() == DataType::kBOOL);
    CHECK_GE(tokens->Dims().size(), 2);

    hidden_dims_ = tokens->Dims();
    hidden_size_ = hidden_dims_.back();
    CHECK_GT(hidden_size_, 0);
    num_tokens_ = tokens->NumElements() / hidden_size_;
    CHECK_EQ(probs->Dims().back(), num_local_experts_);
    CHECK_EQ(probs->NumElements(), static_cast<size_t>(num_tokens_ * num_local_experts_));

    routing_map_ = routing_map->View({num_tokens_, num_local_experts_});
    auto hidden_states_2d = tokens->View({num_tokens_, hidden_size_});
    auto probs_2d = probs->View({num_tokens_, num_local_experts_});
    return {hidden_states_2d, probs_2d};
}

std::vector<std::shared_ptr<Tensor>>
MoEAllGatherTokenDispatcher::TokenDispatch(const std::shared_ptr<Tensor> &hidden_states,
                                           const std::shared_ptr<Tensor> &probs) const {
    // AllGather dispatcher will gather tokens across TP*EP ranks here. For the current single-rank
    // path (tp_size=1, ep_size=1), no communication is required.
    return {hidden_states, probs};
}

const PermutationResult &MoEAllGatherTokenDispatcher::DispatchPostprocess(const std::shared_ptr<Tensor> &hidden_states,
                                                                          const std::shared_ptr<Tensor> &probs) {
    CHECK(routing_map_ != nullptr);
    CHECK_EQ(hidden_states->Dims().size(), 2);
    CHECK_EQ(probs->Dims().size(), 2);
    CHECK_EQ(hidden_states->Dims()[0], probs->Dims()[0]);
    CHECK_EQ(probs->Dims()[1], num_local_experts_);

    // With ep_size=1 all experts are local, so the local expert map/probs are the gathered map/probs.
    // Future EP support should slice [local_expert_start, local_expert_end) after AllGather.
    local_map_ = routing_map_;
    local_probs_ = probs;
    dispatch_ = Permute(hidden_states, local_probs_, local_map_);
    routing_map_ = nullptr;
    return dispatch_;
}

std::shared_ptr<Tensor>
MoEAllGatherTokenDispatcher::CombinePreprocess(const std::shared_ptr<Tensor> &hidden_states) const {
    CHECK(local_map_ != nullptr);
    CHECK(local_probs_ != nullptr);
    return Unpermute(hidden_states, dispatch_.permuted_probs, dispatch_.metadata,
                     std::vector<int64_t>{num_tokens_, hidden_size_});
}

std::shared_ptr<Tensor> MoEAllGatherTokenDispatcher::TokenCombine(const std::shared_ptr<Tensor> &hidden_states) const {
    // AllGather dispatcher will reduce-scatter combined token outputs here. For ep_size=1 this is a no-op.
    return hidden_states;
}

std::shared_ptr<Tensor>
MoEAllGatherTokenDispatcher::CombinePostprocess(const std::shared_ptr<Tensor> &hidden_states) const {
    return hidden_states->View(hidden_dims_);
}

} // namespace infini_train::nn::moe
