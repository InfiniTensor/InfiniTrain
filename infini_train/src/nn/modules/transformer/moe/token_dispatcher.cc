#include "infini_train/include/nn/modules/transformer/moe/token_dispatcher.h"

#include <algorithm>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/autograd/function.h"
#include "infini_train/include/nn/parallel/parallel_functional.h"
#include "infini_train/include/nn/parallel/process_group.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::moe {

namespace {

// FIXME(dcj): Support zero-tensor later.
class EmptyExpertOutput : public autograd::Function {
public:
    static constexpr char kType[] = "EmptyExpertOutputFunction";

    explicit EmptyExpertOutput(DataType output_dtype) : autograd::Function(kType), output_dtype_(output_dtype) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &inputs) override {
        CHECK_EQ(inputs.size(), 2);
        hidden_shape_ = inputs[0]->Dims();
        probs_shape_ = inputs[1]->Dims();
        hidden_dtype_ = inputs[0]->Dtype();
        probs_dtype_ = inputs[1]->Dtype();
        device_ = inputs[0]->GetDevice();
        auto output = std::make_shared<Tensor>(hidden_shape_, output_dtype_, device_);
        output->Fill(0.0f);
        return {output};
    }

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grads) override {
        CHECK_EQ(grads.size(), 1);
        auto hidden_grad = std::make_shared<Tensor>(hidden_shape_, hidden_dtype_, device_);
        auto probs_grad = std::make_shared<Tensor>(probs_shape_, probs_dtype_, device_);
        hidden_grad->Fill(0.0f);
        probs_grad->Fill(0.0f);
        return {hidden_grad, probs_grad};
    }

private:
    DataType output_dtype_;
    DataType hidden_dtype_;
    DataType probs_dtype_;
    Device device_;
    std::vector<int64_t> hidden_shape_;
    std::vector<int64_t> probs_shape_;
};

} // namespace

MoETokenDispatcher::MoETokenDispatcher(const TransformerConfig &config) : config_(config) {}

MoEAllGatherTokenDispatcher::MoEAllGatherTokenDispatcher(int64_t num_local_experts,
                                                         const std::vector<int64_t> &local_expert_ids,
                                                         const TransformerConfig &config,
                                                         const parallel::ProcessGroup *process_group)
    : MoETokenDispatcher(config), local_expert_ids_(local_expert_ids), process_group_(process_group) {
    CHECK_GT(num_local_experts, 0);
    CHECK_EQ(local_expert_ids_.size(), num_local_experts);
    CHECK(std::adjacent_find(local_expert_ids_.begin(), local_expert_ids_.end(),
                             [](int64_t lhs, int64_t rhs) { return rhs != lhs + 1; })
          == local_expert_ids_.end());
}

// Inputs:
//   tokens:      [..., hidden_size]
//   routing_map: [num_tokens, num_global_experts]
//   probs:       [num_tokens, num_global_experts]
// Outputs:
//   tokens:      [num_tokens, hidden_size]
//   probs:       [num_tokens, num_global_experts]
std::vector<std::shared_ptr<Tensor>>
MoEAllGatherTokenDispatcher::DispatchPreprocess(const std::shared_ptr<Tensor> &tokens,
                                                const std::shared_ptr<Tensor> &routing_map,
                                                const std::shared_ptr<Tensor> &probs) {
    hidden_shape_ = tokens->Dims();
    routing_map_ = routing_map;
    return {tokens->View({routing_map->Dims()[0], hidden_shape_.back()}), probs};
}

// Inputs:
//   hidden_states: [num_local_tokens, hidden_size]
//   probs:         [num_local_tokens, num_global_experts]
//   routing_map_:  [num_local_tokens, num_global_experts]
// Outputs:
//   hidden_states: [group_size * num_local_tokens, hidden_size]
//   probs:         [group_size * num_local_tokens, num_global_experts]
std::vector<std::shared_ptr<Tensor>>
MoEAllGatherTokenDispatcher::TokenDispatch(const std::shared_ptr<Tensor> &hidden_states,
                                           const std::shared_ptr<Tensor> &probs) {
    if (process_group_ == nullptr) {
        return {hidden_states, probs};
    }

    // Gather routing_map without autograd. NCCL has no bool datatype, so keep it as uint8
    // until the local expert columns have been sliced.
    auto gathered_shape = routing_map_->Dims();
    gathered_shape[0] *= process_group_->GetGroupSize();
    auto routing_map_uint8 = std::make_shared<Tensor>(routing_map_->To(DataType::kUINT8));
    auto gathered_map_uint8 = std::make_shared<Tensor>(gathered_shape, DataType::kUINT8, routing_map_->GetDevice());
    parallel::function::AllGather(gathered_map_uint8, routing_map_uint8, process_group_);
    routing_map_ = gathered_map_uint8;

    auto global_probs = parallel::function::DifferentiableAllGather(probs, process_group_);
    auto global_hidden_states = parallel::function::DifferentiableAllGather(hidden_states, process_group_);
    return {global_hidden_states, global_probs};
}

// Inputs:
//   hidden_states: [group_tokens, hidden_size]
//   probs:         [group_tokens, num_global_experts]
//   routing_map_:  [group_tokens, num_global_experts]
// Outputs:
//   permutation_result: PermutationResult
PermutationResult MoEAllGatherTokenDispatcher::DispatchPostprocess(const std::shared_ptr<Tensor> &hidden_states,
                                                                   const std::shared_ptr<Tensor> &probs) {
    hidden_shape_before_permute_ = hidden_states->Dims();
    const int64_t local_expert_start = local_expert_ids_.front();
    const int64_t local_expert_end = local_expert_ids_.back() + 1;
    const int64_t num_global_experts = RequireMoEConfig(config_).num_experts;
    std::shared_ptr<Tensor> local_map;
    std::shared_ptr<Tensor> local_probs;
    if (local_expert_start == 0 && local_expert_end == num_global_experts) {
        local_map = routing_map_;
        local_probs = probs;
    } else {
        local_map = routing_map_->Slice(1, local_expert_start, local_expert_end);
        local_probs = probs->Slice(1, local_expert_start, local_expert_end);
    }
    if (local_map->Dtype() != DataType::kBOOL) {
        local_map = std::make_shared<Tensor>(local_map->To(DataType::kBOOL));
    }
    local_probs_dtype_ = local_probs->Dtype();
    auto dispatch = Permute(hidden_states, local_probs, local_map);
    permutation_metadata_ = dispatch.metadata;
    if (dispatch.metadata.sorted_indices->Dims()[0] == 0) {
        empty_route_hidden_states_ = hidden_states;
        empty_route_probs_ = local_probs;
    } else {
        empty_route_hidden_states_ = nullptr;
        empty_route_probs_ = nullptr;
    }
    routing_map_ = nullptr;
    return dispatch;
}

std::shared_ptr<Tensor>
MoEAllGatherTokenDispatcher::CombinePreprocess(const std::shared_ptr<Tensor> &hidden_states) const {
    if (permutation_metadata_.sorted_indices->Dims()[0] == 0) {
        return std::make_shared<EmptyExpertOutput>(hidden_states->Dtype())
            ->Apply({empty_route_hidden_states_, empty_route_probs_})[0];
    }
    return Unpermute(hidden_states, permutation_metadata_, hidden_shape_before_permute_);
}

std::shared_ptr<Tensor> MoEAllGatherTokenDispatcher::TokenCombine(const std::shared_ptr<Tensor> &hidden_states) const {
    if (process_group_ == nullptr) {
        return hidden_states;
    }
    return parallel::function::DifferentiableReduceScatter(hidden_states, process_group_, local_probs_dtype_);
}

std::shared_ptr<Tensor>
MoEAllGatherTokenDispatcher::CombinePostprocess(const std::shared_ptr<Tensor> &hidden_states) const {
    return hidden_states->View(hidden_shape_);
}

} // namespace infini_train::nn::moe
