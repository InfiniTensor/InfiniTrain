#include "infini_train/include/nn/modules/transformer/moe/experts.h"

#include <memory>
#include <string>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/modules/transformer/mlp.h"
#include "infini_train/include/nn/modules/transformer/moe/moe_utils.h"
#include "infini_train/include/nn/modules/transformer/moe/token_dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::moe {

SequentialMLP::SequentialMLP(const TransformerConfig &config) : CloneableModule(kType), config_(config) {
    const auto &moe_config = RequireMoEConfig(config_);
    CHECK(moe_config.expert_impl == MoEConfig::ExpertImpl::kSequential);
    CHECK_EQ(moe_config.expert_parallel_size, 1)
        << "Current InfiniTrain MoE implementation supports expert_parallel_size=1 only";
    CHECK(moe_config.token_dispatcher_type == MoEConfig::TokenDispatcherType::kAllGather)
        << "Current InfiniTrain MoE implementation supports AllGather dispatcher only";

    num_local_experts_ = moe_config.num_experts;
    CHECK_GT(num_local_experts_, 0);

    for (int64_t expert_idx = 0; expert_idx < num_local_experts_; ++expert_idx) {
        modules_[std::string(kExpertNamePrefix) + std::to_string(expert_idx)] = std::make_shared<MLP>(config_);
    }
}

std::vector<std::shared_ptr<Tensor>> SequentialMLP::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 3);
    auto hidden_states = input_tensors[0];
    auto routing_probs = input_tensors[1];
    auto routing_map = input_tensors[2];
    std::unique_ptr<MoETokenDispatcher> dispatcher
        = std::make_unique<MoEAllGatherTokenDispatcher>(num_local_experts_, config_);
    const auto &dispatch = dispatcher->Dispatch(hidden_states, routing_map, routing_probs);

    std::vector<std::shared_ptr<Tensor>> expert_outputs;
    int64_t start = 0;
    for (int64_t expert_idx = 0; expert_idx < num_local_experts_; ++expert_idx) {
        const int64_t num_tokens_for_expert = dispatch.metadata.tokens_per_expert_host[expert_idx];
        const int64_t end = start + num_tokens_for_expert;
        if (num_tokens_for_expert == 0) {
            start = end;
            continue;
        }

        auto expert_input = dispatch.permuted_hidden_states->Slice(0, start, end);
        auto expert_name = std::string(kExpertNamePrefix) + std::to_string(expert_idx);
        expert_outputs.push_back((*modules_.at(expert_name))({expert_input})[0]);
        start = end;
    }
    CHECK_EQ(start, dispatch.permuted_hidden_states->Dims()[0]);
    CHECK(!expert_outputs.empty()) << "No tokens were dispatched to any local expert";

    auto permuted_expert_output
        = expert_outputs.size() == 1 ? expert_outputs[0] : nn::function::Concat(expert_outputs, 0);
    return {dispatcher->Combine(permuted_expert_output)};
}

} // namespace infini_train::nn::moe
