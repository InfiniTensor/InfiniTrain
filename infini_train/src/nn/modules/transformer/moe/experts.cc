#include "infini_train/include/nn/modules/transformer/moe/experts.h"

#include <memory>
#include <string>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/modules/transformer/mlp.h"
#include "infini_train/include/nn/modules/transformer/moe/moe_utils.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::moe {

SequentialMLP::SequentialMLP(const TransformerConfig &config) : CloneableModule(kType), config_(config) {
    const auto &moe_config = RequireMoEConfig(config_);
    CHECK(moe_config.expert_impl == MoEExpertImpl::kSequential);
    CHECK_EQ(moe_config.expert_parallel_size, 1)
        << "Current InfiniTrain MoE implementation supports expert_parallel_size=1 only";
    CHECK(moe_config.dispatcher_type == MoEDispatcherType::kLocal)
        << "Current InfiniTrain MoE implementation supports local dispatch only";

    num_local_experts_ = moe_config.num_experts;
    CHECK_GT(num_local_experts_, 0);

    for (int64_t expert_idx = 0; expert_idx < num_local_experts_; ++expert_idx) {
        modules_[std::string(kExpertNamePrefix) + std::to_string(expert_idx)] = std::make_shared<MLP>(config_);
    }
}

std::vector<std::shared_ptr<Tensor>> SequentialMLP::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 2);
    auto hidden_states = input_tensors[0];
    auto routing_probs = input_tensors[1];
    CHECK_EQ(routing_probs->Dims().back(), num_local_experts_);

    std::shared_ptr<Tensor> output = nullptr;
    const int64_t expert_dim = static_cast<int64_t>(routing_probs->Dims().size()) - 1;
    for (int64_t expert_idx = 0; expert_idx < num_local_experts_; ++expert_idx) {
        auto expert_name = std::string(kExpertNamePrefix) + std::to_string(expert_idx);
        auto expert_output = (*modules_.at(expert_name))({hidden_states})[0];
        auto expert_prob = routing_probs->Slice(expert_dim, expert_idx, expert_idx + 1);
        auto weighted_output = expert_output * expert_prob;
        output = output == nullptr ? weighted_output : output + weighted_output;
    }

    return {output};
}

} // namespace infini_train::nn::moe
