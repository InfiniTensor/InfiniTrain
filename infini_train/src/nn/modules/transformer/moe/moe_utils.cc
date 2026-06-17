#include "infini_train/include/nn/modules/transformer/moe/moe_utils.h"

#include <algorithm>

#include "glog/logging.h"

#include "infini_train/include/autograd/scatter.h"
#include "infini_train/include/autograd/scatter_add.h"
#include "infini_train/include/autograd/topk.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/nn/functional.h"

namespace infini_train::nn::moe {

std::vector<std::shared_ptr<Tensor>>
TopkRoutingWithScoreFunction(const std::shared_ptr<Tensor> &logits, int64_t topk, bool use_pre_softmax,
                             std::optional<float> scaling_factor,
                             const MoEConfig::RouterScoreFunction &score_function) {

    // Megatron TopKRouter returns dense tensors:
    //   routing_probs: [num_tokens, num_experts]
    //   routing_map:   [num_tokens, num_experts], bool
    std::shared_ptr<Tensor> top_probs;
    std::shared_ptr<Tensor> top_indices;

    if (score_function == MoEConfig::RouterScoreFunction::kSoftmax) {
        if (use_pre_softmax) {
            auto scores = function::Softmax(logits, -1);
            auto topk_function = std::make_shared<autograd::TopK>(topk);
            top_probs = topk_function->Apply({scores})[0];
            top_indices = topk_function->TopIndices();
        } else {
            auto topk_function = std::make_shared<autograd::TopK>(topk);
            auto top_scores = topk_function->Apply({logits})[0];
            top_indices = topk_function->TopIndices();
            top_probs = function::Softmax(top_scores, -1);
        }
    } else if (score_function == MoEConfig::RouterScoreFunction::kSigmoid) {
        auto sigmoid_scores = function::Sigmoid(logits);
        auto topk_function = std::make_shared<autograd::TopK>(topk);
        top_probs = topk_function->Apply({sigmoid_scores})[0];
        top_indices = topk_function->TopIndices();
        if (topk > 1) {
            top_probs = top_probs / (top_probs->Sum(-1, true) + 1e-20f);
        }
    } else {
        LOG(FATAL) << "Unsupported MoE router score function";
    }

    if (scaling_factor.has_value()) {
        top_probs = top_probs * scaling_factor.value();
    }

    auto routing_probs = std::make_shared<autograd::Scatter>(logits->Dims())->Apply({top_probs, top_indices})[0];
    auto routing_map_values = std::make_shared<Tensor>(top_indices->Equals(top_indices)->To(DataType::kBOOL));
    auto routing_map = Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
        {logits->GetDevice().type(), "ScatterForward"}, routing_map_values, top_indices, logits->Dims());
    return {routing_probs, routing_map};
}

const MoEConfig &RequireMoEConfig(const TransformerConfig &config) {
    CHECK(config.moe_config.has_value()) << "MoE layer requires TransformerConfig::moe_config";
    return config.moe_config.value();
}

PermutationMetadata BuildPermutationMetadata(const std::shared_ptr<Tensor> &routing_map) {
    CHECK(routing_map->Dtype() == DataType::kBOOL);
    CHECK_EQ(routing_map->Dims().size(), 2);

    const int64_t num_tokens = routing_map->Dims()[0];
    const int64_t num_experts = routing_map->Dims()[1];
    CHECK_GT(num_tokens, 0);
    CHECK_GT(num_experts, 0);

    Tensor routing_map_cpu_storage = routing_map->To(Device());
    auto routing_map_cpu = std::make_shared<Tensor>(routing_map_cpu_storage);
    const auto *routing_map_ptr = static_cast<const bool *>(routing_map_cpu->DataPtr());

    std::vector<int64_t> sorted_indices_host;
    std::vector<int64_t> selected_probs_indices_host;
    std::vector<int64_t> tokens_per_expert;
    sorted_indices_host.reserve(routing_map->NumElements());
    selected_probs_indices_host.reserve(routing_map->NumElements());
    tokens_per_expert.reserve(num_experts);

    for (int64_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
        int64_t tokens_for_expert = 0;
        for (int64_t token_idx = 0; token_idx < num_tokens; ++token_idx) {
            if (routing_map_ptr[token_idx * num_experts + expert_idx]) {
                sorted_indices_host.push_back(token_idx);
                selected_probs_indices_host.push_back(token_idx * num_experts + expert_idx);
                ++tokens_for_expert;
            }
        }
        tokens_per_expert.push_back(tokens_for_expert);
    }

    const int64_t num_dispatched_tokens = static_cast<int64_t>(sorted_indices_host.size());
    auto sorted_indices_cpu
        = std::make_shared<Tensor>(std::vector<int64_t>{num_dispatched_tokens}, DataType::kINT64, Device());
    auto selected_probs_indices_cpu
        = std::make_shared<Tensor>(std::vector<int64_t>{num_dispatched_tokens}, DataType::kINT64, Device());
    auto expanded_sorted_indices_cpu
        = std::make_shared<Tensor>(std::vector<int64_t>{num_dispatched_tokens, 1}, DataType::kINT64, Device());
    auto *sorted_indices_ptr = static_cast<int64_t *>(sorted_indices_cpu->DataPtr());
    auto *selected_probs_indices_ptr = static_cast<int64_t *>(selected_probs_indices_cpu->DataPtr());
    auto *expanded_sorted_indices_ptr = static_cast<int64_t *>(expanded_sorted_indices_cpu->DataPtr());
    for (int64_t idx = 0; idx < num_dispatched_tokens; ++idx) {
        sorted_indices_ptr[idx] = sorted_indices_host[idx];
        selected_probs_indices_ptr[idx] = selected_probs_indices_host[idx];
        expanded_sorted_indices_ptr[idx] = sorted_indices_host[idx];
    }
    auto to_device = [&](const std::shared_ptr<Tensor> &cpu_tensor) -> std::shared_ptr<Tensor> {
        if (routing_map->GetDevice().type() == Device::DeviceType::kCPU) {
            return cpu_tensor;
        }
        return std::make_shared<Tensor>(cpu_tensor->To(routing_map->GetDevice()));
    };

    return {to_device(sorted_indices_cpu), to_device(expanded_sorted_indices_cpu),
            to_device(selected_probs_indices_cpu), tokens_per_expert};
}

PermutationResult Permute(const std::shared_ptr<Tensor> &tokens, const std::shared_ptr<Tensor> &probs,
                          const std::shared_ptr<Tensor> &routing_map) {
    CHECK_EQ(tokens->Dims().size(), 2);
    CHECK(probs->Dims() == routing_map->Dims());
    CHECK(routing_map->Dtype() == DataType::kBOOL);

    const int64_t hidden_size = tokens->Dims()[1];
    auto metadata = BuildPermutationMetadata(routing_map);
    const int64_t num_dispatched_tokens = metadata.sorted_indices->Dims()[0];

    std::shared_ptr<Tensor> permuted_input;
    std::shared_ptr<Tensor> permuted_probs;
    if (num_dispatched_tokens == 0) {
        permuted_input
            = std::make_shared<Tensor>(std::vector<int64_t>{0, hidden_size}, tokens->Dtype(), tokens->GetDevice());
        permuted_probs = std::make_shared<Tensor>(std::vector<int64_t>{0}, probs->Dtype(), probs->GetDevice());
    } else {
        auto expanded_sorted_indices = metadata.expanded_sorted_indices;
        if (hidden_size != 1) {
            expanded_sorted_indices = metadata.expanded_sorted_indices->RepeatInterleave(hidden_size, 1);
        }
        permuted_input = tokens->Gather(0, expanded_sorted_indices);
        permuted_probs
            = probs->View({static_cast<int64_t>(probs->NumElements())})->Gather(0, metadata.selected_probs_indices);
    }

    return {permuted_input, permuted_probs, metadata};
}

std::shared_ptr<Tensor> Unpermute(const std::shared_ptr<Tensor> &permuted_tokens,
                                  const std::shared_ptr<Tensor> &permuted_probs, const PermutationMetadata &metadata,
                                  const std::vector<int64_t> &restore_shape) {
    CHECK_EQ(permuted_tokens->Dims().size(), 2);
    CHECK_EQ(permuted_probs->Dims().size(), 1);
    CHECK_EQ(permuted_tokens->Dims()[0], permuted_probs->Dims()[0]);
    CHECK_EQ(restore_shape.size(), 2);

    auto weighted = permuted_tokens * permuted_probs->View({permuted_probs->Dims()[0], 1});
    auto scatter_indices = metadata.expanded_sorted_indices;
    const int64_t hidden_size = restore_shape[1];
    if (hidden_size != 1) {
        scatter_indices = metadata.expanded_sorted_indices->RepeatInterleave(hidden_size, 1);
    }
    return std::make_shared<autograd::ScatterAdd>(0, restore_shape)->Apply({weighted, scatter_indices})[0];
}

} // namespace infini_train::nn::moe
