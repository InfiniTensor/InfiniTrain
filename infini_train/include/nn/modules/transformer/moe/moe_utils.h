#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "infini_train/include/nn/modules/transformer/transformer_config.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::moe {

struct PermutationMetadata {
    std::shared_ptr<Tensor> sorted_indices;
    std::shared_ptr<Tensor> gather_indices;
    std::shared_ptr<Tensor> route_indices;
    std::shared_ptr<Tensor> tokens_per_expert;
    std::vector<int64_t> tokens_per_expert_host;
};

struct PermutationResult {
    std::shared_ptr<Tensor> permuted_hidden_states;
    std::shared_ptr<Tensor> permuted_probs;
    PermutationMetadata metadata;
};

std::vector<std::shared_ptr<Tensor>> TopkRoutingWithScoreFunction(const std::shared_ptr<Tensor> &logits, int64_t topk,
                                                                  bool use_pre_softmax,
                                                                  std::optional<float> scaling_factor,
                                                                  const MoEConfig::RouterScoreFunction &score_function);

const MoEConfig &RequireMoEConfig(const TransformerConfig &config);
PermutationMetadata BuildPermutationMetadata(const std::shared_ptr<Tensor> &routing_map);
PermutationResult Permute(const std::shared_ptr<Tensor> &hidden_states_2d,
                          const std::shared_ptr<Tensor> &routing_probs_2d,
                          const std::shared_ptr<Tensor> &routing_map_2d);
std::shared_ptr<Tensor> Unpermute(const std::shared_ptr<Tensor> &permuted_hidden_states,
                                  const std::shared_ptr<Tensor> &permuted_probs, const PermutationMetadata &metadata,
                                  const std::vector<int64_t> &restore_shape);

} // namespace infini_train::nn::moe
