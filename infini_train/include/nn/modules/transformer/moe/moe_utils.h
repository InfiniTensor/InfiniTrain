#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "infini_train/include/nn/modules/transformer/transformer_config.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::moe {

// Metadata produced by Megatron-style token permutation.  Tokens are grouped by expert in
// expert-id order: [expert0 tokens][expert1 tokens] ... [expertN tokens].
struct PermutationMetadata {
    // Original token row for each dispatched token after expert-grouped permutation.
    // Shape: [num_dispatched_tokens].  Megatron names this mapping sorted_indices.
    std::shared_ptr<Tensor> sorted_indices;

    // InfiniTrain Gather/ScatterAdd currently consumes a rank-2 index tensor.  This is
    // sorted_indices reshaped to [num_dispatched_tokens, 1], and repeated across hidden
    // dimension at the call sites when needed.
    std::shared_ptr<Tensor> expanded_sorted_indices;

    // Flattened indices into probs.view(-1), one per dispatched token.  Used to gather
    // the routing probability that corresponds to each (token, expert) assignment.
    // Shape: [num_dispatched_tokens].
    std::shared_ptr<Tensor> selected_probs_indices;

    // Number of dispatched tokens assigned to each expert. Kept on host because the
    // current local SequentialMLP uses it only for CPU-side slicing decisions.
    // Shape: [num_experts].
    std::vector<int64_t> tokens_per_expert;
};

struct PermutationResult {
    // Expert-grouped token tensor returned by Permute, matching Megatron's permuted_input.
    // Shape: [num_dispatched_tokens, hidden_size].
    std::shared_ptr<Tensor> permuted_input;

    // Routing probability aligned with each row in permuted_input.
    // Shape: [num_dispatched_tokens].
    std::shared_ptr<Tensor> permuted_probs;

    PermutationMetadata metadata;
};

// Select top-k experts from router logits, matching Megatron topk_routing_with_score_function.
// Args:
//   logits:             [num_tokens, num_experts] router logits.
//   topk:               Number of experts selected for each token.
//   use_pre_softmax:    If true, select top-k logits before score_function normalization.
//   scaling_factor:     Optional multiplier applied to selected routing weights.
//   score_function:     Router score function, currently softmax or sigmoid.
// Returns:
//   [0] routing_probs:  [num_tokens, num_experts] dense routing weights, zero outside top-k.
//   [1] routing_map:    [num_tokens, num_experts] bool token-to-expert assignment map.
std::vector<std::shared_ptr<Tensor>> TopkRoutingWithScoreFunction(const std::shared_ptr<Tensor> &logits, int64_t topk,
                                                                  bool use_pre_softmax,
                                                                  std::optional<float> scaling_factor,
                                                                  const MoEConfig::RouterScoreFunction &score_function);

// Fetch the MoE config from TransformerConfig and fail fast when the layer is misconfigured.
const MoEConfig &RequireMoEConfig(const TransformerConfig &config);

// Build Megatron-style permutation metadata from a dense bool routing map.
// Args:
//   routing_map: [num_tokens, num_experts] bool token-to-expert assignment map.
// Returns:
//   Metadata that groups tokens by expert, records host-side per-expert counts, and stores the
//   index tensors needed to gather permuted tokens/probs and scatter-add them back.
PermutationMetadata BuildPermutationMetadata(const std::shared_ptr<Tensor> &routing_map);

// Permute tokens and probs according to routing_map, matching Megatron moe_utils.permute.
// Args:
//   tokens:      [num_tokens, hidden_size] input token activations.
//   probs:       [num_tokens, num_experts] dense routing probabilities.
//   routing_map: [num_tokens, num_experts] bool token-to-expert assignment map.
PermutationResult Permute(const std::shared_ptr<Tensor> &tokens, const std::shared_ptr<Tensor> &probs,
                          const std::shared_ptr<Tensor> &routing_map);

// Restore permuted expert outputs to token order, matching Megatron moe_utils.unpermute.
// Each expert output row is weighted by its aligned permuted_probs value before scatter-add.
// Args:
//   permuted_tokens: [num_dispatched_tokens, hidden_size] expert outputs in permuted order.
//   permuted_probs:  [num_dispatched_tokens] routing probability for each expert output row.
//   restore_shape:   [num_tokens, hidden_size] output shape before permutation.
std::shared_ptr<Tensor> Unpermute(const std::shared_ptr<Tensor> &permuted_tokens,
                                  const std::shared_ptr<Tensor> &permuted_probs, const PermutationMetadata &metadata,
                                  const std::vector<int64_t> &restore_shape);

} // namespace infini_train::nn::moe
