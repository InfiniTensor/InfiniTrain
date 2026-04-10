#pragma once

#include "infini_train/include/nn/modules/transformer/transformer_config.h"

namespace nn = infini_train::nn;
namespace gpt2 {
inline nn::TransformerConfig GPT2Config() {
    return {.block_size = 1024,
            .vocab_size = 50304,
            .original_vocab_size = 50257,
            .n_layer = 12,
            .n_head = 12,
            .n_kv_head = 12,
            .n_embd = 768,
            .attention_type = nn::AttentionType::kStandard,
            .activation_type = nn::MLPType::kGELU,
            .norm_type = nn::NormType::kLayerNorm,
            .add_bias_linear = true,
            .add_bias_lm_head = false,
            .tie_weights = true,
            .ffn_expansion_ratio = 4.0f,
            .ffn_dim_multiplier = std::nullopt,
            .multiple_of = 1};
}

} // namespace gpt2
