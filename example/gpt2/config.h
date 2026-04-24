#pragma once

#include "glog/logging.h"

#include "infini_train/include/nn/modules/transformer/transformer_config.h"

namespace infini_train {
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

inline void SanitizeGPT2Config(const nn::TransformerConfig &c) {
    CHECK_GT(c.block_size, 0);
    CHECK_GT(c.vocab_size, 0);
    CHECK_GE(c.vocab_size, c.original_vocab_size);
    CHECK_GT(c.n_layer, 0);
    CHECK_GT(c.n_head, 0);
    CHECK_GT(c.n_embd, 0);
    CHECK_EQ(c.n_embd % c.n_head, 0) << "n_embd must be divisible by n_head";
    CHECK_EQ(c.n_kv_head, c.n_head) << "GPT-2 does not use GQA; n_kv_head must equal n_head";
    CHECK(c.attention_type == nn::AttentionType::kStandard) << "GPT-2 requires standard attention";
    CHECK(c.activation_type == nn::MLPType::kGELU) << "GPT-2 requires GELU activation";
    CHECK(c.norm_type == nn::NormType::kLayerNorm) << "GPT-2 requires LayerNorm";
}

} // namespace gpt2
} // namespace infini_train
