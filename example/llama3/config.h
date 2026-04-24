#pragma once

#include "glog/logging.h"

#include "infini_train/include/nn/modules/transformer/transformer_config.h"

namespace infini_train {
namespace llama3 {
inline nn::TransformerConfig LLaMA3Config() {
    return {.block_size = 8192,
            .vocab_size = 128256,
            .original_vocab_size = 128256,
            .n_layer = 16,
            .n_head = 32,
            .n_kv_head = 8,
            .n_embd = 2048,
            .attention_type = nn::AttentionType::kRoPE,
            .activation_type = nn::MLPType::kSwiGLU,
            .norm_type = nn::NormType::kRMSNorm,
            .add_bias_linear = false,
            .add_bias_lm_head = false,
            .tie_weights = false,
            .ffn_expansion_ratio = 4.0f,
            .ffn_dim_multiplier = 1.5f,
            .multiple_of = 256};
}

inline void SanitizeLLaMA3Config(const nn::TransformerConfig &c) {
    CHECK_GT(c.block_size, 0);
    CHECK_GT(c.vocab_size, 0);
    CHECK_GE(c.vocab_size, c.original_vocab_size);
    CHECK_GT(c.n_layer, 0);
    CHECK_GT(c.n_head, 0);
    CHECK_GT(c.n_kv_head, 0);
    CHECK_LE(c.n_kv_head, c.n_head);
    CHECK_EQ(c.n_head % c.n_kv_head, 0) << "n_head must be divisible by n_kv_head for GQA";
    CHECK_GT(c.n_embd, 0);
    CHECK_EQ(c.n_embd % c.n_head, 0) << "n_embd must be divisible by n_head";
    CHECK(c.attention_type == nn::AttentionType::kRoPE) << "LLaMA-3 requires RoPE attention";
    CHECK(c.activation_type == nn::MLPType::kSwiGLU) << "LLaMA-3 requires SwiGLU activation";
    CHECK(c.norm_type == nn::NormType::kRMSNorm) << "LLaMA-3 requires RMSNorm";
    CHECK(!c.add_bias_linear) << "LLaMA-3 has no bias in linear layers";
    CHECK(!c.tie_weights) << "LLaMA-3 does not tie embedding and lm_head weights";
    CHECK(c.ffn_dim_multiplier.has_value()) << "LLaMA-3 requires ffn_dim_multiplier";
    CHECK_GT(c.multiple_of, 0);
}
} // namespace llama3
} // namespace infini_train
