#pragma once

#include "infini_train/include/core/transformer/transformer_config.h"

namespace infini_train::nn::llama3 {
inline nn::TransformerConfig LLaMA3Config() {
    return {.model_type = ModelType::kLLaMA3,
            .block_size = 8192,
            .vocab_size = 128256,
            .original_vocab_size = 128256,
            .n_layer = 16,
            .n_head = 32,
            .n_kv_head = 8,
            .n_embd = 2048,
            .attention_type = nn::AttentionType::kRoPE,
            .activation_type = nn::MLPType::kSwiGLU,
            .norm_type = nn::NormType::kRMSNorm,
            .use_bias = false,
            .tie_weights = false,
            .ffn_expansion_ratio = 4.0f,
            .ffn_dim_multiplier = 1.5f,
            .multiple_of = 256};
}
} // namespace infini_train::nn::llama3
