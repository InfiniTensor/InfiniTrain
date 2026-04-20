#pragma once

#include "infini_train/include/nn/modules/transformer/transformer_config.h"

namespace nn = infini_train::nn;
namespace qwen3 {
inline nn::TransformerConfig Qwen3Config() {
    return {.block_size = 40960,
            .vocab_size = 151936,
            .original_vocab_size = 151936,
            .n_layer = 36,
            .n_head = 32,
            .n_kv_head = 8,
            .n_embd = 4096,
            .attention_type = nn::AttentionType::kRoPE,
            .activation_type = nn::MLPType::kSwiGLU,
            .norm_type = nn::NormType::kRMSNorm,
            .add_bias_linear = false,
            .add_bias_lm_head = false,
            .tie_weights = false,
            .ffn_expansion_ratio = 4.5f, // 4096*4.5*2/3 = 12288
            .ffn_dim_multiplier = std::nullopt,
            .multiple_of = 1,
            .rope_theta = 1000000.0f,
            .use_scaled_rope = false,
            .norm_eps = 1e-6f};
}
} // namespace qwen3
