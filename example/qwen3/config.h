#pragma once
#include <cmath>

#include "glog/logging.h"

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
            .position_embedding_type = nn::PositionEmbeddingType::kRoPE,
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
            .rotary_interleaved = false,
            .norm_eps = 1e-6f,
            .qk_layernorm = true};
}

inline void SanitizeQwen3Config(const nn::TransformerConfig &c) {
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
    const auto head_dim = c.n_embd / c.n_head;
    CHECK_GT(head_dim, 0) << "Qwen3 attention head dimension must be positive";
    CHECK_EQ(head_dim % 2, 0) << "Qwen3 RoPE requires an even attention head dimension";
    CHECK(c.position_embedding_type == nn::PositionEmbeddingType::kRoPE) << "Qwen3 requires RoPE position embedding";
    CHECK(!c.use_scaled_rope) << "Qwen3 scaled RoPE is not implemented";
    CHECK(!c.rotary_interleaved) << "Qwen3 requires half-split rotary embedding";
    CHECK_GT(c.rope_theta, 0.0f);
    CHECK(c.activation_type == nn::MLPType::kSwiGLU) << "Qwen3 requires SwiGLU activation";
    CHECK(c.ffn_type == nn::FFNType::kDense) << "Qwen3-8B requires a dense FFN";
    CHECK_GT(c.ffn_expansion_ratio, 0.0f);
    CHECK(c.norm_type == nn::NormType::kRMSNorm) << "Qwen3 requires RMSNorm";
    CHECK_GT(c.norm_eps, 0.0f);
    CHECK(!c.add_bias_linear) << "Qwen3 has no bias in linear layers";
    CHECK(!c.add_bias_lm_head) << "Qwen3 has no bias in lm_head";
    CHECK(!c.tie_weights) << "Qwen3 does not tie embedding and lm_head weights";
    CHECK_GT(c.multiple_of, 0);
    CHECK(c.qk_layernorm) << "Qwen3 apply laynorm to the query and key embeddings";
}
} // namespace qwen3
