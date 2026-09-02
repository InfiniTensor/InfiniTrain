#pragma once

#include "glog/logging.h"

#include "infini_train/include/nn/modules/transformer/transformer_config.h"

namespace nn = infini_train::nn;
namespace fm9gv {

inline nn::TransformerConfig FM9GVTextConfig() {
    return {.block_size = 32768,
            .vocab_size = 73448,
            .original_vocab_size = 73448,
            .n_layer = 62,
            .n_head = 40,
            .n_kv_head = 40,
            .n_embd = 2560,
            .position_embedding_type = nn::PositionEmbeddingType::kRoPE,
            .activation_type = nn::MLPType::kSwiGLU,
            .ffn_type = nn::FFNType::kDense,
            .norm_type = nn::NormType::kRMSNorm,
            .attention_variant = nn::AttentionVariant::kFM9G,
            .add_bias_linear = false,
            .add_bias_lm_head = false,
            .tie_weights = false,
            .ffn_expansion_ratio = 4.0f,
            .ffn_dim_multiplier = std::nullopt,
            .multiple_of = 1,
            .ffn_hidden_size = 6400,
            .moe_config = std::nullopt,
            .rope_theta = 10000.0f,
            .use_scaled_rope = false,
            .q_lora_rank = 768,
            .kv_lora_rank = 256,
            .qk_nope_head_dim = 64,
            .qk_rope_head_dim = 32,
            .v_head_dim = 64,
            .scale_emb = 12.0f,
            .scale_depth = 1.4f,
            .dim_model_base = 256,
            .rope_short_factors = {1.0591234f, 1.1241891f, 1.2596936f, 1.5380380f, 2.0939825f, 3.1446936f,
                                   4.9379526f, 7.5245420f, 10.4754580f, 13.0620474f, 14.8553066f, 15.9060173f,
                                   16.4619617f, 16.7403069f, 16.8758106f, 16.9408760f},
            .norm_eps = 1e-5f};
}

inline nn::TransformerConfig FM9GVTinyTextConfig() {
    auto config = FM9GVTextConfig();
    config.block_size = 128;
    config.n_layer = 2;
    config.n_head = 4;
    config.n_kv_head = 4;
    config.n_embd = 128;
    config.ffn_hidden_size = 256;
    config.q_lora_rank = 32;
    config.kv_lora_rank = 16;
    config.qk_nope_head_dim = 16;
    config.qk_rope_head_dim = 8;
    config.v_head_dim = 32;
    config.rope_short_factors.resize(config.qk_rope_head_dim / 2);
    return config;
}

inline void SanitizeFM9GVTextConfig(const nn::TransformerConfig &c) {
    CHECK(c.position_embedding_type == nn::PositionEmbeddingType::kRoPE);
    CHECK(c.attention_variant == nn::AttentionVariant::kFM9G);
    CHECK(c.activation_type == nn::MLPType::kSwiGLU);
    CHECK(c.norm_type == nn::NormType::kRMSNorm);
    CHECK_GT(c.ffn_hidden_size, 0);
    CHECK_GT(c.q_lora_rank, 0);
    CHECK_GT(c.kv_lora_rank, 0);
    CHECK_GT(c.qk_nope_head_dim, 0);
    CHECK_GT(c.qk_rope_head_dim, 0);
    CHECK_GT(c.v_head_dim, 0);
}

} // namespace fm9gv
