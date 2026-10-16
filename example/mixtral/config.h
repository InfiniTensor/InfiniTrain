#pragma once

#include <string_view>

#include "glog/logging.h"

#include "infini_train/include/nn/modules/transformer/transformer_config.h"

namespace nn = infini_train::nn;

namespace mixtral {

inline constexpr std::string_view kTinyModelPreset = "tiny";
inline constexpr std::string_view kMixtral8x7BModelPreset = "mixtral-8x7b";

inline nn::TransformerConfig TinyMixtralConfig() {
    nn::TransformerConfig config;
    config.block_size = 32768;  // Same as Mixtral/Megatron --max-position-embeddings.
    config.vocab_size = 128256; // Validation data uses LLaMA3 token ids; real Mixtral uses 32000.
    config.original_vocab_size = 128256;
    config.n_layer = 32;
    config.n_head = 4;    // Scaled down; preserves Mixtral 4:1 GQA ratio.
    config.n_kv_head = 1; // Scaled down with n_head; real Mixtral uses 8 KV heads.
    config.n_embd = 512;  // Scaled down from Mixtral/Megatron --hidden-size 4096.
    config.position_embedding_type = nn::PositionEmbeddingType::kRoPE;
    config.activation_type = nn::MLPType::kSwiGLU;
    config.ffn_type = nn::FFNType::kMoE;
    config.norm_type = nn::NormType::kRMSNorm;
    config.add_bias_linear = false;
    config.add_bias_lm_head = false;
    config.tie_weights = false;
    config.norm_eps = 1e-5f;
    config.rope_theta = 1000000.0f;
    config.use_scaled_rope = false;
    config.rotary_interleaved = true;

    nn::MoEConfig moe_config;
    moe_config.num_experts = 8;
    moe_config.router_topk = 2;
    moe_config.moe_ffn_hidden_size = 1792; // Scaled down as 512 * 3.5; real Mixtral uses 14336.
    moe_config.token_dispatcher_type = nn::MoEConfig::TokenDispatcherType::kAllGather; // Single-rank validation path.
    moe_config.expert_impl = nn::MoEConfig::ExpertImpl::kSequential;                   // Local correctness path.
    config.moe_config = moe_config;
    return config;
}

inline nn::TransformerConfig Mixtral8x7BConfig() {
    auto config = TinyMixtralConfig();
    config.vocab_size = 32000;
    config.original_vocab_size = 32000;
    config.n_head = 32;
    config.n_kv_head = 8;
    config.n_embd = 4096;
    config.rotary_interleaved = false;
    config.moe_config->moe_ffn_hidden_size = 14336;
    return config;
}

inline nn::TransformerConfig ConfigForPreset(std::string_view preset) {
    if (preset == kTinyModelPreset) {
        return TinyMixtralConfig();
    }
    if (preset == kMixtral8x7BModelPreset) {
        return Mixtral8x7BConfig();
    }
    LOG(FATAL) << "Unsupported Mixtral model preset: " << preset
               << ". Expected " << kTinyModelPreset << " or " << kMixtral8x7BModelPreset;
    return {};
}

inline void SanitizeMixtralConfig(const nn::TransformerConfig &c) {
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
    CHECK(c.position_embedding_type == nn::PositionEmbeddingType::kRoPE)
        << "Mixtral requires RoPE position embedding";
    CHECK(c.activation_type == nn::MLPType::kSwiGLU) << "Mixtral requires SwiGLU activation";
    CHECK(c.ffn_type == nn::FFNType::kMoE) << "Mixtral requires MoE FFN";
    CHECK(c.norm_type == nn::NormType::kRMSNorm) << "Mixtral requires RMSNorm";
    CHECK(!c.add_bias_linear) << "Mixtral has no bias in linear layers";
    CHECK(!c.add_bias_lm_head) << "Mixtral has no bias in lm_head";
    CHECK(!c.tie_weights) << "Mixtral does not tie embedding and lm_head weights";
    CHECK(!c.use_scaled_rope) << "Mixtral uses unscaled RoPE";
    CHECK(c.moe_config.has_value()) << "Mixtral requires MoE config";

    const auto &moe = c.moe_config.value();
    CHECK_GT(moe.num_experts, 0);
    CHECK_GT(moe.router_topk, 0);
    CHECK_LE(moe.router_topk, moe.num_experts);
    CHECK_GT(moe.moe_ffn_hidden_size, 0);
    CHECK(moe.token_dispatcher_type == nn::MoEConfig::TokenDispatcherType::kAllGather)
        << "Mixtral uses the Megatron-style AllGather dispatcher";
    CHECK(moe.expert_impl == nn::MoEConfig::ExpertImpl::kSequential)
        << "Mixtral validation uses SequentialMLP experts";
}

} // namespace mixtral
