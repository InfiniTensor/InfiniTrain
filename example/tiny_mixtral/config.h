#pragma once

#include "glog/logging.h"

#include "infini_train/include/nn/modules/transformer/transformer_config.h"

namespace nn = infini_train::nn;

namespace tiny_mixtral {

inline nn::TransformerConfig TinyMixtralConfig() {
    nn::TransformerConfig config;
    config.block_size = 32768;  // Same as Mixtral/Megatron --max-position-embeddings.
    config.vocab_size = 128256; // Validation data uses LLaMA3 token ids; real Mixtral uses 32000.
    config.original_vocab_size = 128256;
    config.n_layer = 2;   // Tiny scale; Megatron --num-layers 32.
    config.n_head = 4;    // Tiny scale; preserves the Megatron 4:1 GQA ratio.
    config.n_kv_head = 1; // Tiny scale; Megatron --num-query-groups 8.
    config.n_embd = 32;   // Tiny scale; Megatron --hidden-size 4096.
    config.attention_type = nn::AttentionType::kRoPE;
    config.activation_type = nn::MLPType::kSwiGLU;
    config.ffn_type = nn::FFNType::kMoE;
    config.norm_type = nn::NormType::kRMSNorm;
    config.add_bias_linear = false;
    config.add_bias_lm_head = false;
    config.tie_weights = false;
    config.ffn_expansion_ratio = 3.5f;
    config.norm_eps = 1e-5f;
    config.rope_theta = 1000000.0f;
    config.use_scaled_rope = false;

    nn::MoEConfig moe_config;
    moe_config.num_experts = 8;
    moe_config.expert_parallel_size = 1; // Single-rank validation scale.
    moe_config.router_topk = 2;
    moe_config.moe_ffn_hidden_size = 112; // Tiny scale; Megatron --ffn-hidden-size 14336.
    moe_config.token_dispatcher_type = nn::MoEConfig::TokenDispatcherType::kAllGather; // Single-rank validation path.
    moe_config.expert_impl = nn::MoEConfig::ExpertImpl::kSequential;                   // Local correctness path.
    config.moe_config = moe_config;
    return config;
}

inline void SanitizeTinyMixtralConfig(const nn::TransformerConfig &c) {
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
    CHECK(c.attention_type == nn::AttentionType::kRoPE) << "tiny Mixtral requires RoPE attention";
    CHECK(c.activation_type == nn::MLPType::kSwiGLU) << "tiny Mixtral requires SwiGLU activation";
    CHECK(c.ffn_type == nn::FFNType::kMoE) << "tiny Mixtral requires MoE FFN";
    CHECK(c.norm_type == nn::NormType::kRMSNorm) << "tiny Mixtral requires RMSNorm";
    CHECK(!c.add_bias_linear) << "tiny Mixtral has no bias in linear layers";
    CHECK(!c.add_bias_lm_head) << "tiny Mixtral has no bias in lm_head";
    CHECK(!c.tie_weights) << "tiny Mixtral does not tie embedding and lm_head weights";
    CHECK(!c.use_scaled_rope) << "tiny Mixtral precision validation keeps scaled RoPE disabled";
    CHECK(c.moe_config.has_value()) << "tiny Mixtral requires MoE config";

    const auto &moe = c.moe_config.value();
    CHECK_GT(moe.num_experts, 0);
    CHECK_EQ(moe.expert_parallel_size, 1) << "tiny Mixtral single-rank validation expects EP=1";
    CHECK_GT(moe.router_topk, 0);
    CHECK_LE(moe.router_topk, moe.num_experts);
    CHECK_GT(moe.moe_ffn_hidden_size, 0);
    CHECK(moe.token_dispatcher_type == nn::MoEConfig::TokenDispatcherType::kAllGather)
        << "tiny Mixtral uses the Megatron-style AllGather dispatcher";
    CHECK(moe.expert_impl == nn::MoEConfig::ExpertImpl::kSequential)
        << "tiny Mixtral validation uses SequentialMLP experts";
}

} // namespace tiny_mixtral
