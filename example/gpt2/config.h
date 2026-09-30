#pragma once

#include <string>
#include <unordered_map>

#include "glog/logging.h"

#include "infini_train/include/nn/modules/transformer/transformer_config.h"

namespace nn = infini_train::nn;
namespace gpt2 {
struct GPT2Size {
    int64_t layers;
    int64_t heads;
    int64_t hidden;
};
inline const std::unordered_map<std::string, GPT2Size> kModelSizes = {
    {"gpt2", {12, 12, 768}},        {"d12", {12, 12, 768}},  {"gpt2-medium", {24, 16, 1024}}, {"d24", {24, 16, 1024}},
    {"gpt2-large", {36, 20, 1280}}, {"d36", {36, 20, 1280}}, {"gpt2-xl", {48, 25, 1600}},     {"d48", {48, 25, 1600}},
};

inline nn::TransformerConfig GPT2Config() {
    return {.block_size = 1024,
            .vocab_size = 50304,
            .original_vocab_size = 50257,
            .n_layer = 12,
            .n_head = 12,
            .n_kv_head = 12,
            .n_embd = 768,
            .position_embedding_type = nn::PositionEmbeddingType::kLearnedAbsolute,
            .activation_type = nn::MLPType::kGELU,
            .norm_type = nn::NormType::kLayerNorm,
            .add_bias_linear = true,
            .add_bias_lm_head = false,
            .tie_weights = true,
            .ffn_expansion_ratio = 4.0f,
            .ffn_dim_multiplier = std::nullopt,
            .multiple_of = 1,
            .rotary_interleaved = true};
}

inline nn::TransformerConfig GPT2Config(const std::string &name, int tensor_parallel_size) {
    CHECK_GT(tensor_parallel_size, 0);
    const auto &size = kModelSizes.at(name);
    auto config = GPT2Config();
    config.n_layer = size.layers;
    config.n_head = config.n_kv_head = size.heads;
    config.n_embd = size.hidden;
    // Scratch models pad vocabulary only as required by TP.
    config.vocab_size
        = (config.original_vocab_size + tensor_parallel_size - 1) / tensor_parallel_size * tensor_parallel_size;
    return config;
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
    CHECK(c.position_embedding_type == nn::PositionEmbeddingType::kLearnedAbsolute)
        << "GPT-2 requires learned absolute position embedding";
    CHECK(c.activation_type == nn::MLPType::kGELU) << "GPT-2 requires GELU activation";
    CHECK(c.norm_type == nn::NormType::kLayerNorm) << "GPT-2 requires LayerNorm";
}

} // namespace gpt2
