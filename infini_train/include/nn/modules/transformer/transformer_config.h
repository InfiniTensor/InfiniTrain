#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

namespace infini_train::nn {

enum class AttentionType {
    kStandard, // Standard attention
    kRoPE      // Rotary Position Embedding
};

enum class MLPType {
    kGELU,  // GELU activation
    kSwiGLU // SwiGLU activation
};

enum class NormType {
    kLayerNorm, // LayerNorm
    kRMSNorm    // RMSNorm
};

enum class ActivationRecomputeGranularity {
    kNone,      // Disable activation recompute.
    kFull,      // Recompute full transformer layers.
    kSelective, // Recompute selected transformer submodules.
};

enum class ActivationRecomputeMethod {
    kNone,    // Recompute every layer when granularity is full.
    kUniform, // Uniformly divide layers into recompute chunks.
    kBlock,   // Recompute only the first N layers in the local chunk/stage.
};

struct ActivationRecomputeOptions {
    ActivationRecomputeGranularity granularity = ActivationRecomputeGranularity::kNone;
    ActivationRecomputeMethod method = ActivationRecomputeMethod::kNone;
    int64_t num_layers = 0;
};

struct TransformerConfig {
    int64_t block_size = 1024;           // Max seq_len
    int64_t vocab_size = 50304;          // Vocab size
    int64_t original_vocab_size = 50257; // Original vocab size before padding
    int64_t n_layer = 12;                // Num of transformer layers
    int64_t n_head = 12;                 // Num of heads in MHA
    int64_t n_kv_head = 12;              // Num of Key/Value heads (<= n_head, < n_head if using GQA)
    int64_t n_embd = 768;                // Hidden size

    AttentionType attention_type = AttentionType::kStandard; // Attention mechanism type
    MLPType activation_type = MLPType::kGELU;                // MLP activation type
    NormType norm_type = NormType::kLayerNorm;               // Normalization type

    bool add_bias_linear = true; // Whether to add learnable bias to all Linear layers in the Transformer block,
                                 // including: attention QKV projection, attention output projection, MLP FC layers (and
                                 // SwiGLU second projection), and MLP output projection.
    bool add_bias_lm_head = false; // Whether to add bias to the LM head (output embedding).
    bool tie_weights = true;       // Tie embedding and lm_head weights

    // FFN config
    float ffn_expansion_ratio = 4.0f;               // MLP output: n_embd * ffn_expansion_ratio
    std::optional<float> ffn_dim_multiplier = 1.5f; // FFN dim multiplier
    int64_t multiple_of = 256;                      // FFN dims must be multiple of this number

    // RoPE config
    float rope_theta = 500000.0f; // theta in RoPE
    bool use_scaled_rope = false; // scaled RoPE

    // Normalization
    float norm_eps = 1e-5f; // epsilon in RMSNorm

    // Activation recomputation, aligned with Megatron-Core TransformerConfig.
    ActivationRecomputeGranularity recompute_granularity = ActivationRecomputeGranularity::kNone;
    ActivationRecomputeMethod recompute_method = ActivationRecomputeMethod::kNone;
    int64_t recompute_num_layers = 0;

    // Inference
    bool use_kv = false;            // kv cache
    bool flash = false;             // flash attention
    int64_t max_gen_batch_size = 4; // max batch size during inference

    bool UseGQA() const;
    int GetChunkSize() const;
    bool RecomputeEnabled() const;
};

void SetActivationRecomputeConfig(TransformerConfig *config, bool enabled, std::string_view granularity,
                                  std::string_view method, int64_t num_layers);
ActivationRecomputeOptions GetActivationRecomputeOptions(const TransformerConfig &config);
void ApplyActivationRecomputeOptions(TransformerConfig *config, const ActivationRecomputeOptions &options);
} // namespace infini_train::nn
