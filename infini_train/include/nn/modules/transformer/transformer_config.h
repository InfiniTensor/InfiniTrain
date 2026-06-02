#pragma once

#include <cstdint>
#include <optional>

namespace infini_train::nn {

enum class ModelType {
    kGPT2,   // GPT-2
    kLLaMA3, // LLaMA3
};

enum class PositionEmbeddingType {
    kLearnedAbsolute, // Megatron: learned_absolute
    kRoPE,            // Megatron: rope
    kYarn,            // Megatron: yarn
    kMRoPE,           // Megatron: mrope
    kRelative,        // Megatron: relative
    kNone             // Megatron: none
};

enum class MLPType {
    kGELU,  // GELU activation
    kSwiGLU // SwiGLU activation
};

enum class NormType {
    kLayerNorm, // LayerNorm
    kRMSNorm    // RMSNorm
};

struct TransformerConfig {
    int64_t block_size = 1024;           // Max seq_len
    int64_t vocab_size = 50304;          // Vocab size
    int64_t original_vocab_size = 50257; // Original vocab size before padding
    int64_t n_layer = 12;                // Num of transformer layers
    int64_t n_head = 12;                 // Num of heads in MHA
    int64_t n_kv_head = 12;              // Num of Key/Value heads (<= n_head, < n_head if using GQA)
    int64_t n_embd = 768;                // Hidden size

    PositionEmbeddingType position_embedding_type = PositionEmbeddingType::kLearnedAbsolute; // Position embedding type.
    MLPType activation_type = MLPType::kGELU;                                                // MLP activation type
    NormType norm_type = NormType::kLayerNorm;                                               // Normalization type

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

    // MLA config
    bool multi_latent_attention = false;               // Use MLA instead of standard causal self-attention.
    std::optional<int64_t> q_lora_rank = std::nullopt; // nullopt means direct linear_q_proj path.
    int64_t kv_lora_rank = 0;                          // 0 falls back to n_embd in MLASelfAttention.
    int64_t qk_nope_head_dim = 0;                      // 0 falls back to n_embd / n_head.
    int64_t qk_rope_head_dim = 0;                      // 0 falls back to n_embd / n_head.
    int64_t v_head_dim = 0;                            // 0 falls back to n_embd / n_head.
    bool q_down_proj_use_tp = false;                   // Use ColumnParallelLinear for linear_q_down_proj.
    bool kv_down_proj_use_tp = false;                  // Use ColumnParallelLinear for linear_kv_down_proj.

    // Normalization
    float norm_eps = 1e-5f; // epsilon in RMSNorm

    // Inference
    bool use_kv = false;            // kv cache
    bool flash = false;             // flash attention
    int64_t max_gen_batch_size = 4; // max batch size during inference

    bool UseGQA() const;
    int GetChunkSize() const;
};
} // namespace infini_train::nn
