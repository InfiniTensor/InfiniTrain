#pragma once

#include <cstdint>
#include <optional>

namespace infini_train::nn {

enum class AttentionType {
    kStandard, // Standard attention
    kRoPE      // Rotary Position Embedding
};

enum class MLPType {
    kGELU,  // GELU activation
    kSwiGLU // SwiGLU activation
};

enum class FFNType {
    kDense, // Standard dense MLP
    kMoE    // Mixture-of-Experts MLP
};

enum class NormType {
    kLayerNorm, // LayerNorm
    kRMSNorm    // RMSNorm
};

enum class MoERouterType {
    kTopK // Top-k router. The initial implementation supports top-1.
};

enum class MoEDispatcherType {
    kLocal,    // No cross-rank token exchange
    kAllGather // Reserved for expert parallel MoE
};

enum class MoEExpertImpl {
    kSequential // Run local experts sequentially
};

struct MoEConfig {
    int64_t num_experts = 0;
    int64_t expert_parallel_size = 1;
    int64_t router_topk = 1;
    float aux_loss_coeff = 0.0f;
    std::optional<float> expert_capacity_factor = std::nullopt;
    bool pad_expert_input_to_capacity = false;
    int64_t moe_ffn_hidden_size = 0;
    MoERouterType router_type = MoERouterType::kTopK;
    MoEDispatcherType dispatcher_type = MoEDispatcherType::kLocal;
    MoEExpertImpl expert_impl = MoEExpertImpl::kSequential;
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
    FFNType ffn_type = FFNType::kDense;                      // Feed-forward module type
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
    std::optional<MoEConfig> moe_config = std::nullopt;

    // RoPE config
    float rope_theta = 500000.0f; // theta in RoPE
    bool use_scaled_rope = false; // scaled RoPE

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
