#pragma once
#include <cstdint>
#include <optional>
#include <string>

namespace infini_train::nn {

enum class AttentionType {
    kStandard, // Standard attention (GPT2 style, no RoPE)
    kRoPE      // Rotary Position Embedding (LLaMA3 style)
};

enum class MLPType {
    kGELU,  // GELU activation (GPT2 style)
    kSwiGLU // SwiGLU activation (LLaMA3 style)
};

enum class NormType {
    kLayerNorm, // LayerNorm (GPT2 style)
    kRMSNorm    // RMSNorm (LLaMA3 style)
};

class TransformerConfig {
public:
    static constexpr char kGPT2Name[] = "GPT2";
    static constexpr char kLLaMA3Name[] = "LLaMA3";

    std::string model_type = "";

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

    bool use_bias = true;    // Linear layers bias (GPT2: true, LLaMA3: false)
    bool use_gqa = false;    // Grouped Query Attention
    bool use_rope = false;   // Rotary Position Embedding
    bool tie_weights = true; // Tie embedding and lm_head weights

    // FFN config
    float ffn_expansion_ratio = 4.0f;               // MLP output: n_embd * ffn_expansion_ratio
    std::optional<float> ffn_dim_multiplier = 1.5f; // FFN dim multiplier
    int64_t multiple_of = 256;                      // FFN dims must be multiple of this number

    // RoPE config
    float rope_theta = 500000.0f; // theta in RoPE
    bool use_scaled_rope = false; // scaled RoPE

    // Normalization
    float norm_eps = 1e-5f; // epsilon in RMSNorm

    // Inference
    bool use_kv = false;            // kv cache
    bool flash = false;             // flash attention
    int64_t max_gen_batch_size = 4; // max batch size during inference

    static TransformerConfig GPT2() { return {}; }

    static TransformerConfig LLaMA3() {
        return {.model_type = kLLaMA3Name,
                .block_size = 8192,
                .vocab_size = 128256,
                .n_layer = 16,
                .n_head = 32,
                .n_kv_head = 8,
                .n_embd = 2048,
                .attention_type = AttentionType::kRoPE,
                .activation_type = MLPType::kSwiGLU,
                .norm_type = NormType::kRMSNorm,
                .use_bias = false,
                .use_gqa = true,
                .use_rope = true,
                .tie_weights = false};
    }
};
} // namespace infini_train::nn
