#pragma once
#include <cstdint>
#include <optional>

namespace infini_train::nn {

struct TransformerConfig {
    int64_t block_size = 8192;
    ; // Max seq_len
    int64_t vocab_size = 128256;
    ;                            // Vocab size
    int64_t original_vocab_size; // Original vocab size before padding
    int64_t n_layer = 16;        // Num of transformer layers
    int64_t n_head = 32;         // Num of heads in MHA
    int64_t n_kv_head = 8;       // Num of Key/Value heads（< n_head if using GQA）
    int64_t n_embd = 2048;       // Hidden size

    // FFN config
    std::optional<float> ffn_dim_multiplier = 1.5f; // FFN dim multiplier
    int64_t multiple_of = 256;                      // FFN dims must be multiple of this number

    // Pos embedding
    float rope_theta = 500000.0f; // theta in RoPE
    bool use_scaled_rope = false; // scaled RoPE

    // RMSNorm
    float norm_eps = 1e-5f; // epsilon in RMSNorm

    // Inference
    bool use_kv = false;            // kv cache
    bool flash = false;             // flash attention
    int64_t max_gen_batch_size = 4; // max batch size during inference
};

} // namespace infini_train::nn
