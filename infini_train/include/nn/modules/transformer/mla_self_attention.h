#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/modules/transformer/transformer_config.h"

namespace infini_train::nn {

class MLASelfAttention : public infini_train::nn::CloneableModule<MLASelfAttention> {
public:
    static constexpr char kType[] = "MLASelfAttention";

    static constexpr char kQAProjLayerName[] = "q_a_proj";
    static constexpr char kQANormLayerName[] = "q_a_layernorm";
    static constexpr char kQBProjLayerName[] = "q_b_proj";
    static constexpr char kKVAProjLayerName[] = "kv_a_proj_with_mqa";
    static constexpr char kKVANormLayerName[] = "kv_a_layernorm";
    static constexpr char kKVBProjLayerName[] = "kv_b_proj";
    static constexpr char kCProjLayerName[] = "c_proj";

    static constexpr char kParamBiasName[] = "bias";

    explicit MLASelfAttention(const TransformerConfig &config);
    MLASelfAttention(const TransformerConfig &config, int64_t q_lora_rank, int64_t kv_lora_rank,
                     int64_t qk_nope_head_dim, int64_t qk_rope_head_dim, int64_t v_head_dim);

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;

private:
    TransformerConfig config_;
    int64_t n_head_ = 0;
    int64_t n_embd_ = 0;
    int64_t local_n_head_ = 0;

    int64_t q_lora_rank_ = 0;
    int64_t kv_lora_rank_ = 0;
    int64_t qk_nope_head_dim_ = 0;
    int64_t qk_rope_head_dim_ = 0;
    int64_t qk_head_dim_ = 0;
    int64_t v_head_dim_ = 0;

    void SetupAttention(const TransformerConfig &config, int64_t q_lora_rank, int64_t kv_lora_rank,
                        int64_t qk_nope_head_dim, int64_t qk_rope_head_dim, int64_t v_head_dim);

};

} // namespace infini_train::nn
