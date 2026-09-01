#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/modules/transformer/transformer_config.h"

namespace infini_train::nn {

class CausalSelfAttention : public infini_train::nn::CloneableModule<CausalSelfAttention> {
public:
    static constexpr char kType[] = "CausalSelfAttention";
    static constexpr char kCAttnLayerName[] = "c_attn";
    static constexpr char kCProjLayerName[] = "c_proj";
    static constexpr char kQAProjLayerName[] = "q_a_proj";
    static constexpr char kQALayerNormLayerName[] = "q_a_layernorm";
    static constexpr char kQBProjLayerName[] = "q_b_proj";
    static constexpr char kKVAProjWithMQALayerName[] = "kv_a_proj_with_mqa";
    static constexpr char kKVALayerNormLayerName[] = "kv_a_layernorm";
    static constexpr char kKVBProjLayerName[] = "kv_b_proj";
    static constexpr char kOProjLayerName[] = "o_proj";

    static constexpr char kParamBiasName[] = "bias";

    explicit CausalSelfAttention(const TransformerConfig &config);

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;

private:
    TransformerConfig config_;
    int64_t n_head_ = 0;
    int64_t n_embd_ = 0;
    int64_t local_n_head_ = 0;

    int64_t n_kv_head_ = 0;
    int64_t n_rep_ = 0;
    int64_t head_dim_ = 0;
    int64_t q_lora_rank_ = 0;
    int64_t kv_lora_rank_ = 0;
    int64_t qk_nope_head_dim_ = 0;
    int64_t qk_rope_head_dim_ = 0;
    int64_t v_head_dim_ = 0;
    int64_t q_head_dim_ = 0;

    // Setup method for different attention modes
    void SetupAttention(const TransformerConfig &config);

    // GQA helper method
    std::shared_ptr<infini_train::Tensor> RepeatKV(const std::shared_ptr<infini_train::Tensor> &x, int64_t n_rep);
    std::vector<std::shared_ptr<infini_train::Tensor>>
    ForwardFM9G(const std::vector<std::shared_ptr<infini_train::Tensor>> &x);
};
} // namespace infini_train::nn
