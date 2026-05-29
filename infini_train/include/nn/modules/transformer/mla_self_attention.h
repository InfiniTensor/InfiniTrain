#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/modules/transformer/transformer_config.h"

namespace infini_train::nn {

class MLASelfAttention : public infini_train::nn::CloneableModule<MLASelfAttention> {
public:
    static constexpr char kType[] = "MLASelfAttention";

    static constexpr char kLinearQProjLayerName[] = "linear_q_proj";
    static constexpr char kLinearQDownProjLayerName[] = "linear_q_down_proj";
    static constexpr char kQLayerNormLayerName[] = "q_layernorm";
    static constexpr char kLinearQUpProjLayerName[] = "linear_q_up_proj";
    static constexpr char kLinearKVDownProjLayerName[] = "linear_kv_down_proj";
    static constexpr char kKVLayerNormLayerName[] = "kv_layernorm";
    static constexpr char kLinearKVUpProjLayerName[] = "linear_kv_up_proj";
    static constexpr char kLinearProjLayerName[] = "linear_proj";

    static constexpr char kParamBiasName[] = "bias";

    explicit MLASelfAttention(const TransformerConfig &config);

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

    bool use_q_lora_ = true;
    bool q_down_proj_use_tp_ = false;
    bool kv_down_proj_use_tp_ = false;

    void SetupAttention(const TransformerConfig &config);
};

} // namespace infini_train::nn
