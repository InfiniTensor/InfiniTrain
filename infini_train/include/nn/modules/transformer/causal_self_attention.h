#pragma once

#include <memory>
#include <tuple>
#include <vector>

#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/modules/transformer/transformer_config.h"

namespace infini_train::nn {

class CausalSelfAttention : public infini_train::nn::CloneableModule<CausalSelfAttention> {
public:
    static constexpr char kType[] = "CausalSelfAttention";
    static constexpr char kCAttnLayerName[] = "c_attn";
    static constexpr char kCProjLayerName[] = "c_proj";

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

    // Setup method for different attention modes
    void SetupAttention(const TransformerConfig &config);

    // Standard attention forward (GPT2 style: no RoPE, no GQA)
    std::vector<std::shared_ptr<infini_train::Tensor>>
    ForwardStandard(const std::vector<std::shared_ptr<infini_train::Tensor>> &x);

    // RoPE-aware attention forward (LLaMA3 style: with RoPE, optional GQA)
    std::vector<std::shared_ptr<infini_train::Tensor>>
    ForwardWithRoPE(const std::vector<std::shared_ptr<infini_train::Tensor>> &x);

    // RoPE helper methods
    std::tuple<std::shared_ptr<infini_train::Tensor>, std::shared_ptr<infini_train::Tensor>>
    ApplyRotaryEmbedding(const std::shared_ptr<infini_train::Tensor> &xq,
                         const std::shared_ptr<infini_train::Tensor> &xk,
                         const std::shared_ptr<infini_train::Tensor> &freqs_cis);

    // GQA helper method
    std::shared_ptr<infini_train::Tensor> RepeatKV(const std::shared_ptr<infini_train::Tensor> &x, int64_t n_rep);
};
} // namespace infini_train::nn
