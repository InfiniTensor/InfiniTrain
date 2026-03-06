#pragma once

#include <vector>

#include "infini_train/include/core/transformer/spec_utils.h"
#include "infini_train/include/core/transformer/transformer_config.h"
#include "infini_train/include/nn/modules/module.h"

namespace infini_train::nn {

class RMSNorm : public infini_train::nn::CloneableModule<RMSNorm> {
public:
    static constexpr char kType[] = "RMSNorm";
    static constexpr char kParamWeightName[] = "weight";

    explicit RMSNorm(int64_t dim, float eps = 1e-6f, infini_train::Device device = infini_train::Device());

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;

private:
    float eps_ = 1e-5f;
};

class NewGELU : public infini_train::nn::CloneableModule<NewGELU> {
public:
    static constexpr char kType[] = "NewGELU";
    NewGELU() : CloneableModule(kType) {}

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;
};

class CausalSelfAttention : public infini_train::nn::CloneableModule<CausalSelfAttention> {
public:
    static constexpr char kType[] = "CausalSelfAttention";
    static constexpr char kCAttnLayerName[] = "c_attn";
    static constexpr char kCProjLayerName[] = "c_proj";

    static constexpr char kParamBiasName[] = "bias";

    explicit CausalSelfAttention(const TransformerConfig &config, const ModuleSpec &spec = {});

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
};

class MLP : public infini_train::nn::CloneableModule<MLP> {
public:
    static constexpr char kType[] = "MLP";
    static constexpr char kCFcLayerName[] = "c_fc";
    static constexpr char kGeluLayerName[] = "gelu";
    static constexpr char kCProjLayerName[] = "c_proj";

    static constexpr char kCFc2LayerName[] = "c_fc2";
    static constexpr char kSiluLayerName[] = "silu";

    explicit MLP(const TransformerConfig &config, const ModuleSpec &spec = {});

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;
};

struct TransformerBlockSubmodules {
    ModuleSpec ln1;
    ModuleSpec attn;
    ModuleSpec ln2;
    ModuleSpec mlp;
};

class TransformerBlock : public infini_train::nn::CloneableModule<TransformerBlock> {
public:
    static constexpr char kType[] = "Block";
    static constexpr char kLn1LayerName[] = "ln_1";
    static constexpr char kAttnLayerName[] = "attn";
    static constexpr char kLn2LayerName[] = "ln_2";
    static constexpr char kMlpLayerName[] = "mlp";

    explicit TransformerBlock(const TransformerConfig &config, const ModuleSpec &spec = {});

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;
};

} // namespace infini_train::nn