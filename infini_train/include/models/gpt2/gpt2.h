#pragma once

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/modules/transformer/config.h"
#include "infini_train/include/nn/modules/transformer/transformer_kernel.h"
#include "infini_train/include/nn/parallel/pp/pipeline_parallel.h"
#include "infini_train/include/nn/parallel/pp/pipeline_stage.h"
#include "infini_train/include/tensor.h"

class NewGELU : public infini_train::nn::CloneableModule<NewGELU> {
public:
    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;
};

class CausalSelfAttention : public infini_train::nn::CloneableModule<CausalSelfAttention> {
public:
    static constexpr char kCAttnLayerName[] = "c_attn";
    static constexpr char kCProjLayerName[] = "c_proj";

    static constexpr char kParamBiasName[] = "bias";

    explicit CausalSelfAttention(const infini_train::nn::TransformerConfig &config);

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;

private:
    infini_train::nn::TransformerConfig config_;
    int64_t n_head_ = 0;
    int64_t n_embd_ = 0;

    int64_t local_n_head_ = 0;
};

class MLP : public infini_train::nn::CloneableModule<MLP> {
public:
    static constexpr char kCFcLayerName[] = "c_fc";
    static constexpr char kGeluLayerName[] = "gelu";
    static constexpr char kCProjLayerName[] = "c_proj";

    explicit MLP(const infini_train::nn::TransformerConfig &config);

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;
};

class Block : public infini_train::nn::CloneableModule<Block> {
public:
    static constexpr char kLn1LayerName[] = "ln_1";
    static constexpr char kAttnLayerName[] = "attn";
    static constexpr char kLn2LayerName[] = "ln_2";
    static constexpr char kMlpLayerName[] = "mlp";

    explicit Block(const infini_train::nn::TransformerConfig &config);

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;
};

class GPT2Kernel : public infini_train::nn::TransformerKernel {
public:
    bool UseAbsolutePositionEmbedding() const override { return true; }

    std::shared_ptr<infini_train::nn::Module> MakeBlock(const infini_train::nn::TransformerConfig &config) override;

    std::shared_ptr<infini_train::nn::Module> MakeFinalNorm(const infini_train::nn::TransformerConfig &config) override;
};

class GPT2 : public infini_train::nn::CloneableModule<GPT2> {
public:
    static constexpr char kTransformerLayerName[] = "transformer";

    enum class ModelType : int8_t {
        kGPT2,
        kGPT2Medium,
        kGPT2Large,
        kGPT2XL,
    };

    explicit GPT2(const infini_train::nn::TransformerConfig &config);

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;

    static std::shared_ptr<GPT2> FromPretrained(ModelType model_type);
    static std::shared_ptr<GPT2> FromLLMC(const std::string &filepath);

    int GetChunkSize() const;

private:
    const infini_train::nn::TransformerConfig config_;
    const infini_train::nn::parallel::StageInfo stage_info_;
};
