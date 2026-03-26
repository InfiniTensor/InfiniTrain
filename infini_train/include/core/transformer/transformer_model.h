#pragma once

#include "infini_train/include/core/transformer/spec_utils.h"
#include "infini_train/include/core/transformer/transformer_layer.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/pp/pipeline_parallel.h"

namespace infini_train::nn {
class TransformerConfig;

class TransformerFirstStage : public infini_train::nn::CloneableModule<TransformerFirstStage> {
public:
    static constexpr char kType[] = "TransformerFirstStage";
    static constexpr char kWTELayerName[] = "wte";
    static constexpr char kWPELayerName[] = "wpe";

    explicit TransformerFirstStage(const TransformerConfig &config, const ModuleSpec &spec = {});

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;

private:
    TransformerConfig config_;
    ModuleSpec spec_;
};

class TransformerChunk : public infini_train::nn::CloneableModule<TransformerChunk> {
public:
    static constexpr char kType[] = "TransformerChunk";
    static constexpr char kHLayerName[] = "h";
    static constexpr char kFreqsCisName[] = "freqs_cis";

    TransformerChunk(const TransformerConfig &config, int start_layer, int end_layer, const ModuleSpec &spec = {});

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;

private:
    const TransformerConfig config_;
    ModuleSpec spec_;

    // RoPE helper method
    std::shared_ptr<infini_train::Tensor> PrecomputeFreqsCis(int64_t dim, int64_t end, float theta = 10000.0f,
                                                             bool use_scaled = false,
                                                             infini_train::Device device = infini_train::Device());
};

class TransformerLastStage : public infini_train::nn::CloneableModule<TransformerLastStage> {
public:
    static constexpr char kType[] = "TransformerLastStage";
    static constexpr char kLnFLayerName[] = "ln_f";
    static constexpr char kLMHeadLayerName[] = "lm_head";

    explicit TransformerLastStage(const TransformerConfig &config, const ModuleSpec &spec = {});

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;

private:
    const TransformerConfig config_;
    ModuleSpec spec_;
};

class TransformerModel : public infini_train::nn::CloneableModule<TransformerModel> {
public:
    static constexpr char kType[] = "Transformer";
    static constexpr char kTransformerModelName[] = "transformer";

    explicit TransformerModel(const TransformerConfig config, const ModuleSpec &spec = {});

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;

    const TransformerConfig &GetConfig() const { return config_; }

    const TransformerConfig config_;

private:
    const infini_train::nn::parallel::StageInfo stage_info_;
    ModuleSpec spec_;
};
} // namespace infini_train::nn
