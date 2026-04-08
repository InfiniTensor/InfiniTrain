#pragma once

#include <vector>

#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/modules/transformer/spec_utils.h"
#include "infini_train/include/nn/modules/transformer/transformer_config.h"
#include "infini_train/include/nn/parallel/pp/pipeline_parallel.h"

namespace infini_train::nn {
class TransformerLayer : public CloneableModule<TransformerLayer> {
public:
    static constexpr char kType[] = "Layer";
    static constexpr char kLn1LayerName[] = "ln_1";
    static constexpr char kAttnLayerName[] = "attn";
    static constexpr char kLn2LayerName[] = "ln_2";
    static constexpr char kMlpLayerName[] = "mlp";

    explicit TransformerLayer(const TransformerConfig &config, const ModuleSpec &spec = {});

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;
};

class TransformerFirstStage : public CloneableModule<TransformerFirstStage> {
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

class TransformerChunk : public CloneableModule<TransformerChunk> {
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
};

class TransformerLastStage : public CloneableModule<TransformerLastStage> {
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

class TransformerModel : public CloneableModule<TransformerModel> {
public:
    static constexpr char kType[] = "Transformer";
    static constexpr char kTransformerModelName[] = "transformer";

    explicit TransformerModel(const TransformerConfig config, const ModuleSpec &spec = {});

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;

    const TransformerConfig &Config() const { return config_; }

private:
    const TransformerConfig config_;
    const infini_train::nn::parallel::StageInfo stage_info_;
    ModuleSpec spec_;
};

} // namespace infini_train::nn
