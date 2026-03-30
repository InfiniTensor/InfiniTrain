#pragma once

#include <vector>

#include "infini_train/include/core/transformer/spec_utils.h"
#include "infini_train/include/core/transformer/transformer_config.h"
#include "infini_train/include/nn/modules/module.h"

namespace infini_train::nn {
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
} // namespace infini_train::nn
