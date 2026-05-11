#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/modules/transformer/transformer_config.h"

namespace infini_train::nn::moe {

class MoELayer : public CloneableModule<MoELayer> {
public:
    static constexpr char kType[] = "MoELayer";
    static constexpr char kRouterLayerName[] = "router";
    static constexpr char kExpertsLayerName[] = "experts";

    explicit MoELayer(const TransformerConfig &config);

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

private:
    TransformerConfig config_;
};

} // namespace infini_train::nn::moe
