#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/modules/transformer/transformer_config.h"

namespace infini_train::nn::moe {

class TopKRouter : public CloneableModule<TopKRouter> {
public:
    static constexpr char kType[] = "TopKRouter";
    static constexpr char kParamWeightName[] = "weight";
    static constexpr char kParamBiasName[] = "bias";

    explicit TopKRouter(const TransformerConfig &config);

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

private:
    TransformerConfig config_;
};

} // namespace infini_train::nn::moe
