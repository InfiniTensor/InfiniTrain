#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/modules/transformer/transformer_config.h"

namespace infini_train::nn::moe {

class SequentialMLP : public CloneableModule<SequentialMLP> {
public:
    static constexpr char kType[] = "SequentialMLP";
    static constexpr char kExpertNamePrefix[] = "expert_";

    explicit SequentialMLP(const TransformerConfig &config);

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

private:
    TransformerConfig config_;
    int64_t num_local_experts_ = 0;
};

} // namespace infini_train::nn::moe
