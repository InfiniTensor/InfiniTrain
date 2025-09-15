#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/nn/modules/module.h"

namespace infini_train {
class Tensor;
class Device;
} // namespace infini_train

namespace infini_train::nn {
class Linear : public CloneableModule<Linear> {
public:
    static constexpr char kType[] = "Linear";

    static constexpr char kParamWeightName[] = "weight";
    static constexpr char kParamBiasName[] = "bias";

    Linear(int64_t in_features, int64_t out_features, bool bias = true, const Device *device = nullptr);
    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

private:
    void ResetParameters();
    bool bias_ = true;
};
} // namespace infini_train::nn
