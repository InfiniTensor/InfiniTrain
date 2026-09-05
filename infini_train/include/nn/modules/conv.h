#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/module.h"

namespace infini_train {
class Tensor;
class Device;
} // namespace infini_train

namespace infini_train::nn {
class Conv2d : public CloneableModule<Conv2d> {
public:
    static constexpr char kType[] = "Conv2d";

    static constexpr char kParamWeightName[] = "weight";
    static constexpr char kParamBiasName[] = "bias";

    Conv2d(int64_t in_channels, int64_t out_channels, int64_t kernel_size, bool bias = true, Device device = Device());
    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

    bool has_bias() const { return bias_; }

private:
    void ResetParameters();
    bool bias_ = true;
};
} // namespace infini_train::nn
