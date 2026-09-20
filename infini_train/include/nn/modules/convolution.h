#pragma once

#include "infini_train/include/nn/modules/module.h"

namespace infini_train::nn {
// Contiguous FP32 NCHW input; square kernel, scalar stride and zero padding.
// Dilation and groups are fixed to 1. Weight layout: [out_channels, in_channels, k, k].
class Conv2d : public CloneableModule<Conv2d> {
public:
    static constexpr char kType[] = "Conv2d";
    static constexpr char kParamWeightName[] = "weight";
    static constexpr char kParamBiasName[] = "bias";
    Conv2d(int64_t in_channels, int64_t out_channels, int64_t kernel_size, int64_t stride = 1, int64_t padding = 0,
           bool bias = true, Device device = Device());
    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &inputs) override;
    bool has_bias() const { return bias_; }

private:
    int64_t stride_, padding_;
    bool bias_;
};
} // namespace infini_train::nn
