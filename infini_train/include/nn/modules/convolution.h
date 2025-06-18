#pragma once

#include <memory>
#include <string>
#include <vector>

#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
class Conv2D : public Module {
public:
    static constexpr char kType[] = "Conv2D";

    static constexpr char kParamKernelName[] = "kernel";
    static constexpr char kParamBiasName[] = "bias";

    Conv2D(int64_t in_channels, int64_t out_channels, size_t kernel_size, size_t stride = 1, size_t padding = 0,
           size_t dilation = 1, size_t groups = 1, bool bias = true, std::string padding_mode = "zeros",
           Device device = Device());
    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

private:
    void ResetParameters();
    const int64_t in_channels_;
    const int64_t out_channels_;
    const size_t kernel_size_;
    const size_t stride_ = 1;
    const size_t padding_ = 0;
    const size_t dilation_ = 1;
    const size_t groups_ = 1;
    const bool bias_ = true;
    const std::string padding_mode_ = "zeros";
};
} // namespace infini_train::nn