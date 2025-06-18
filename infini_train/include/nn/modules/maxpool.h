#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
class MaxPool2D : public Module {
public:
    static constexpr char kType[] = "MaxPool2D";

    MaxPool2D(size_t kernel_size, size_t stride, size_t padding = 0, size_t dilation = 1, Device device = Device());
    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

private:
    void ResetParameters();
    const size_t kernel_size_;
    const size_t stride_;
    const size_t padding_ = 0;
    const size_t dilation_ = 1;
};
} // namespace infini_train::nn
