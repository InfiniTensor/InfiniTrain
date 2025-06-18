#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/autograd/function.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
class MaxPool2D : public Function {
public:
    static constexpr char kType[] = "MaxPool2DFunction";

    MaxPool2D(size_t kernel_size, size_t stride, size_t padding, size_t dilation)
        : Function(kType), kernel_size_(kernel_size), stride_(stride), padding_(padding), dilation_(dilation) {}
    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    const size_t kernel_size_;
    const size_t stride_;
    const size_t padding_;
    const size_t dilation_;
};
} // namespace infini_train::autograd