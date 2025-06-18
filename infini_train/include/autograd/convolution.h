#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/autograd/function.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
class Conv2D : public Function {
public:
    static constexpr char kType[] = "Conv2DFunction";

    Conv2D(int64_t in_channels, int64_t out_channels, size_t kernel_size, size_t stride, size_t padding,
           size_t dilation, size_t groups, bool bias)
        : Function(kType), in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size),
          stride_(stride), padding_(padding), dilation_(dilation), groups_(groups), bias_(bias) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    const int64_t in_channels_;
    const int64_t out_channels_;
    const size_t kernel_size_;
    const size_t stride_;
    const size_t padding_;
    const size_t dilation_;
    const size_t groups_;
    bool bias_;
};
} // namespace infini_train::autograd