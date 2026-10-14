#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "infini_train/include/autograd/function.h"

namespace infini_train {
class Tensor;
}

namespace infini_train::autograd {

class Conv2d : public Function {
public:
    static constexpr char kType[] = "Conv2dFunction";

    Conv2d(int64_t stride, int64_t padding) : Function(kType), stride_(stride), padding_(padding) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    int64_t stride_ = 1;
    int64_t padding_ = 0;
    bool bias_ = false;
    int64_t in_channels_ = 0;
    int64_t out_channels_ = 0;
    int64_t kernel_size_ = 0;
    std::vector<int64_t> input_dims_;
};
} // namespace infini_train::autograd
