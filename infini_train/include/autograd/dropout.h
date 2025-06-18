#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/autograd/function.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
class Dropout : public Function {
public:
    static constexpr char kType[] = "DropoutFunction";

    Dropout(float p, bool training, bool inplace) : Function(kType), p_(p), training_(training), inplace_(inplace) {}
    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    const float p_;
    const bool training_;
    const bool inplace_;
};
} // namespace infini_train::autograd