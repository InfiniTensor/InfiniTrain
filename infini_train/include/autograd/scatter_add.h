#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/autograd/function.h"

namespace infini_train {
class Tensor;
}

namespace infini_train::autograd {

class ScatterAdd : public Function {
public:
    static constexpr char kType[] = "ScatterAddFunction";

    ScatterAdd(int64_t dim, const std::vector<int64_t> &output_dims)
        : Function(kType), dim_(dim), output_dims_(output_dims) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    int64_t dim_ = 0;
    std::vector<int64_t> output_dims_;
};

} // namespace infini_train::autograd
