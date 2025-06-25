#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/autograd/function.h"

namespace infini_train::autograd {
class AccumulateGrad final : public Function {
public:
    explicit AccumulateGrad(std::shared_ptr<Tensor> grad, float learning_rate = 1.0f);

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &) override;

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &) override;

    void BackwardPartial(const std::shared_ptr<Tensor> &grad_output, int) override;

private:
    std::shared_ptr<Tensor> grad_ = nullptr;
    float learning_rate_ = 1.0f;
};
} // namespace infini_train::autograd
