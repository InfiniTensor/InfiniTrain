#pragma once

#include "infini_train/include/autograd/function.h"

namespace infini_train::autograd {
class Conv2d : public Function {
public:
    static constexpr char kType[] = "Conv2dFunction";
    explicit Conv2d(int64_t stride = 1, int64_t padding = 0) : Function(kType), stride_(stride), padding_(padding) {}
    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &inputs) override;
    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &inputs,
                      const std::vector<std::shared_ptr<Tensor>> &outputs) override;
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grads) override;

private:
    int64_t stride_, padding_;
    bool bias_ = false;
};
} // namespace infini_train::autograd
