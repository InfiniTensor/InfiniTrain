#pragma once

#include <cstdint>
#include <vector>

#include "infini_train/include/tensor.h"

namespace infini_train {
class Optimizer {
public:
    explicit Optimizer(const std::vector<std::shared_ptr<Tensor>> &params);

    void ZeroGrad();

    virtual void Step() = 0;

protected:
    std::vector<std::shared_ptr<Tensor>> params_;
};

namespace optimizers {
class SGD : public Optimizer {
public:
    SGD(const std::vector<std::shared_ptr<Tensor>> &params, float learning_rate);

    void Step() override;

private:
    const float learning_rate_ = 0.0;
};

class Adam : public Optimizer {
public:
    Adam(const std::vector<std::shared_ptr<Tensor>> &params, float learning_rate = 1e-3, float beta1 = 0.9,
         float beta2 = 0.999, float eps = 1e-8);

    void Step() override;

private:
    int64_t t_;
    const float learning_rate_;
    const float beta1_;
    const float beta2_;
    const float eps_;
    std::vector<std::shared_ptr<Tensor>> m_;
    std::vector<std::shared_ptr<Tensor>> v_;
};
} // namespace optimizers
} // namespace infini_train
