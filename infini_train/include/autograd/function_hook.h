#pragma once

#include <memory>

#include "infini_train/include/nn/parallel_functional.h"
#include "infini_train/include/nn/reduce_op_type.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
class PostAccumulateGradHook {
public:
    virtual void operator()(const std::shared_ptr<Tensor> &tensor) = 0;
    virtual ~PostAccumulateGradHook() = default;
};

class AllReducePostAccumulateHook : public PostAccumulateGradHook {
public:
    AllReducePostAccumulateHook(infini_train::nn::parallel::function::ReduceOpType reduce_op) : reduce_op_(reduce_op) {}

    void operator()(const std::shared_ptr<Tensor> &tensor) override;

private:
    infini_train::nn::parallel::function::ReduceOpType reduce_op_;
};
} // namespace infini_train::autograd
