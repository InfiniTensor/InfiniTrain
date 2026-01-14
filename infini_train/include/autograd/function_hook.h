#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "infini_train/include/nn/parallel/reduce_op_type.h"

namespace infini_train {
class Tensor;

namespace nn::parallel {
class ProcessGroup;
} // namespace nn::parallel
} // namespace infini_train

namespace infini_train::autograd {
class Function;

class PostAccumulateGradHook {
public:
    virtual void operator()(const std::shared_ptr<Tensor> &tensor) = 0;
    virtual ~PostAccumulateGradHook() = default;
};

class AllReducePostAccumulateHook : public PostAccumulateGradHook {
public:
    AllReducePostAccumulateHook(infini_train::nn::parallel::function::ReduceOpType reduce_op,
                                const infini_train::nn::parallel::ProcessGroup *pg = nullptr);

    void operator()(const std::shared_ptr<Tensor> &tensor) override;

private:
    infini_train::nn::parallel::function::ReduceOpType reduce_op_;
    const infini_train::nn::parallel::ProcessGroup *pg_ = nullptr;
};

} // namespace infini_train::autograd
