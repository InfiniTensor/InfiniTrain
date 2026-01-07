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

class HookHandle {
public:
    virtual ~HookHandle() = default;
    virtual void Remove() = 0;
};

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

// Forward pre-hook: called before forward pass
using FunctionForwardPreHook = std::function<void(Function*, const std::vector<std::shared_ptr<Tensor>>&)>;

// Forward post-hook: called after forward pass
using FunctionForwardPostHook = std::function<void(Function*, const std::vector<std::shared_ptr<Tensor>>&,
                                                    const std::vector<std::shared_ptr<Tensor>>&)>;

// Backward pre-hook: called before backward pass
using FunctionBackwardPreHook = std::function<void(Function*, const std::vector<std::shared_ptr<Tensor>>&)>;

// Backward post-hook: called after backward pass
using FunctionBackwardPostHook = std::function<void(Function*, const std::vector<std::shared_ptr<Tensor>>&,
                                                     const std::vector<std::shared_ptr<Tensor>>&)>;

template <typename HookType>
class FunctionHookHandleImpl : public HookHandle {
public:
    FunctionHookHandleImpl(std::vector<HookType>* hooks, size_t id) : hooks_(hooks), id_(id) {}

    void Remove() override {
        if (!removed_ && hooks_ && id_ < hooks_->size()) {
            (*hooks_)[id_] = nullptr;
            removed_ = true;
        }
    }

private:
    std::vector<HookType>* hooks_;
    size_t id_;
    bool removed_ = false;
};
} // namespace infini_train::autograd
