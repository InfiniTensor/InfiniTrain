#pragma once

#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace infini_train {
class Tensor;
}

namespace infini_train::autograd {
class HookHandle;
using FunctionForwardPreHook = std::function<void(class Function*, const std::vector<std::shared_ptr<Tensor>>&)>;
using FunctionForwardPostHook = std::function<void(class Function*, const std::vector<std::shared_ptr<Tensor>>&,
                                                    const std::vector<std::shared_ptr<Tensor>>&)>;
using FunctionBackwardPreHook = std::function<void(class Function*, const std::vector<std::shared_ptr<Tensor>>&)>;
using FunctionBackwardPostHook = std::function<void(class Function*, const std::vector<std::shared_ptr<Tensor>>&,
                                                     const std::vector<std::shared_ptr<Tensor>>&)>;

class Function : public std::enable_shared_from_this<Function> {
public:
    static constexpr char kUndefinedType[] = "Undefined";

    Function() : type_(kUndefinedType) {}
    explicit Function(const std::string &type) : type_(type) {}

    virtual ~Function() = default;

    virtual std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) = 0;
    virtual void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                              const std::vector<std::shared_ptr<Tensor>> &output_tensors) {}
    virtual std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) = 0;

    std::vector<std::shared_ptr<Tensor>> Apply(const std::vector<std::shared_ptr<Tensor>> &input_tensors);
    virtual void BackwardPartial(const std::shared_ptr<Tensor> &grad_output, int idx);

    void IncreaseDependenciesNumber();

    std::shared_ptr<HookHandle> RegisterForwardPreHook(FunctionForwardPreHook hook);
    std::shared_ptr<HookHandle> RegisterForwardPostHook(FunctionForwardPostHook hook);
    std::shared_ptr<HookHandle> RegisterBackwardPreHook(FunctionBackwardPreHook hook);
    std::shared_ptr<HookHandle> RegisterBackwardPostHook(FunctionBackwardPostHook hook);

    const std::string& type() const { return type_; }

protected:
    std::vector<std::shared_ptr<Tensor>> saved_tensors_;

private:
    std::vector<std::pair<std::shared_ptr<Function>, int>> next_functions_;
    int dependencies_number_ = 0;
    int dependencies_reached_ = 0;
    int grad_outputs_reached_ = 0;
    std::vector<std::shared_ptr<Tensor>> grad_outputs_;
    const std::string type_ = kUndefinedType;
    std::vector<FunctionForwardPreHook> forward_pre_hooks_;
    std::vector<FunctionForwardPostHook> forward_post_hooks_;
    std::vector<FunctionBackwardPreHook> backward_pre_hooks_;
    std::vector<FunctionBackwardPostHook> backward_post_hooks_;
    bool precision_check_registered_ = false;
};
} // namespace infini_train::autograd
