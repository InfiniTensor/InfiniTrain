#pragma once

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace infini_train {
class Tensor;
class HookHandle;
template <typename HookType> class HookHandleImpl;
} // namespace infini_train

namespace infini_train::autograd {

class FunctionCtx {
public:
    using SavedTensorPackHook = std::function<std::shared_ptr<void>(const std::shared_ptr<Tensor> &)>;
    using SavedTensorUnpackHook = std::function<std::shared_ptr<Tensor>(const std::shared_ptr<void> &)>;

    struct SavedTensorHooks {
        SavedTensorPackHook pack;
        SavedTensorUnpackHook unpack;
    };

    class SavedTensorHooksGuard {
    public:
        explicit SavedTensorHooksGuard(SavedTensorHooks hooks);
        ~SavedTensorHooksGuard();

        SavedTensorHooksGuard(const SavedTensorHooksGuard &) = delete;
        SavedTensorHooksGuard &operator=(const SavedTensorHooksGuard &) = delete;

    private:
        size_t depth_ = 0;
    };

    FunctionCtx();

    void SaveForBackward(const std::vector<std::shared_ptr<Tensor>> &tensors);

    std::vector<std::shared_ptr<Tensor>> GetSavedTensors() const;

    void MarkNonDifferentiable(const std::vector<std::shared_ptr<Tensor>> &outputs);

    const std::vector<bool> &needs_input_grad() const;

private:
    struct SavedTensorEntry {
        std::shared_ptr<Tensor> tensor;
        std::shared_ptr<void> hook_state;
        SavedTensorUnpackHook unpack;
    };

    friend class Function;

    void set_needs_input_grad(std::vector<bool> needs_input_grad);

    void SaveVariables(const std::vector<std::shared_ptr<Tensor>> &outputs);
    void ReleaseVariables();

    bool IsNonDifferentiable(const std::shared_ptr<Tensor> &output) const;

    std::vector<std::shared_ptr<Tensor>> to_save_;
    std::vector<SavedTensorEntry> saved_tensor_entries_;
    std::vector<bool> needs_input_grad_;
    std::vector<Tensor *> non_differentiable_;
};

class Function : public std::enable_shared_from_this<Function> {
public:
    template <typename HookType> using FunctionHookHandleImpl = infini_train::HookHandleImpl<HookType>;

    using FunctionPreHook = std::function<void(Function *, const std::vector<std::shared_ptr<Tensor>> &)>;
    using FunctionPostHook = std::function<void(Function *, const std::vector<std::shared_ptr<Tensor>> &,
                                                const std::vector<std::shared_ptr<Tensor>> &)>;

    static constexpr char kUndefinedType[] = "Undefined";

    Function();
    explicit Function(const std::string &type);

    virtual ~Function();

    virtual std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) = 0;
    virtual void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                              const std::vector<std::shared_ptr<Tensor>> &output_tensors);
    virtual std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) = 0;

    std::vector<std::shared_ptr<Tensor>> Apply(const std::vector<std::shared_ptr<Tensor>> &input_tensors);
    virtual void BackwardPartial(std::shared_ptr<Tensor> grad_output, int idx);

    void IncreaseDependenciesNumber();

    std::shared_ptr<infini_train::HookHandle> RegisterForwardPreHook(FunctionPreHook hook);
    std::shared_ptr<infini_train::HookHandle> RegisterForwardPostHook(FunctionPostHook hook);
    std::shared_ptr<infini_train::HookHandle> RegisterBackwardPreHook(FunctionPreHook hook);
    std::shared_ptr<infini_train::HookHandle> RegisterBackwardPostHook(FunctionPostHook hook);

    const std::string &type() const;

protected:
    FunctionCtx ctx_;

private:
    std::vector<std::pair<std::shared_ptr<Function>, int>> next_functions_;
    int dependencies_number_ = 0;
    int dependencies_reached_ = 0;
    int grad_outputs_reached_ = 0;
    std::vector<std::shared_ptr<Tensor>> grad_outputs_;
    const std::string type_ = kUndefinedType;
    std::vector<FunctionPreHook> forward_pre_hooks_;
    std::vector<FunctionPostHook> forward_post_hooks_;
    std::vector<FunctionPreHook> backward_pre_hooks_;
    std::vector<FunctionPostHook> backward_post_hooks_;
    bool precision_check_registered_ = false;
};
} // namespace infini_train::autograd
