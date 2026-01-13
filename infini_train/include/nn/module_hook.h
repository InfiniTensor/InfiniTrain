#pragma once

#include <functional>
#include <memory>
#include <vector>

namespace infini_train {
class Tensor;

namespace nn {
class Module;

// Forward pre-hook: called before forward pass
// Args: (module, input_tensors)
using ForwardPreHook = std::function<void(Module*, const std::vector<std::shared_ptr<Tensor>>&)>;

// Forward post-hook: called after forward pass
// Args: (module, input_tensors, output_tensors)
using ForwardPostHook = std::function<void(Module*, const std::vector<std::shared_ptr<Tensor>>&,
                                           const std::vector<std::shared_ptr<Tensor>>&)>;

// Backward pre-hook: called before backward pass
// Args: (module, grad_output)
using BackwardPreHook = std::function<void(Module*, const std::vector<std::shared_ptr<Tensor>>&)>;

// Backward post-hook: called after backward pass
// Args: (module, grad_input, grad_output)
using BackwardPostHook = std::function<void(Module*, const std::vector<std::shared_ptr<Tensor>>&,
                                            const std::vector<std::shared_ptr<Tensor>>&)>;

class ModuleHookHandle {
public:
    virtual ~ModuleHookHandle() = default;
    virtual void Remove() = 0;
};

template <typename HookType>
class ModuleHookHandleImpl : public ModuleHookHandle {
public:
    ModuleHookHandleImpl(std::vector<HookType>* hooks, size_t id) : hooks_(hooks), id_(id) {}

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

} // namespace nn
} // namespace infini_train
