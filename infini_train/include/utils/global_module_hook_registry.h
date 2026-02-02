#pragma once

#include "infini_train/include/common/hook.h"
#include "infini_train/include/tensor.h"
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

namespace infini_train {
namespace nn {
class Module;
}

namespace utils {

// Global Module Hook Registry
// Global hooks that are executed on every forward/backward pass
class GlobalModuleHookRegistry {
public:
    using ModuleForwardPreHook = std::function<void(nn::Module *, const std::vector<std::shared_ptr<Tensor>> &inputs)>;

    using ModuleForwardHook = std::function<void(nn::Module *, const std::vector<std::shared_ptr<Tensor>> &inputs,
                                                 const std::vector<std::shared_ptr<Tensor>> &outputs)>;

    using ModuleFullBackwardHook
        = std::function<void(nn::Module *, const std::vector<std::shared_ptr<Tensor>> &grad_outputs,
                             const std::vector<std::shared_ptr<Tensor>> &grad_inputs)>;

    static GlobalModuleHookRegistry &Instance();

    // PyTorch-style registration: RegisterModule* prefix
    std::unique_ptr<HookHandle> RegisterModuleForwardPreHook(ModuleForwardPreHook hook);
    std::unique_ptr<HookHandle> RegisterModuleForwardHook(ModuleForwardHook hook);
    std::unique_ptr<HookHandle> RegisterModuleFullBackwardHook(ModuleFullBackwardHook hook);

    // Call hooks (called by Module::operator())
    void CallModuleForwardPreHooks(nn::Module *module, const std::vector<std::shared_ptr<Tensor>> &inputs);
    void CallModuleForwardHooks(nn::Module *module, const std::vector<std::shared_ptr<Tensor>> &inputs,
                                const std::vector<std::shared_ptr<Tensor>> &outputs);
    void CallModuleFullBackwardHooks(nn::Module *module, const std::vector<std::shared_ptr<Tensor>> &grad_outputs,
                                     const std::vector<std::shared_ptr<Tensor>> &grad_inputs);
    bool HasModuleBackwardHooks() const;

private:
    GlobalModuleHookRegistry() = default;

    std::vector<ModuleForwardPreHook> module_forward_pre_hooks_;
    std::vector<ModuleForwardHook> module_forward_hooks_;
    std::vector<ModuleFullBackwardHook> module_full_backward_hooks_;
    mutable std::mutex mutex_;
};

} // namespace utils
} // namespace infini_train
