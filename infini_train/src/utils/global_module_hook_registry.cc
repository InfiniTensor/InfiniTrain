#include "infini_train/include/utils/global_module_hook_registry.h"

namespace infini_train::utils {

GlobalModuleHookRegistry &GlobalModuleHookRegistry::Instance() {
    static GlobalModuleHookRegistry instance;
    return instance;
}

std::unique_ptr<HookHandle> GlobalModuleHookRegistry::RegisterModuleForwardPreHook(ModuleForwardPreHook hook) {
    std::lock_guard<std::mutex> lock(mutex_);
    module_forward_pre_hooks_.push_back(std::move(hook));
    return std::make_unique<HookHandleImpl<ModuleForwardPreHook>>(&module_forward_pre_hooks_,
                                                                  module_forward_pre_hooks_.size() - 1);
}

std::unique_ptr<HookHandle> GlobalModuleHookRegistry::RegisterModuleForwardHook(ModuleForwardHook hook) {
    std::lock_guard<std::mutex> lock(mutex_);
    module_forward_hooks_.push_back(std::move(hook));
    return std::make_unique<HookHandleImpl<ModuleForwardHook>>(&module_forward_hooks_,
                                                               module_forward_hooks_.size() - 1);
}

std::unique_ptr<HookHandle> GlobalModuleHookRegistry::RegisterModuleFullBackwardHook(ModuleFullBackwardHook hook) {
    std::lock_guard<std::mutex> lock(mutex_);
    module_full_backward_hooks_.push_back(std::move(hook));
    return std::make_unique<HookHandleImpl<ModuleFullBackwardHook>>(&module_full_backward_hooks_,
                                                                    module_full_backward_hooks_.size() - 1);
}

void GlobalModuleHookRegistry::CallModuleForwardPreHooks(nn::Module *module,
                                                         const std::vector<std::shared_ptr<Tensor>> &inputs) {
    std::vector<ModuleForwardPreHook> snapshot;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        snapshot = module_forward_pre_hooks_;
    }
    for (const auto &hook : snapshot) {
        if (hook) {
            hook(module, inputs);
        }
    }
}

void GlobalModuleHookRegistry::CallModuleForwardHooks(nn::Module *module,
                                                      const std::vector<std::shared_ptr<Tensor>> &inputs,
                                                      const std::vector<std::shared_ptr<Tensor>> &outputs) {
    std::vector<ModuleForwardHook> snapshot;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        snapshot = module_forward_hooks_;
    }
    for (const auto &hook : snapshot) {
        if (hook) {
            hook(module, inputs, outputs);
        }
    }
}

void GlobalModuleHookRegistry::CallModuleFullBackwardHooks(nn::Module *module,
                                                           const std::vector<std::shared_ptr<Tensor>> &grad_outputs,
                                                           const std::vector<std::shared_ptr<Tensor>> &grad_inputs) {
    std::vector<ModuleFullBackwardHook> snapshot;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        snapshot = module_full_backward_hooks_;
    }
    for (const auto &hook : snapshot) {
        if (hook) {
            hook(module, grad_outputs, grad_inputs);
        }
    }
}

bool GlobalModuleHookRegistry::HasModuleBackwardHooks() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return !module_full_backward_hooks_.empty();
}

} // namespace infini_train::utils
