#pragma once

#include "infini_train/include/common/hook.h"
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <vector>

namespace infini_train {
namespace nn {
class Module;
}

namespace utils {

// Global Module Hook Registry
// Manages hooks that need to be applied to all modules
class GlobalModuleHookRegistry {
public:
    using ModuleHookRegistrar = std::function<void(nn::Module *)>;

    static GlobalModuleHookRegistry &Instance();

    // Register a hook registrar, which will be called for all modules on their first forward pass
    // Returns a HookHandle that can be used to remove the hook
    std::unique_ptr<HookHandle> RegisterHook(ModuleHookRegistrar registrar);

    // Apply all registered hooks to the specified module (called by Module::operator())
    void ApplyHooks(nn::Module *module);

private:
    GlobalModuleHookRegistry() = default;

    std::vector<ModuleHookRegistrar> registrars_;
    std::unordered_set<nn::Module *> applied_modules_;
    mutable std::mutex mutex_;
};

} // namespace utils
} // namespace infini_train
