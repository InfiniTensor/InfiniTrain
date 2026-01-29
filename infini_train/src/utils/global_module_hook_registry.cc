#include "infini_train/include/utils/global_module_hook_registry.h"

namespace infini_train::utils {

GlobalModuleHookRegistry &GlobalModuleHookRegistry::Instance() {
    static GlobalModuleHookRegistry instance;
    return instance;
}

std::unique_ptr<HookHandle> GlobalModuleHookRegistry::RegisterHook(ModuleHookRegistrar registrar) {
    std::lock_guard<std::mutex> lock(mutex_);
    registrars_.push_back(std::move(registrar));
    return std::make_unique<HookHandleImpl<ModuleHookRegistrar>>(&registrars_, registrars_.size() - 1);
}

void GlobalModuleHookRegistry::ApplyHooks(nn::Module *module) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (applied_modules_.contains(module)) {
        return;
    }
    for (const auto &registrar : registrars_) {
        if (registrar) {
            registrar(module);
        }
    }
    applied_modules_.insert(module);
}

} // namespace infini_train::utils
