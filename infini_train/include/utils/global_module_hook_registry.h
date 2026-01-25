#pragma once

#include <functional>
#include <mutex>
#include <unordered_set>
#include <vector>

namespace infini_train {
namespace nn {
class Module;
}

namespace utils {

// 全局 Module Hook 注册器
// 管理需要应用到所有 module 的 hooks
class GlobalModuleHookRegistry {
public:
    using ModuleHookRegistrar = std::function<void(nn::Module *)>;

    static GlobalModuleHookRegistry &Instance();

    // 注册一个 hook registrar，之后所有 module 在首次 forward 时会调用它
    void RegisterHook(ModuleHookRegistrar registrar);

    // 为指定 module 应用所有已注册的 hooks（由 Module::operator() 调用）
    void ApplyHooks(nn::Module *module);

private:
    GlobalModuleHookRegistry() = default;

    std::vector<ModuleHookRegistrar> registrars_;
    std::unordered_set<nn::Module *> applied_modules_;
    mutable std::mutex mutex_;
};

} // namespace utils
} // namespace infini_train
