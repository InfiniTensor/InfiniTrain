#include "infini_train/include/core/kernel_provider/infiniops_registry.h"

#include <iostream>
#include <mutex>
#include <set>
#include <string>

namespace infini_train::kernel_provider {

namespace {

const std::set<std::string> kEnabledKernelWhitelist = {
    "Gemm",
    "AddForward",
};

} // namespace

bool InfiniOpsEnabled() {
#ifdef USE_INFINIOPS
    return true;
#else
    return false;
#endif
}

bool InfiniOpsEnabled(const KeyT &key) {
    return InfiniOpsEnabled() && key.first == Device::DeviceType::kCUDA && kEnabledKernelWhitelist.contains(key.second);
}

const KernelFunction *LookupInfiniOpsKernel(const KeyT &key) {
    const auto *kernel = InfiniOpsRegistry::Instance().Lookup(key.second);

    static std::mutex log_mutex;
    static std::set<std::string> logged_use;
    static std::set<std::string> logged_fallback;

    const auto log_key = std::to_string(static_cast<int>(key.first)) + ":" + key.second;
    std::lock_guard<std::mutex> lock(log_mutex);
    if (kernel != nullptr) {
        if (logged_use.insert(log_key).second) {
            std::cout << "[InfiniOps] use " << key.second << " on device " << static_cast<int>(key.first) << std::endl;
        }
    } else if (logged_fallback.insert(log_key).second) {
        std::cout << "[InfiniOps] fallback " << key.second << " on device " << static_cast<int>(key.first) << std::endl;
    }

    return kernel;
}

} // namespace infini_train::kernel_provider
