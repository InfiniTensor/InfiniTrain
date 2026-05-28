#pragma once

#include <map>
#include <string>
#include <utility>

#include "glog/logging.h"

#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"

namespace infini_train::kernel_provider {

using KeyT = std::pair<Device::DeviceType, std::string>;

class InfiniOpsRegistry {
public:
    static InfiniOpsRegistry &Instance() {
        static InfiniOpsRegistry instance;
        return instance;
    }

    const KernelFunction *Lookup(const std::string &kernel_name) const {
        auto it = name_to_kernel_map_.find(kernel_name);
        return it == name_to_kernel_map_.end() ? nullptr : &it->second;
    }

    template <typename FuncT> void Register(const std::string &kernel_name, FuncT &&kernel) {
        CHECK(!name_to_kernel_map_.contains(kernel_name)) << "InfiniOps kernel already registered: " << kernel_name;
        name_to_kernel_map_.emplace(kernel_name, kernel);
    }

private:
    std::map<std::string, KernelFunction> name_to_kernel_map_;
};

// Bridge functions used by Dispatcher::GetKernel. Implemented in
// infiniops_registry.cc; declared here for users that already include
// the full registry header (e.g. unit tests).
bool InfiniOpsEnabled();
bool InfiniOpsEnabled(const KeyT &key);
const KernelFunction *LookupInfiniOpsKernel(const KeyT &key);

} // namespace infini_train::kernel_provider

#define REGISTER_INFINIOPS_KERNEL(kernel_name, kernel_func)                                                            \
    static const bool _##kernel_name##_infiniops_registered##__COUNTER__ = []() {                                      \
        infini_train::kernel_provider::InfiniOpsRegistry::Instance().Register(#kernel_name, kernel_func);              \
        return true;                                                                                                   \
    }();
