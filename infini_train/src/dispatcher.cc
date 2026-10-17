#include "infini_train/include/dispatcher.h"

#ifdef PROFILE_MODE
#include "infini_train/include/profiler.h"
#endif

namespace infini_train {

Dispatcher &Dispatcher::Instance() {
    static Dispatcher instance;
    return instance;
}

bool Dispatcher::HasKernel(const KeyT &key) const { return key_to_kernel_map_.contains(key); }

const KernelFunction &Dispatcher::GetKernel(KeyT key) const {
    CHECK(key_to_kernel_map_.contains(key))
        << "Kernel not found: " << key.second << " on device: " << static_cast<int>(key.first);
#ifdef PROFILE_MODE
    SetProfileContext(key.second, key.first);
#endif
    return key_to_kernel_map_.at(key);
}

} // namespace infini_train
