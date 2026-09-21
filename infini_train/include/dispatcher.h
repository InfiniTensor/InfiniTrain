#pragma once

#include <map>
#include <string>
#include <type_traits>
#include <utility>

#include "glog/logging.h"

#include "infini_train/include/common/common.h"
#include "infini_train/include/device.h"
#include "infini_train/include/utils/nvtx.h"
#ifdef PROFILE_MODE
#include "infini_train/include/profiler.h"
#endif

namespace infini_train {

class KernelFunction {
public:
    // The operator name is stored per instance rather than read from a
    // thread_local context: SetProfileContext() only records the outermost
    // dispatch, so nested dispatches would overwrite each other's name.
    template <typename FuncT>
    explicit KernelFunction(std::string name, FuncT &&func)
        : name_(std::move(name)), func_ptr_(reinterpret_cast<void *>(func)) {}

    // TODO(dcj): support auto-deduction of return type and parameter types
    template <typename RetT, class... ArgsT> RetT Call(ArgsT... args) const {
#ifdef NVTX_MODE
        utils::NvtxRange nvtx_range(name_.c_str());
#endif
#ifdef PROFILE_MODE
        const auto &ctx = GetProfileContext();
        Profiler::Instance().StartRecord(ctx.name, ctx.device);
#endif

        using FuncT = RetT (*)(ArgsT...);
        auto fn = reinterpret_cast<FuncT>(func_ptr_);

        if constexpr (std::is_void_v<RetT>) {
            fn(std::forward<ArgsT>(args)...);
#ifdef PROFILE_MODE
            Profiler::Instance().EndRecord(ctx.name, ctx.device);
#endif
            return;
        } else {
            RetT ret = fn(std::forward<ArgsT>(args)...);
#ifdef PROFILE_MODE
            Profiler::Instance().EndRecord(ctx.name, ctx.device);
#endif
            return ret;
        }
    }

private:
    std::string name_;
    void *func_ptr_ = nullptr;
};

class Dispatcher {
public:
    using KeyT = std::pair<Device::DeviceType, std::string>;

    static Dispatcher &Instance() {
        static Dispatcher instance;
        return instance;
    }

    bool HasKernel(const KeyT &key) const { return key_to_kernel_map_.contains(key); }

    const KernelFunction &GetKernel(KeyT key) const {
        CHECK(key_to_kernel_map_.contains(key))
            << "Kernel not found: " << key.second << " on device: " << static_cast<int>(key.first);
#ifdef PROFILE_MODE
        SetProfileContext(key.second, key.first);
#endif
        return key_to_kernel_map_.at(key);
    }

    template <typename FuncT> void Register(const KeyT &key, FuncT &&kernel) {
        CHECK(!key_to_kernel_map_.contains(key))
            << "Kernel already registered: " << key.second << " on device: " << static_cast<int>(key.first);
        key_to_kernel_map_.try_emplace(key, key.second, kernel);
    }

    template <typename RetT, class... ArgsT> RetT Call(KeyT key, ArgsT... args) const {
        auto kernel = this->GetKernel(key);
#ifdef PROFILE_MODE
        SetProfileContext(key.second, key.first);
#endif
        return kernel.Call<RetT>(std::forward<ArgsT>(args)...);
    }

private:
    std::map<KeyT, KernelFunction> key_to_kernel_map_;
};

} // namespace infini_train

#define REGISTER_KERNEL(device, kernel_name, kernel_func)                                                              \
    [[maybe_unused]] static const bool CAT(infini_train_kernel_, CAT(kernel_name, CAT(_registered_, __COUNTER__)))     \
        = []() {                                                                                                       \
              infini_train::Dispatcher::Instance().Register({device, #kernel_name}, kernel_func);                      \
              return true;                                                                                             \
          }();
