#pragma once

#include <cstddef>
#include <utility>

#include "glog/logging.h"

#include "infini_train/include/device.h"

namespace infini_train {

// ============================================================
// DispatchStub — 设备无关的函数指针分发（仿 PyTorch DispatchStub）
// ============================================================
// 每个 stub 按 DeviceType 存一组函数指针，调用时按设备查表分发。
// 新增后端只需在 DeviceType 枚举加一项，kCount 自动适配。
//
// 使用：
//   1. DECLARE_DISPATCH(fn_type, name)  — 声明 extern 全局 stub
//   2. DEFINE_DISPATCH(name)            — 定义全局 stub 实例
//   3. REGISTER_DISPATCH(name, dt, fn)  — 后端注册函数指针
//   4. name(device_type, args...)       — 调用，自动分发
//
template <typename FnPtr>
class DispatchStub {
public:
    using FnType = FnPtr;

    static constexpr size_t kNumDevices = static_cast<size_t>(Device::DeviceType::kCount);

    void register_kernel(Device::DeviceType dt, FnPtr fn) {
        table_[static_cast<size_t>(dt)] = fn;
    }

    template <typename... Args>
    auto operator()(Device::DeviceType dt, Args&&... args) const {
        auto idx = static_cast<size_t>(dt);
        CHECK(idx < kNumDevices) << "Invalid device type " << static_cast<int>(dt);
        CHECK(table_[idx] != nullptr) << "Dispatch kernel not registered for device type "
                                      << static_cast<int>(dt);
        return (*table_[idx])(std::forward<Args>(args)...);
    }

private:
    FnPtr table_[kNumDevices] = {};
};

// ---- 宏 ----

// 声明：放在头文件里，告诉其他编译单元"这个 stub 存在"
#define DECLARE_DISPATCH(fn_type, name) \
    extern DispatchStub<fn_type> name

// 定义：放在一个 .cpp 里，分配实际存储空间
#define DEFINE_DISPATCH(name) \
    DispatchStub<decltype(name)::FnType> name

// 注册：放在后端 .cpp/.cu 里，静态初始化时把函数填进表
#define INFINI_TRAIN_CONCAT_IMPL(x, y) x##y
#define INFINI_TRAIN_CONCAT(x, y)      INFINI_TRAIN_CONCAT_IMPL(x, y)

#define REGISTER_DISPATCH(name, device_type, fn)                                      \
    static const bool INFINI_TRAIN_CONCAT(name##_registered_, __COUNTER__) = []() {   \
        (name).register_kernel((device_type), (fn));                                  \
        return true;                                                                  \
    }();

} // namespace infini_train
