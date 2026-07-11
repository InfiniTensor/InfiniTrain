/// distribution_kernels.cpp
///
/// 桥接层：Generator Handle → CPUGeneratorImpl* → 分布函子
///
/// - get_generator_or_default: 解析 Generator → 具体 Impl 指针
/// - *_cpu_kernel:             CPU 后端实现（加锁 + 遍历 + 分布函子）
/// - REGISTER_DISPATCH:        将内核注册到 dispatch 表
///
/// 仿 PyTorch aten/src/ATen/native/cpu/DistributionKernels.cpp（253 行）

#include <mutex>
#include <optional>

#include "infini_train/include/core/runtime/distribution_kernels.h"
#include "infini_train/include/core/runtime/distribution_stubs.h"
#include "infini_train/include/core/runtime/distributions_helper.h"
#include "infini_train/src/core/runtime/cpu/cpu_generator_impl.h"

namespace infini_train {
namespace {

// ============================================================
// get_generator_or_default — Generator Handle → 具体 Impl 指针
// ============================================================
// 仿 PyTorch at::native::get_generator_or_default
//
// 如果调用方显式传了 Generator，就用那个；
// 否则 fallback 到当前设备的默认 Generator。
//
template <typename T>
T *get_generator_or_default(const std::optional<Generator> &gen, const Generator &default_gen) {
    if (gen.has_value() && gen->defined()) {
        return gen->get<T>();
    }
    return default_gen.get<T>();
}

// ---- DEFINE_DISPATCH：为 stub 分配全局存储 ----
// 注意：必须在 namespace infini_train 层级，与 DECLARE_DISPATCH 声明的 extern 匹配。
// 不能放在匿名 namespace 里，否则会创建两个不同的变量。

} // namespace

DEFINE_DISPATCH(uniform_stub);
DEFINE_DISPATCH(normal_stub);

namespace {

// ---- CPU 内核实现 ----

void uniform_cpu_kernel(void *data, int64_t n, double from, double to,
                         const std::optional<Generator> &gen) {
    auto *cpu_gen = get_generator_or_default<core::cpu::CPUGeneratorImpl>(
        gen, core::cpu::getDefaultCPUGenerator());

    std::lock_guard<std::mutex> lock(cpu_gen->mutex_);
    auto *buf = static_cast<float *>(data);
    uniform_real_distribution<float> dist(static_cast<float>(from), static_cast<float>(to));
    for (int64_t i = 0; i < n; ++i) {
        buf[i] = dist(cpu_gen);
    }
}

void normal_cpu_kernel(void *data, int64_t n, double mean, double std,
                        const std::optional<Generator> &gen) {
    auto *cpu_gen = get_generator_or_default<core::cpu::CPUGeneratorImpl>(
        gen, core::cpu::getDefaultCPUGenerator());

    std::lock_guard<std::mutex> lock(cpu_gen->mutex_);
    auto *buf = static_cast<float *>(data);
    normal_distribution<float> dist(static_cast<float>(mean), static_cast<float>(std));
    for (int64_t i = 0; i < n; ++i) {
        buf[i] = dist(cpu_gen);
    }
}

} // namespace

// ---- REGISTER_DISPATCH：启动时自动把内核填进 dispatch 表 ----

REGISTER_DISPATCH(uniform_stub, Device::DeviceType::kCPU, &uniform_cpu_kernel);
REGISTER_DISPATCH(normal_stub, Device::DeviceType::kCPU, &normal_cpu_kernel);

// ============================================================
// 公共 API：包装层（仿 PyTorch Distributions.cpp 中的 uniform_/normal_ 入口函数）
// ============================================================
// init.cc 等上层代码调用这些函数，不直接碰 uniform_stub()。

void uniform_kernel(void *data, int64_t n, double from, double to,
                    Device::DeviceType device_type, const std::optional<Generator> &gen) {
    uniform_stub(device_type, data, n, from, to, gen);
}

void normal_kernel(void *data, int64_t n, double mean, double std,
                   Device::DeviceType device_type, const std::optional<Generator> &gen) {
    normal_stub(device_type, data, n, mean, std, gen);
}

} // namespace infini_train
