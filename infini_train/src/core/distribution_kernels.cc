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

#include "infini_train/include/core/runtime/distribution_kernels.h"
#include "infini_train/include/core/runtime/distribution_stubs.h"
#include "infini_train/include/core/runtime/distributions_helper.h"
#include "infini_train/include/tensor.h"
#include "infini_train/src/core/runtime/cpu/cpu_generator_impl.h"

namespace infini_train {

DEFINE_DISPATCH(uniform_stub);
DEFINE_DISPATCH(normal_stub);

namespace {

void check_distribution_tensor(const Tensor &tensor) {
    CHECK_EQ(static_cast<int>(tensor.Dtype()), static_cast<int>(DataType::kFLOAT32))
        << "Uniform and Normal initialization currently support FLOAT32 tensors only";
}

// ---- CPU 内核实现 ----

void uniform_cpu_kernel(Tensor &tensor, double from, double to,
                         const std::optional<Generator> &gen) {
    CHECK(tensor.GetDevice().IsCPU());
    auto *cpu_gen = get_generator_or_default<core::cpu::CPUGeneratorImpl>(
        gen, core::cpu::getDefaultCPUGenerator());

    std::lock_guard<std::mutex> lock(cpu_gen->mutex_);
    auto *buf = static_cast<float *>(tensor.DataPtr());
    uniform_real_distribution<float> dist(static_cast<float>(from), static_cast<float>(to));
    for (int64_t i = 0; i < tensor.NumElements(); ++i) {
        buf[i] = dist(cpu_gen);
    }
}

void normal_cpu_kernel(Tensor &tensor, double mean, double std,
                        const std::optional<Generator> &gen) {
    CHECK(tensor.GetDevice().IsCPU());
    auto *cpu_gen = get_generator_or_default<core::cpu::CPUGeneratorImpl>(
        gen, core::cpu::getDefaultCPUGenerator());

    std::lock_guard<std::mutex> lock(cpu_gen->mutex_);
    auto *buf = static_cast<float *>(tensor.DataPtr());
    normal_distribution<float> dist(static_cast<float>(mean), static_cast<float>(std));
    for (int64_t i = 0; i < tensor.NumElements(); ++i) {
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

void uniform_kernel(Tensor &tensor, double from, double to,
                    const std::optional<Generator> &gen) {
    check_distribution_tensor(tensor);
    uniform_stub(tensor.GetDevice().type(), tensor, from, to, gen);
}

void normal_kernel(Tensor &tensor, double mean, double std,
                   const std::optional<Generator> &gen) {
    check_distribution_tensor(tensor);
    normal_stub(tensor.GetDevice().type(), tensor, mean, std, gen);
}

} // namespace infini_train
