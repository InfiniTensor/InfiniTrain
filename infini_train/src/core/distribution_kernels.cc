/// distribution_kernels.cpp
///
/// 桥接层：Generator Handle → CPUGeneratorImpl* → 分布函子
///
/// - get_generator_or_default: 解析 Generator → 具体 Impl 指针
/// - *_cpu_kernel:             CPU 后端实现（加锁 + 遍历 + 分布函子）
/// - REGISTER_DISPATCH:        将内核注册到 dispatch 表
///
/// 仿 PyTorch aten/src/ATen/native/cpu/DistributionKernels.cpp（253 行）

#include <limits>
#include <mutex>

#include "infini_train/include/core/runtime/distribution_kernels.h"
#include "infini_train/include/core/runtime/distribution_stubs.h"
#include "infini_train/include/core/runtime/distributions_helper.h"
#include "infini_train/include/tensor.h"
#include "infini_train/src/core/runtime/cpu/cpu_dispatch.h"
#include "infini_train/src/core/runtime/cpu/cpu_generator_impl.h"

namespace infini_train {

DEFINE_DISPATCH(uniform_stub);
DEFINE_DISPATCH(normal_stub);

namespace {

void check_distribution_tensor(const Tensor &tensor) {
    CHECK(IsFloatingPointDType(tensor.Dtype()))
        << "Uniform and Normal initialization support floating-point tensors only";
}

struct DistributionBounds {
    double lowest;
    double max;
};

DistributionBounds distribution_bounds(DataType dtype) {
    switch (dtype) {
    case DataType::kFLOAT16: {
        const double max = static_cast<float>(FP16(static_cast<uint16_t>(0x7bff), FP16::from_bits()));
        return {-max, max};
    }
    case DataType::kBFLOAT16: {
        const double max = static_cast<float>(BF16(static_cast<uint16_t>(0x7f7f), BF16::from_bits()));
        return {-max, max};
    }
    case DataType::kFLOAT32:
        return {-std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
    case DataType::kFLOAT64:
        return {-std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
    default:
        LOG(FATAL) << "Unsupported distribution dtype: " << kDataTypeToDesc.at(dtype);
        return {};
    }
}

void check_uniform_parameters(const Tensor &tensor, double from, double to) {
    const auto bounds = distribution_bounds(tensor.Dtype());
    CHECK_GE(from, bounds.lowest) << "uniform expects from to be within the range of "
                                  << kDataTypeToDesc.at(tensor.Dtype());
    CHECK_LE(from, bounds.max) << "uniform expects from to be within the range of "
                               << kDataTypeToDesc.at(tensor.Dtype());
    CHECK_GE(to, bounds.lowest) << "uniform expects to to be within the range of "
                                << kDataTypeToDesc.at(tensor.Dtype());
    CHECK_LE(to, bounds.max) << "uniform expects to to be within the range of "
                             << kDataTypeToDesc.at(tensor.Dtype());
    CHECK_LE(from, to) << "uniform expects a [from, to) range, but found from=" << from << " > to=" << to;
    CHECK_LE(to - from, bounds.max) << "uniform expects to - from to fit in "
                                    << kDataTypeToDesc.at(tensor.Dtype());
}

void check_normal_parameters(double std) {
    CHECK_GE(std, 0.0) << "normal expects std >= 0.0, but found std=" << std;
}

// ---- CPU 内核实现 ----

template <typename storage_t, typename random_t>
void uniform_cpu_kernel_impl(Tensor &tensor, double from, double to,
                             core::cpu::CPUGeneratorImpl *generator) {
    auto *buf = static_cast<storage_t *>(tensor.DataPtr());
    uniform_real_distribution<random_t> dist(static_cast<random_t>(from), static_cast<random_t>(to));
    const storage_t from_value = static_cast<storage_t>(from);
    const random_t to_value = static_cast<random_t>(static_cast<storage_t>(to));
    for (int64_t i = 0; i < tensor.NumElements(); ++i) {
        const storage_t value = static_cast<storage_t>(dist(generator));
        buf[i] = static_cast<random_t>(value) == to_value ? from_value : value;
    }
}

template <typename storage_t, typename random_t>
void normal_cpu_kernel_impl(Tensor &tensor, double mean, double std,
                            core::cpu::CPUGeneratorImpl *generator) {
    auto *buf = static_cast<storage_t *>(tensor.DataPtr());
    normal_distribution<random_t> dist(static_cast<random_t>(mean), static_cast<random_t>(std));
    for (int64_t i = 0; i < tensor.NumElements(); ++i) {
        buf[i] = static_cast<storage_t>(dist(generator));
    }
}

void uniform_cpu_kernel(Tensor &tensor, double from, double to,
                         const std::optional<Generator> &gen) {
    CHECK(tensor.GetDevice().IsCPU());
    auto *cpu_gen = get_generator_or_default<core::cpu::CPUGeneratorImpl>(
        gen, core::cpu::getDefaultCPUGenerator());

    std::lock_guard<std::mutex> lock(cpu_gen->mutex_);
    core::cpu::DispatchCpuFunc<DataType::kFLOAT16, DataType::kBFLOAT16, DataType::kFLOAT32, DataType::kFLOAT64>(
        tensor.Dtype(),
        [&]<typename storage_t>() {
            using random_t = std::conditional_t<std::is_same_v<storage_t, double>, double, float>;
            uniform_cpu_kernel_impl<storage_t, random_t>(tensor, from, to, cpu_gen);
        },
        "CPU uniform");
}

void normal_cpu_kernel(Tensor &tensor, double mean, double std,
                        const std::optional<Generator> &gen) {
    CHECK(tensor.GetDevice().IsCPU());
    auto *cpu_gen = get_generator_or_default<core::cpu::CPUGeneratorImpl>(
        gen, core::cpu::getDefaultCPUGenerator());

    std::lock_guard<std::mutex> lock(cpu_gen->mutex_);
    core::cpu::DispatchCpuFunc<DataType::kFLOAT16, DataType::kBFLOAT16, DataType::kFLOAT32, DataType::kFLOAT64>(
        tensor.Dtype(),
        [&]<typename storage_t>() {
            using random_t = std::conditional_t<std::is_same_v<storage_t, double>, double, float>;
            normal_cpu_kernel_impl<storage_t, random_t>(tensor, mean, std, cpu_gen);
        },
        "CPU normal");
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
    check_uniform_parameters(tensor, from, to);
    uniform_stub(tensor.GetDevice().type(), tensor, from, to, gen);
}

void normal_kernel(Tensor &tensor, double mean, double std,
                   const std::optional<Generator> &gen) {
    check_distribution_tensor(tensor);
    check_normal_parameters(std);
    normal_stub(tensor.GetDevice().type(), tensor, mean, std, gen);
}

} // namespace infini_train
