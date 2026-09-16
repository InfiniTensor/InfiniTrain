#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <type_traits>

#include "glog/logging.h"

#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/generator.h"
#include "infini_train/include/generator_impl.h"
#include "infini_train/include/tensor.h"
#include "infini_train/src/core/runtime/cpu/cpu_dispatch.h"
#include "infini_train/src/core/runtime/cpu/cpu_generator_impl.h"
#include "infini_train/src/kernels/cpu/distributions_helper.h"

namespace infini_train::kernels::cpu {
namespace {

constexpr int kMaxUniformAttempts = 3;

template <typename StorageT, typename RandomT>
void UniformImpl(Tensor &tensor, double from, double to, core::cpu::CPUGeneratorImpl *generator) {
    auto *buf = static_cast<StorageT *>(tensor.DataPtr());
    common::cpu::UniformRealDistribution<RandomT> dist(static_cast<RandomT>(from), static_cast<RandomT>(to));
    const StorageT from_value = static_cast<StorageT>(from);
    if (from == to) {
        for (int64_t i = 0; i < tensor.NumElements(); ++i) { buf[i] = from_value; }
        return;
    }
    for (int64_t i = 0; i < tensor.NumElements(); ++i) {
        StorageT value;
        int attempt = 0;
        do {
            value = static_cast<StorageT>(dist(generator));
            ++attempt;
        } while (attempt < kMaxUniformAttempts
                 && (static_cast<double>(value) < from || static_cast<double>(value) >= to));
        // Bounded retries may still leave out-of-range values in very narrow intervals.
        buf[i] = static_cast<double>(value) == to ? from_value : value;
    }
}

template <typename StorageT, typename RandomT>
void NormalImpl(Tensor &tensor, double mean, double std, core::cpu::CPUGeneratorImpl *generator) {
    auto *buf = static_cast<StorageT *>(tensor.DataPtr());
    common::cpu::NormalDistribution<RandomT> dist(static_cast<RandomT>(mean), static_cast<RandomT>(std));
    for (int64_t i = 0; i < tensor.NumElements(); ++i) { buf[i] = static_cast<StorageT>(dist(generator)); }
}

} // namespace

void Uniform(const std::shared_ptr<Tensor> tensor, double from, double to, const std::optional<Generator> gen) {
    CHECK(tensor->GetDevice().IsCPU());
    auto &cpu_generator = GetGeneratorOrDefault<core::cpu::CPUGeneratorImpl>(gen, core::cpu::GetDefaultCpuGenerator(),
                                                                             tensor->GetDevice());

    std::lock_guard<std::mutex> lock(cpu_generator.mutex_);
    core::cpu::DispatchCpuFunc<DataType::kFLOAT16, DataType::kBFLOAT16, DataType::kFLOAT32, DataType::kFLOAT64>(
        tensor->Dtype(),
        [&]<typename StorageT>() {
            using RandomT = std::conditional_t<std::is_same_v<StorageT, double>, double, float>;
            UniformImpl<StorageT, RandomT>(*tensor, from, to, &cpu_generator);
        },
        "CPU uniform");
}

void Normal(const std::shared_ptr<Tensor> tensor, double mean, double std, const std::optional<Generator> gen) {
    CHECK(tensor->GetDevice().IsCPU());
    auto &cpu_generator = GetGeneratorOrDefault<core::cpu::CPUGeneratorImpl>(gen, core::cpu::GetDefaultCpuGenerator(),
                                                                             tensor->GetDevice());

    std::lock_guard<std::mutex> lock(cpu_generator.mutex_);
    core::cpu::DispatchCpuFunc<DataType::kFLOAT16, DataType::kBFLOAT16, DataType::kFLOAT32, DataType::kFLOAT64>(
        tensor->Dtype(),
        [&]<typename StorageT>() {
            using RandomT = std::conditional_t<std::is_same_v<StorageT, double>, double, float>;
            NormalImpl<StorageT, RandomT>(*tensor, mean, std, &cpu_generator);
        },
        "CPU normal");
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_DISTRIBUTION_KERNEL(kernel_name)                                                                  \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_DISTRIBUTION_KERNEL(Uniform)
REGISTER_CPU_DISTRIBUTION_KERNEL(Normal)

#undef REGISTER_CPU_DISTRIBUTION_KERNEL
