#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <type_traits>

#include "glog/logging.h"

#include "infini_train/include/common/cpu/distributions_helper.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/generator.h"
#include "infini_train/include/tensor.h"
#include "infini_train/src/core/runtime/cpu/cpu_dispatch.h"
#include "infini_train/src/core/runtime/cpu/cpu_generator_impl.h"

namespace infini_train::kernels::cpu {
namespace {

template <typename storage_t, typename random_t>
void UniformImpl(Tensor &tensor, double from, double to, core::cpu::CPUGeneratorImpl *generator) {
    auto *buf = static_cast<storage_t *>(tensor.DataPtr());
    common::cpu::uniform_real_distribution<random_t> dist(static_cast<random_t>(from), static_cast<random_t>(to));
    const storage_t from_value = static_cast<storage_t>(from);
    const random_t to_value = static_cast<random_t>(static_cast<storage_t>(to));
    for (int64_t i = 0; i < tensor.NumElements(); ++i) {
        const storage_t value = static_cast<storage_t>(dist(generator));
        // [from, to) is half-open: a sample landing exactly on `to` is mapped back to `from`.
        buf[i] = static_cast<random_t>(value) == to_value ? from_value : value;
    }
}

template <typename storage_t, typename random_t>
void NormalImpl(Tensor &tensor, double mean, double std, core::cpu::CPUGeneratorImpl *generator) {
    auto *buf = static_cast<storage_t *>(tensor.DataPtr());
    common::cpu::normal_distribution<random_t> dist(static_cast<random_t>(mean), static_cast<random_t>(std));
    for (int64_t i = 0; i < tensor.NumElements(); ++i) { buf[i] = static_cast<storage_t>(dist(generator)); }
}

} // namespace

void Uniform(const std::shared_ptr<Tensor> tensor, double from, double to, const std::optional<Generator> gen) {
    CHECK(tensor->GetDevice().IsCPU());
    auto *cpu_generator
        = get_generator_or_default<core::cpu::CPUGeneratorImpl>(gen, core::cpu::getDefaultCPUGenerator());

    std::lock_guard<std::mutex> lock(cpu_generator->mutex_);
    core::cpu::DispatchCpuFunc<DataType::kFLOAT16, DataType::kBFLOAT16, DataType::kFLOAT32, DataType::kFLOAT64>(
        tensor->Dtype(),
        [&]<typename storage_t>() {
            using random_t = std::conditional_t<std::is_same_v<storage_t, double>, double, float>;
            UniformImpl<storage_t, random_t>(*tensor, from, to, cpu_generator);
        },
        "CPU uniform");
}

void Normal(const std::shared_ptr<Tensor> tensor, double mean, double std, const std::optional<Generator> gen) {
    CHECK(tensor->GetDevice().IsCPU());
    auto *cpu_generator
        = get_generator_or_default<core::cpu::CPUGeneratorImpl>(gen, core::cpu::getDefaultCPUGenerator());

    std::lock_guard<std::mutex> lock(cpu_generator->mutex_);
    core::cpu::DispatchCpuFunc<DataType::kFLOAT16, DataType::kBFLOAT16, DataType::kFLOAT32, DataType::kFLOAT64>(
        tensor->Dtype(),
        [&]<typename storage_t>() {
            using random_t = std::conditional_t<std::is_same_v<storage_t, double>, double, float>;
            NormalImpl<storage_t, random_t>(*tensor, mean, std, cpu_generator);
        },
        "CPU normal");
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_DISTRIBUTION_KERNEL(kernel_name)                                                                  \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_DISTRIBUTION_KERNEL(Uniform)
REGISTER_CPU_DISTRIBUTION_KERNEL(Normal)

#undef REGISTER_CPU_DISTRIBUTION_KERNEL
