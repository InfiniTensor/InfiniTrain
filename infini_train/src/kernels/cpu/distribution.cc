#include <cmath>
#include <memory>
#include <mutex>
#include <optional>
#include <random>

#include "infini_train/include/core/cpu_generator.h"
#include "infini_train/include/core/generator.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
namespace {
template <typename Dist> void FillKernel(float *data, int64_t numel, core::CPUGeneratorImpl *impl, Dist dist) {
    std::lock_guard<std::mutex> lock(impl->mutex_);
    auto &engine = impl->Engine();
    for (int64_t i = 0; i < numel; ++i) { data[i] = dist(engine); }
}
} // namespace
// Convenience wrappers matching PyTorch's named distribution templates.

// uniform_impl_: fills buffer with values from [from, to).
void Uniform(std::shared_ptr<Tensor> &tensor, float from, float to, core::CPUGeneratorImpl *generator) {
    const int64_t numel = tensor->NumElements();
    float *buffer = static_cast<float *>(tensor->DataPtr());
    FillKernel(buffer, numel, generator, std::uniform_real_distribution<float>(from, to));
}

// normal_impl_: fills buffer with values from N(mean, std).
void Normal(std::shared_ptr<Tensor> &tensor, float mean, float std, core::CPUGeneratorImpl *generator) {
    float *buffer = static_cast<float *>(tensor->DataPtr());
    const int64_t numel = tensor->NumElements();
    FillKernel(buffer, numel, generator, std::normal_distribution<float>(mean, std));
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_DISTRIBUTION_KERNEL(kernel_name)                                                                  \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_DISTRIBUTION_KERNEL(Uniform)
REGISTER_CPU_DISTRIBUTION_KERNEL(Normal)

#undef REGISTER_CPU_DISTRIBUTION_KERNEL