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

#include <iostream>
namespace infini_train::kernels::cpu {

std::shared_ptr<Tensor> Dropout(const std::shared_ptr<Tensor> &input, float p, core::CPUGeneratorImpl *impl) {
    CHECK(impl != nullptr);
    auto device = input->GetDevice();
    float keep_prob = 1.0f - p;
    float scale = 1.0f / keep_prob;
    auto mask = std::make_shared<Tensor>(input->Dims(), DataType::kFLOAT32, device);
    auto *input_ptr = static_cast<float *>(input->DataPtr());
    int64_t numel = mask->NumElements();
    std::bernoulli_distribution dist(keep_prob);
    {
        std::lock_guard<std::mutex> lock(impl->mutex_);
        auto &engine = impl->Engine();
        for (int64_t i = 0; i < numel; i++) { input_ptr[i] = dist(engine) ? input_ptr[i] * scale : 0.0f; }
    }
    return input;
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_DROPOUT_KERNEL(kernel_name)                                                                       \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_DROPOUT_KERNEL(Dropout)
#undef REGISTER_CPU_DROPOUT_KERNEL