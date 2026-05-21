#include <cstddef>
#include <memory>
#include <mutex>
#include <random>

#include "glog/logging.h"

#include "infini_train/include/core/generator/cpu_generator_impl.h"
#include "infini_train/include/core/generator/generator_impl.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {

void NormalRandom(std::shared_ptr<Tensor> tensor, float mean, float std, core::GeneratorImpl *impl) {
    CHECK(impl != nullptr) << "NormalRandom: GeneratorImpl is null";
    CHECK_EQ(static_cast<int>(tensor->Dtype()), static_cast<int>(DataType::kFLOAT32))
        << "NormalRandom currently only supports FP32";

    auto *cpu_impl = static_cast<core::CPUGeneratorImpl *>(impl);
    std::lock_guard<std::mutex> lk(cpu_impl->mutex());
    auto &eng = cpu_impl->engine();
    std::normal_distribution<float> dist(mean, std);

    auto *data = static_cast<float *>(tensor->DataPtr());
    const size_t n = static_cast<size_t>(tensor->NumElements());
    for (size_t i = 0; i < n; ++i) { data[i] = dist(eng); }
}

} // namespace infini_train::kernels::cpu

REGISTER_KERNEL(infini_train::Device::DeviceType::kCPU, NormalRandom, infini_train::kernels::cpu::NormalRandom)
