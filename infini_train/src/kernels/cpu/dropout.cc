#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <ostream>
#include <tuple>
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

template <typename StorageT, typename RandomT>
void DropoutForwardImpl(Tensor &output, Tensor &mask, const Tensor &input, double p,
                        core::cpu::CPUGeneratorImpl *generator) {
    auto *output_data = static_cast<StorageT *>(output.DataPtr());
    auto *mask_data = static_cast<uint8_t *>(mask.DataPtr());
    const auto *input_data = static_cast<const StorageT *>(input.DataPtr());
    const int64_t n = input.NumElements();

    if (p == 0.0) {
        for (int64_t index = 0; index < n; ++index) {
            mask_data[index] = 1;
            output_data[index] = input_data[index];
        }
        return;
    }
    if (p == 1.0) {
        for (int64_t index = 0; index < n; ++index) {
            mask_data[index] = 0;
            output_data[index] = static_cast<StorageT>(0.0);
        }
        return;
    }

    const RandomT scale = static_cast<RandomT>(1.0 / (1.0 - p));
    common::cpu::UniformRealDistribution<RandomT> distribution(static_cast<RandomT>(0), static_cast<RandomT>(1));
    for (int64_t index = 0; index < n; ++index) {
        const bool keep = distribution(generator) >= static_cast<RandomT>(p);
        mask_data[index] = keep ? 1 : 0;
        output_data[index] = keep ? static_cast<StorageT>(static_cast<RandomT>(input_data[index]) * scale)
                                  : static_cast<StorageT>(0.0);
    }
}

template <typename StorageT, typename RandomT>
void DropoutBackwardImpl(Tensor &grad_input, const Tensor &grad_output, const Tensor &mask, double p) {
    auto *grad_input_data = static_cast<StorageT *>(grad_input.DataPtr());
    const auto *grad_output_data = static_cast<const StorageT *>(grad_output.DataPtr());
    const auto *mask_data = static_cast<const uint8_t *>(mask.DataPtr());
    const RandomT scale = p == 1.0 ? static_cast<RandomT>(0) : static_cast<RandomT>(1.0 / (1.0 - p));

    for (int64_t index = 0; index < grad_output.NumElements(); ++index) {
        grad_input_data[index] = mask_data[index]
                                   ? static_cast<StorageT>(static_cast<RandomT>(grad_output_data[index]) * scale)
                                   : static_cast<StorageT>(0.0);
    }
}

} // namespace

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
DropoutForward(const std::shared_ptr<Tensor> input, double p, const std::optional<Generator> gen) {
    CHECK(input->GetDevice().IsCPU());
    CHECK(IsFloatingPointDType(input->Dtype())) << "Dropout supports floating-point tensors only";
    CHECK_GE(p, 0.0) << "dropout probability has to be between 0 and 1, but got " << p;
    CHECK_LE(p, 1.0) << "dropout probability has to be between 0 and 1, but got " << p;

    auto output = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());
    auto mask = std::make_shared<Tensor>(input->Dims(), DataType::kUINT8, input->GetDevice());

    if (p == 0.0 || p == 1.0) {
        core::cpu::DispatchCpuFunc<DataType::kFLOAT16, DataType::kBFLOAT16, DataType::kFLOAT32, DataType::kFLOAT64>(
            input->Dtype(),
            [&]<typename StorageT>() {
                using RandomT = std::conditional_t<std::is_same_v<StorageT, double>, double, float>;
                DropoutForwardImpl<StorageT, RandomT>(*output, *mask, *input, p, nullptr);
            },
            "CPU dropout forward");
        return {output, mask};
    }

    auto &cpu_generator = GetGeneratorOrDefault<core::cpu::CPUGeneratorImpl>(gen, core::cpu::GetDefaultCpuGenerator(),
                                                                             input->GetDevice());
    std::lock_guard<std::mutex> lock(cpu_generator.mutex_);
    core::cpu::DispatchCpuFunc<DataType::kFLOAT16, DataType::kBFLOAT16, DataType::kFLOAT32, DataType::kFLOAT64>(
        input->Dtype(),
        [&]<typename StorageT>() {
            using RandomT = std::conditional_t<std::is_same_v<StorageT, double>, double, float>;
            DropoutForwardImpl<StorageT, RandomT>(*output, *mask, *input, p, &cpu_generator);
        },
        "CPU dropout forward");
    return {output, mask};
}

std::shared_ptr<Tensor> DropoutBackward(const std::shared_ptr<Tensor> grad_output, const std::shared_ptr<Tensor> mask,
                                        double p) {
    CHECK(grad_output->GetDevice().IsCPU());
    auto grad_input = std::make_shared<Tensor>(grad_output->Dims(), grad_output->Dtype(), grad_output->GetDevice());

    core::cpu::DispatchCpuFunc<DataType::kFLOAT16, DataType::kBFLOAT16, DataType::kFLOAT32, DataType::kFLOAT64>(
        grad_output->Dtype(),
        [&]<typename StorageT>() {
            using RandomT = std::conditional_t<std::is_same_v<StorageT, double>, double, float>;
            DropoutBackwardImpl<StorageT, RandomT>(*grad_input, *grad_output, *mask, p);
        },
        "CPU dropout backward");
    return grad_input;
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_DROPOUT_KERNEL(kernel_name)                                                                       \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_DROPOUT_KERNEL(DropoutForward)
REGISTER_CPU_DROPOUT_KERNEL(DropoutBackward)

#undef REGISTER_CPU_DROPOUT_KERNEL
