#include <mutex>
#include <type_traits>

#include "infini_train/include/core/runtime/dropout_stubs.h"
#include "infini_train/include/core/runtime/distributions_helper.h"
#include "infini_train/include/tensor.h"
#include "infini_train/src/core/runtime/cpu/cpu_dispatch.h"
#include "infini_train/src/core/runtime/cpu/cpu_generator_impl.h"

namespace infini_train::kernels::cpu {
namespace {

template <typename storage_t, typename random_t>
void dropout_forward_cpu_impl(Tensor &output, Tensor &mask, const Tensor &input, double p,
                              core::cpu::CPUGeneratorImpl *generator) {
    auto *output_data = static_cast<storage_t *>(output.DataPtr());
    auto *mask_data = static_cast<uint8_t *>(mask.DataPtr());
    const auto *input_data = static_cast<const storage_t *>(input.DataPtr());
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
            output_data[index] = static_cast<storage_t>(0.0);
        }
        return;
    }

    const random_t scale = static_cast<random_t>(1.0 / (1.0 - p));
    uniform_real_distribution<random_t> distribution(static_cast<random_t>(0), static_cast<random_t>(1));
    for (int64_t index = 0; index < n; ++index) {
        const bool keep = distribution(generator) >= static_cast<random_t>(p);
        mask_data[index] = keep ? 1 : 0;
        output_data[index] = keep
            ? static_cast<storage_t>(static_cast<random_t>(input_data[index]) * scale)
            : static_cast<storage_t>(0.0);
    }
}

template <typename storage_t, typename random_t>
void dropout_backward_cpu_impl(Tensor &grad_input, const Tensor &grad_output, const Tensor &mask, double p) {
    auto *grad_input_data = static_cast<storage_t *>(grad_input.DataPtr());
    const auto *grad_output_data = static_cast<const storage_t *>(grad_output.DataPtr());
    const auto *mask_data = static_cast<const uint8_t *>(mask.DataPtr());
    const random_t scale = p == 1.0 ? static_cast<random_t>(0) : static_cast<random_t>(1.0 / (1.0 - p));

    for (int64_t index = 0; index < grad_output.NumElements(); ++index) {
        grad_input_data[index] = mask_data[index]
            ? static_cast<storage_t>(static_cast<random_t>(grad_output_data[index]) * scale)
            : static_cast<storage_t>(0.0);
    }
}

void dropout_forward_cpu(Tensor &output, Tensor &mask, const Tensor &input, double p,
                         const std::optional<Generator> &generator) {
    CHECK(input.GetDevice().IsCPU());

    core::cpu::DispatchCpuFunc<DataType::kFLOAT16, DataType::kBFLOAT16, DataType::kFLOAT32, DataType::kFLOAT64>(
        input.Dtype(),
        [&]<typename storage_t>() {
            using random_t = std::conditional_t<std::is_same_v<storage_t, double>, double, float>;
            if (p == 0.0 || p == 1.0) {
                dropout_forward_cpu_impl<storage_t, random_t>(output, mask, input, p, nullptr);
                return;
            }
            auto *cpu_generator = get_generator_or_default<core::cpu::CPUGeneratorImpl>(
                generator, core::cpu::getDefaultCPUGenerator());
            std::lock_guard<std::mutex> lock(cpu_generator->mutex_);
            dropout_forward_cpu_impl<storage_t, random_t>(output, mask, input, p, cpu_generator);
        },
        "CPU dropout forward");
}

void dropout_backward_cpu(Tensor &grad_input, const Tensor &grad_output, const Tensor &mask, double p) {
    CHECK(grad_output.GetDevice().IsCPU());
    core::cpu::DispatchCpuFunc<DataType::kFLOAT16, DataType::kBFLOAT16, DataType::kFLOAT32, DataType::kFLOAT64>(
        grad_output.Dtype(),
        [&]<typename storage_t>() {
            using random_t = std::conditional_t<std::is_same_v<storage_t, double>, double, float>;
            dropout_backward_cpu_impl<storage_t, random_t>(grad_input, grad_output, mask, p);
        },
        "CPU dropout backward");
}

} // namespace

using infini_train::dropout_backward_stub;
using infini_train::dropout_forward_stub;

REGISTER_DISPATCH(dropout_forward_stub, Device::DeviceType::kCPU, &dropout_forward_cpu);
REGISTER_DISPATCH(dropout_backward_stub, Device::DeviceType::kCPU, &dropout_backward_cpu);

} // namespace infini_train::kernels::cpu
