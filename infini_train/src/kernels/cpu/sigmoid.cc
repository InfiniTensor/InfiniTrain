#include <cmath>
#include <memory>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
std::shared_ptr<Tensor> SigmoidForward(const std::shared_ptr<Tensor> &input) {
    auto output = std::make_shared<Tensor>(input->Dims(), DataType::kFLOAT32);
    const float *input_ptr = static_cast<const float *>(input->DataPtr());
    float *output_ptr = static_cast<float *>(output->DataPtr());

    const int64_t numel = input->NumElements();
    for (int64_t idx = 0; idx < numel; ++idx) { output_ptr[idx] = 1.0f / (1.0f + exp(-input_ptr[idx])); }

    return output;
}

std::shared_ptr<Tensor> SigmoidBackward(const std::shared_ptr<Tensor> &output,
                                        const std::shared_ptr<Tensor> &grad_output) {
    auto grad_input = std::make_shared<Tensor>(output->Dims(), DataType::kFLOAT32);
    const float *output_ptr = static_cast<const float *>(output->DataPtr());
    const float *grad_output_ptr = static_cast<const float *>(grad_output->DataPtr());
    float *grad_input_ptr = static_cast<float *>(grad_input->DataPtr());

    const int64_t numel = output->NumElements();
    for (int64_t idx = 0; idx < numel; ++idx) {
        const float y = output_ptr[idx];
        grad_input_ptr[idx] = grad_output_ptr[idx] * y * (1.0f - y);
    }
    return grad_input;
}
} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_SIGMOID_KERNEL(kernel_name)                                                                       \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_SIGMOID_KERNEL(SigmoidForward)
REGISTER_CPU_SIGMOID_KERNEL(SigmoidBackward)

#undef REGISTER_CPU_SIGMOID_KERNEL
