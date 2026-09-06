#include <memory>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
std::shared_ptr<Tensor> ReluForward(const std::shared_ptr<Tensor> &input) {
    auto output = std::make_shared<Tensor>(input->Dims(), DataType::kFLOAT32);
    const float *input_ptr = static_cast<const float *>(input->DataPtr());
    float *output_ptr = static_cast<float *>(output->DataPtr());

    const int64_t numel = input->NumElements();
    for (int64_t idx = 0; idx < numel; ++idx) { output_ptr[idx] = input_ptr[idx] > 0.0f ? input_ptr[idx] : 0.0f; }

    return output;
}

std::shared_ptr<Tensor> ReluBackward(const std::shared_ptr<Tensor> &output,
                                     const std::shared_ptr<Tensor> &grad_output) {
    auto grad_input = std::make_shared<Tensor>(output->Dims(), DataType::kFLOAT32);
    const float *output_ptr = static_cast<const float *>(output->DataPtr());
    const float *grad_output_ptr = static_cast<const float *>(grad_output->DataPtr());
    float *grad_input_ptr = static_cast<float *>(grad_input->DataPtr());

    const int64_t numel = output->NumElements();
    for (int64_t idx = 0; idx < numel; ++idx) {
        grad_input_ptr[idx] = output_ptr[idx] > 0.0f ? grad_output_ptr[idx] : 0.0f;
    }
    return grad_input;
}
} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_RELU_KERNEL(kernel_name)                                                                          \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_RELU_KERNEL(ReluForward)
REGISTER_CPU_RELU_KERNEL(ReluBackward)

#undef REGISTER_CPU_RELU_KERNEL
