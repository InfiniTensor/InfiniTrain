#include <memory>

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {

std::shared_ptr<Tensor> ReLUForward(const std::shared_ptr<Tensor> &input) {
    CHECK(input->Dtype() == DataType::kFLOAT32);
    auto output = std::make_shared<Tensor>(input->Dims(), DataType::kFLOAT32);
    const auto *input_data = static_cast<const float *>(input->DataPtr());
    auto *output_data = static_cast<float *>(output->DataPtr());
    for (size_t index = 0; index < input->NumElements(); ++index) {
        output_data[index] = input_data[index] > 0.0f ? input_data[index] : 0.0f;
    }
    return output;
}

std::shared_ptr<Tensor> ReLUBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &grad_output) {
    CHECK(input->Dtype() == DataType::kFLOAT32);
    CHECK(grad_output->Dtype() == DataType::kFLOAT32);
    CHECK(input->GetDevice() == grad_output->GetDevice());
    CHECK(input->Dims() == grad_output->Dims());
    auto grad_input = std::make_shared<Tensor>(input->Dims(), DataType::kFLOAT32);
    const auto *input_data = static_cast<const float *>(input->DataPtr());
    const auto *grad_output_data = static_cast<const float *>(grad_output->DataPtr());
    auto *grad_input_data = static_cast<float *>(grad_input->DataPtr());
    for (size_t index = 0; index < input->NumElements(); ++index) {
        grad_input_data[index] = input_data[index] > 0.0f ? grad_output_data[index] : 0.0f;
    }
    return grad_input;
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_RELU_KERNEL(kernel_name)                                                                          \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_RELU_KERNEL(ReLUForward)
REGISTER_CPU_RELU_KERNEL(ReLUBackward)

#undef REGISTER_CPU_RELU_KERNEL
