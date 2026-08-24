#include <cmath>
#include <memory>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
namespace {
// Constants of the NewGELU tanh approximation: beta = sqrt(2/pi), kappa = 0.044715.
constexpr float kGeluBeta = 0.7978845608028654f;
constexpr float kGeluKappa = 0.044715f;
} // namespace

std::shared_ptr<Tensor> NewGELUForward(const std::shared_ptr<Tensor> &input) {
    auto output = std::make_shared<Tensor>(input->Dims(), DataType::kFLOAT32);
    const float *input_ptr = static_cast<const float *>(input->DataPtr());
    float *output_ptr = static_cast<float *>(output->DataPtr());

    const int64_t numel = input->NumElements();
    for (int64_t idx = 0; idx < numel; ++idx) {
        const float x = input_ptr[idx];
        const float inner = kGeluBeta * (x + kGeluKappa * x * x * x);
        output_ptr[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
    return output;
}

std::shared_ptr<Tensor> NewGELUBackward(const std::shared_ptr<Tensor> &grad_output,
                                        const std::shared_ptr<Tensor> &input) {
    auto grad_input = std::make_shared<Tensor>(grad_output->Dims(), DataType::kFLOAT32);
    const float *grad_output_ptr = static_cast<const float *>(grad_output->DataPtr());
    const float *input_ptr = static_cast<const float *>(input->DataPtr());
    float *grad_input_ptr = static_cast<float *>(grad_input->DataPtr());

    const int64_t numel = grad_output->NumElements();
    for (int64_t idx = 0; idx < numel; ++idx) {
        const float x = input_ptr[idx];
        const float x_sq = x * x;
        const float inner = kGeluBeta * (x + kGeluKappa * x_sq * x);
        const float tanh_inner = tanhf(inner);
        const float left_derivative = 0.5f * (1.0f + tanh_inner);
        const float right_derivative
            = 0.5f * x * (1.0f - tanh_inner * tanh_inner) * kGeluBeta * (1.0f + 3.0f * kGeluKappa * x_sq);
        grad_input_ptr[idx] = grad_output_ptr[idx] * (left_derivative + right_derivative);
    }
    return grad_input;
}
} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_GELU_KERNEL(kernel_name)                                                                          \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_GELU_KERNEL(NewGELUForward)
REGISTER_CPU_GELU_KERNEL(NewGELUBackward)

#undef REGISTER_CPU_GELU_KERNEL
