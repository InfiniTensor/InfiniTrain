#include <cmath>
#include <memory>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
std::shared_ptr<Tensor> SwiGLUForward(const std::shared_ptr<Tensor> &input) {
    CHECK(input->Dtype() == DataType::kFLOAT32);
    CHECK(input->IsContiguous());
    auto output_dims = input->Dims();
    CHECK_GT(output_dims.size(), 0);
    CHECK_EQ(output_dims.back() % 2, 0);
    const int64_t hidden = output_dims.back() / 2;
    CHECK_GT(hidden, 0);
    output_dims.back() = hidden;

    auto output = std::make_shared<Tensor>(output_dims, input->Dtype(), input->GetDevice());
    const float *input_ptr = static_cast<const float *>(input->DataPtr());
    float *output_ptr = static_cast<float *>(output->DataPtr());
    const int64_t rows = output->NumElements() / hidden;
    for (int64_t row = 0; row < rows; ++row) {
        const int64_t input_base = row * 2 * hidden;
        const int64_t output_base = row * hidden;
        for (int64_t col = 0; col < hidden; ++col) {
            const float up = input_ptr[input_base + col];
            const float gate = input_ptr[input_base + hidden + col];
            output_ptr[output_base + col] = up * gate / (1.0f + std::exp(-gate));
        }
    }
    return output;
}

std::shared_ptr<Tensor> SwiGLUBackward(const std::shared_ptr<Tensor> &input,
                                       const std::shared_ptr<Tensor> &grad_output) {
    CHECK(input->Dtype() == DataType::kFLOAT32);
    CHECK(grad_output->Dtype() == input->Dtype());
    CHECK(input->IsContiguous());
    CHECK(grad_output->IsContiguous());
    CHECK_GT(input->Dims().size(), 0);
    const int64_t hidden = input->Dims().back() / 2;
    CHECK_GT(hidden, 0);
    CHECK_EQ(grad_output->NumElements() * 2, input->NumElements());

    auto grad_input = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());
    const float *input_ptr = static_cast<const float *>(input->DataPtr());
    const float *grad_output_ptr = static_cast<const float *>(grad_output->DataPtr());
    float *grad_input_ptr = static_cast<float *>(grad_input->DataPtr());
    const int64_t rows = grad_output->NumElements() / hidden;
    for (int64_t row = 0; row < rows; ++row) {
        const int64_t input_base = row * 2 * hidden;
        const int64_t output_base = row * hidden;
        for (int64_t col = 0; col < hidden; ++col) {
            const float up = input_ptr[input_base + col];
            const float gate = input_ptr[input_base + hidden + col];
            const float grad = grad_output_ptr[output_base + col];
            const float sigmoid = 1.0f / (1.0f + std::exp(-gate));
            grad_input_ptr[input_base + col] = grad * gate * sigmoid;
            grad_input_ptr[input_base + hidden + col] = grad * up * sigmoid * (1.0f + gate * (1.0f - sigmoid));
        }
    }
    return grad_input;
}
} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_SWIGLU_KERNEL(kernel_name)                                                                        \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_SWIGLU_KERNEL(SwiGLUForward)
REGISTER_CPU_SWIGLU_KERNEL(SwiGLUBackward)

#undef REGISTER_CPU_SWIGLU_KERNEL
