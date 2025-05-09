#include <cmath>
#include <cstdint>
#include <memory>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
std::shared_ptr<Tensor> SoftmaxForward(const std::shared_ptr<Tensor> &input, int64_t dim) {
    dim = dim < 0 ? input->Dims().size() + dim : dim;
    auto output = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());

    const float *input_data = static_cast<const float *>(input->DataPtr());
    float *output_data = static_cast<float *>(output->DataPtr());

    const auto &dims = input->Dims();
    int outer = 1;
    int axis = dims[dim];
    int inner = 1;

    for (int i = 0; i < dim; ++i) { outer *= dims[i]; }
    for (int i = dim + 1; i < dims.size(); ++i) { inner *= dims[i]; }

    for (int o = 0; o < outer; ++o) {
        for (int i = 0; i < inner; ++i) {
            int offset = o * axis * inner + i;

            // find max value in the axis dimension
            float max_val = input_data[offset];
            for (int j = 1; j < axis; ++j) { max_val = std::max(max_val, input_data[offset + j * inner]); }

            // compute sum_exp
            float sum_exp = 0.0f;
            for (int j = 0; j < axis; ++j) {
                output_data[offset + j * inner] = std::exp(input_data[offset + j * inner] - max_val);
                sum_exp += output_data[offset + j * inner];
            }

            // normalize
            for (int j = 0; j < axis; ++j) { output_data[offset + j * inner] /= sum_exp; }
        }
    }
    return output;
}

std::shared_ptr<Tensor> SoftmaxBackward(const std::shared_ptr<Tensor> &grad_output,
                                        const std::shared_ptr<Tensor> &output, int64_t dim) {
    dim = dim < 0 ? output->Dims().size() + dim : dim;
    auto grad_input = std::make_shared<Tensor>(output->Dims(), output->Dtype(), output->GetDevice());
    const float *grad_output_data = static_cast<const float *>(grad_output->DataPtr());
    const float *output_data = static_cast<const float *>(output->DataPtr());
    float *grad_input_data = static_cast<float *>(grad_input->DataPtr());

    const auto &dims = output->Dims();
    int outer = 1;
    int inner = 1;
    int axis = dims[dim];

    for (int i = 0; i < dim; ++i) { outer *= dims[i]; }
    for (int i = dim + 1; i < dims.size(); ++i) { inner *= dims[i]; }

    for (int o = 0; o < outer; ++o) {
        for (int i = 0; i < inner; ++i) {
            int offset = o * axis * inner + i;

            // dot product: y * dy
            float dot = 0.0f;
            for (int j = 0; j < axis; ++j) {
                dot += output_data[offset + j * inner] * grad_output_data[offset + j * inner];
            }
            for (int j = 0; j < axis; ++j) {
                grad_input_data[offset + j * inner]
                    = output_data[offset + j * inner] * (grad_output_data[offset + j * inner] - dot);
            }
        }
    }
    return grad_input;
}
} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_SOFTMAX_KERNEL(kernel_name)                                                                       \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_SOFTMAX_KERNEL(SoftmaxForward)
REGISTER_CPU_SOFTMAX_KERNEL(SoftmaxBackward)

#undef REGISTER_CPU_SOFTMAX_KERNEL
