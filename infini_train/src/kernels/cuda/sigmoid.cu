#include "infini_train/include/kernels/cuda/sigmoid.h"

#include <cmath>
#include <cstddef>
#include <memory>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {
__global__ void SigmoidForwardKernel(const float *input_ptr, float *output_ptr, size_t num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        output_ptr[idx] = 1.0f / (1.0f + exp(-input_ptr[idx]));
    }
}

std::shared_ptr<Tensor> SigmoidForward(const std::shared_ptr<Tensor> &input) {
    size_t num_elements = input->NumElements();

    auto output = std::make_shared<Tensor>(input->Dims(), DataType::kFLOAT32, Device(DeviceType::kCUDA, 0));

    const float *input_ptr = static_cast<const float *>(input->DataPtr());
    float *output_ptr = static_cast<float *>(output->DataPtr());

    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    SigmoidForwardKernel<<<num_blocks, threads_per_block>>>(input_ptr, output_ptr, num_elements);

    return output;
}

__global__ void SigmoidBackwardKernel(const float *output_ptr, const float *grad_output_ptr, float *grad_input_ptr,
                                      size_t num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        grad_input_ptr[idx] = grad_output_ptr[idx] * output_ptr[idx] * (1.0f - output_ptr[idx]);
    }
}

std::shared_ptr<Tensor> SigmoidBackward(const std::shared_ptr<Tensor> &output,
                                        const std::shared_ptr<Tensor> &grad_output) {
    size_t num_elements = output->NumElements();

    auto grad_input = std::make_shared<Tensor>(output->Dims(), DataType::kFLOAT32, Device(DeviceType::kCUDA, 0));

    const float *output_ptr = static_cast<const float *>(output->DataPtr());
    const float *grad_output_ptr = static_cast<float *>(grad_output->DataPtr());
    float *grad_input_ptr = static_cast<float *>(grad_input->DataPtr());

    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    SigmoidBackwardKernel<<<num_blocks, threads_per_block>>>(output_ptr, grad_output_ptr, grad_input_ptr, num_elements);

    return grad_input;
}
} // namespace infini_train::kernels::cuda
