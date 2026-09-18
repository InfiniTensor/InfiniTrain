#include <cstddef>
#include <memory>

#include "glog/logging.h"

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/cuda/cuda_dispatch.h"
#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

namespace infini_train::kernels::cuda {
namespace {
__global__ void ReLUForwardKernel(float *output, const float *input, size_t num_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float x = input[idx];
        output[idx] = x > 0.0f ? x : 0.0f;
    }
}

__global__ void ReLUBackwardKernel(float *grad_input, const float *input, const float *grad_output,
                                   size_t num_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        grad_input[idx] = input[idx] > 0.0f ? grad_output[idx] : 0.0f;
    }
}

void LaunchForward(const std::shared_ptr<Tensor> &output, const std::shared_ptr<Tensor> &input) {
    const size_t num_elements = static_cast<size_t>(output->NumElements());
    if (num_elements == 0) {
        return;
    }
    const int threads_per_block = 256;
    const int num_blocks = static_cast<int>((num_elements + threads_per_block - 1) / threads_per_block);
    auto device = output->GetDevice();
    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();
    ReLUForwardKernel<<<num_blocks, threads_per_block, 0, cuda_stream>>>(
        static_cast<float *>(output->DataPtr()), static_cast<const float *>(input->DataPtr()), num_elements);
}

void LaunchBackward(const std::shared_ptr<Tensor> &grad_input, const std::shared_ptr<Tensor> &input,
                    const std::shared_ptr<Tensor> &grad_output) {
    const size_t num_elements = static_cast<size_t>(grad_input->NumElements());
    if (num_elements == 0) {
        return;
    }
    const int threads_per_block = 256;
    const int num_blocks = static_cast<int>((num_elements + threads_per_block - 1) / threads_per_block);
    auto device = grad_input->GetDevice();
    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();
    ReLUBackwardKernel<<<num_blocks, threads_per_block, 0, cuda_stream>>>(
        static_cast<float *>(grad_input->DataPtr()), static_cast<const float *>(input->DataPtr()),
        static_cast<const float *>(grad_output->DataPtr()), num_elements);
}
} // namespace

std::shared_ptr<Tensor> ReLUForward(const std::shared_ptr<Tensor> &input) {
    CHECK(input->Dtype() == DataType::kFLOAT32) << "CUDA ReLUForward currently supports float32 only";
    auto output = std::make_shared<Tensor>(input->Dims(), DataType::kFLOAT32, input->GetDevice());
    LaunchForward(output, input);
    return output;
}

std::shared_ptr<Tensor> ReLUBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &grad_output) {
    CHECK(input->Dtype() == DataType::kFLOAT32) << "CUDA ReLUBackward currently supports float32 only";
    CHECK(grad_output->Dtype() == DataType::kFLOAT32) << "CUDA ReLUBackward currently supports float32 only";
    auto grad_input = std::make_shared<Tensor>(input->Dims(), DataType::kFLOAT32, input->GetDevice());
    LaunchBackward(grad_input, input, grad_output);
    return grad_input;
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_RELU_KERNEL(kernel_name)                                                                         \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_RELU_KERNEL(ReLUForward)
REGISTER_CUDA_RELU_KERNEL(ReLUBackward)

#undef REGISTER_CUDA_RELU_KERNEL
