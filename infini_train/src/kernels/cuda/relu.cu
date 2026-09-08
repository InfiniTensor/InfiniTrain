#include <memory>

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

namespace infini_train::kernels::cuda {
namespace {

template <bool kBackward>
__global__ void ReLUKernel(const float *input, const float *grad_output, float *output, size_t num_elements) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elements) {
        return;
    }
    if constexpr (kBackward) {
        output[index] = input[index] > 0.0f ? grad_output[index] : 0.0f;
    } else {
        output[index] = input[index] > 0.0f ? input[index] : 0.0f;
    }
}

cudaStream_t GetCudaStream(const Device &device) {
    return dynamic_cast<core::cuda::CudaStream *>(core::GetDeviceGuardImpl(device.type())->GetStream(device))
        ->cuda_stream();
}

} // namespace

std::shared_ptr<Tensor> ReLUForward(const std::shared_ptr<Tensor> &input) {
    CHECK(input->Dtype() == DataType::kFLOAT32);
    auto output = std::make_shared<Tensor>(input->Dims(), DataType::kFLOAT32, input->GetDevice());
    constexpr int kThreads = 256;
    const int blocks = (input->NumElements() + kThreads - 1) / kThreads;
    ReLUKernel<false><<<blocks, kThreads, 0, GetCudaStream(input->GetDevice())>>>(
        static_cast<const float *>(input->DataPtr()), nullptr, static_cast<float *>(output->DataPtr()),
        input->NumElements());
    CUDA_CHECK(cudaGetLastError());
    return output;
}

std::shared_ptr<Tensor> ReLUBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &grad_output) {
    CHECK(input->Dtype() == DataType::kFLOAT32);
    CHECK(grad_output->Dtype() == DataType::kFLOAT32);
    CHECK(input->GetDevice() == grad_output->GetDevice());
    CHECK(input->Dims() == grad_output->Dims());
    auto grad_input = std::make_shared<Tensor>(input->Dims(), DataType::kFLOAT32, input->GetDevice());
    constexpr int kThreads = 256;
    const int blocks = (input->NumElements() + kThreads - 1) / kThreads;
    ReLUKernel<true><<<blocks, kThreads, 0, GetCudaStream(input->GetDevice())>>>(
        static_cast<const float *>(input->DataPtr()), static_cast<const float *>(grad_output->DataPtr()),
        static_cast<float *>(grad_input->DataPtr()), input->NumElements());
    CUDA_CHECK(cudaGetLastError());
    return grad_input;
}

} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_RELU_KERNEL(kernel_name)                                                                         \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_RELU_KERNEL(ReLUForward)
REGISTER_CUDA_RELU_KERNEL(ReLUBackward)

#undef REGISTER_CUDA_RELU_KERNEL
