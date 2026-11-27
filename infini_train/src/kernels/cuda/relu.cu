#include <cstddef>
#include <memory>

#include "infini_train/include/common/cuda/kernel_helper.cuh"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/cuda/cuda_dispatch.h"
#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

namespace infini_train::kernels::cuda {
namespace {

template <typename T>
__global__ void ReLUForwardKernel(const T *__restrict__ input,
                                  T *__restrict__ output, size_t numel) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < numel; idx += stride) {
    const float x = common::cuda::Cast<float>(input[idx]);
    output[idx] = common::cuda::Cast<T>(x > 0.0f ? x : 0.0f);
  }
}

template <typename T>
__global__ void ReLUBackwardKernel(const T *__restrict__ input,
                                   const T *__restrict__ grad_output,
                                   T *__restrict__ grad_input, size_t numel) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < numel; idx += stride) {
    const float x = common::cuda::Cast<float>(input[idx]);
    grad_input[idx] = x > 0.0f ? grad_output[idx] : common::cuda::Cast<T>(0.0f);
  }
}

template <typename T>
__global__ void ReLU6ForwardKernel(const T *__restrict__ input,
                                   T *__restrict__ output, size_t numel) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < numel; idx += stride) {
    const float x = common::cuda::Cast<float>(input[idx]);
    const float y = x < 0.0f ? 0.0f : (x > 6.0f ? 6.0f : x);
    output[idx] = common::cuda::Cast<T>(y);
  }
}

template <typename T>
__global__ void ReLU6BackwardKernel(const T *__restrict__ input,
                                    const T *__restrict__ grad_output,
                                    T *__restrict__ grad_input, size_t numel) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < numel; idx += stride) {
    const float x = common::cuda::Cast<float>(input[idx]);
    grad_input[idx] =
        (x > 0.0f && x < 6.0f) ? grad_output[idx] : common::cuda::Cast<T>(0.0f);
  }
}

cudaStream_t GetCudaStream(const Device &device) {
  return dynamic_cast<infini_train::core::cuda::CudaStream *>(
             infini_train::core::GetDeviceGuardImpl(device.type())
                 ->GetStream(device))
      ->cuda_stream();
}

}  // namespace

std::shared_ptr<Tensor> ReLUForward(const std::shared_ptr<Tensor> &input) {
  auto output = std::make_shared<Tensor>(input->Dims(), input->Dtype(),
                                         input->GetDevice());
  const size_t numel = input->NumElements();
  if (numel == 0) {
    return output;
  }
  const int threads = 256;
  const int blocks = static_cast<int>(
      std::min<size_t>((numel + threads - 1) / threads, 65535));
  const auto stream = GetCudaStream(input->GetDevice());

  core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
      input->Dtype(),
      [=]<typename T>() {
        ReLUForwardKernel<<<blocks, threads, 0, stream>>>(
            static_cast<const T *>(input->DataPtr()),
            static_cast<T *>(output->DataPtr()), numel);
      },
      "CUDA ReLUForward");
  return output;
}

std::shared_ptr<Tensor> ReLUBackward(
    const std::shared_ptr<Tensor> &input,
    const std::shared_ptr<Tensor> &grad_output) {
  auto grad_input = std::make_shared<Tensor>(input->Dims(), input->Dtype(),
                                             input->GetDevice());
  const size_t numel = input->NumElements();
  if (numel == 0) {
    return grad_input;
  }
  const int threads = 256;
  const int blocks = static_cast<int>(
      std::min<size_t>((numel + threads - 1) / threads, 65535));
  const auto stream = GetCudaStream(input->GetDevice());

  core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
      input->Dtype(),
      [=]<typename T>() {
        ReLUBackwardKernel<<<blocks, threads, 0, stream>>>(
            static_cast<const T *>(input->DataPtr()),
            static_cast<const T *>(grad_output->DataPtr()),
            static_cast<T *>(grad_input->DataPtr()), numel);
      },
      "CUDA ReLUBackward");
  return grad_input;
}

std::shared_ptr<Tensor> ReLU6Forward(const std::shared_ptr<Tensor> &input) {
  auto output = std::make_shared<Tensor>(input->Dims(), input->Dtype(),
                                         input->GetDevice());
  const size_t numel = input->NumElements();
  if (numel == 0) {
    return output;
  }
  const int threads = 256;
  const int blocks = static_cast<int>(
      std::min<size_t>((numel + threads - 1) / threads, 65535));
  const auto stream = GetCudaStream(input->GetDevice());

  core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
      input->Dtype(),
      [=]<typename T>() {
        ReLU6ForwardKernel<<<blocks, threads, 0, stream>>>(
            static_cast<const T *>(input->DataPtr()),
            static_cast<T *>(output->DataPtr()), numel);
      },
      "CUDA ReLU6Forward");
  return output;
}

std::shared_ptr<Tensor> ReLU6Backward(
    const std::shared_ptr<Tensor> &input,
    const std::shared_ptr<Tensor> &grad_output) {
  auto grad_input = std::make_shared<Tensor>(input->Dims(), input->Dtype(),
                                             input->GetDevice());
  const size_t numel = input->NumElements();
  if (numel == 0) {
    return grad_input;
  }
  const int threads = 256;
  const int blocks = static_cast<int>(
      std::min<size_t>((numel + threads - 1) / threads, 65535));
  const auto stream = GetCudaStream(input->GetDevice());

  core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
      input->Dtype(),
      [=]<typename T>() {
        ReLU6BackwardKernel<<<blocks, threads, 0, stream>>>(
            static_cast<const T *>(input->DataPtr()),
            static_cast<const T *>(grad_output->DataPtr()),
            static_cast<T *>(grad_input->DataPtr()), numel);
      },
      "CUDA ReLU6Backward");
  return grad_input;
}

}  // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_RELU_KERNEL(kernel_name)                          \
  REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, \
                  infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_RELU_KERNEL(ReLUForward)
REGISTER_CUDA_RELU_KERNEL(ReLUBackward)
REGISTER_CUDA_RELU_KERNEL(ReLU6Forward)
REGISTER_CUDA_RELU_KERNEL(ReLU6Backward)

#undef REGISTER_CUDA_RELU_KERNEL
