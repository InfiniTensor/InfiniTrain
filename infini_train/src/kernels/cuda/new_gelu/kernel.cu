#include "infini_train/src/kernels/cuda/new_gelu/kernel.h"

#include <algorithm>
#include <cassert>
#include <cmath>

#include <cuda_runtime.h>

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"

#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"
#include "infini_train/src/kernels/cuda/new_gelu/kernel.cuh"

namespace infini_train::kernels::cuda {

__global__ void NewGELUForwardKernel(const float* input, size_t count,
                                     float* output, float* saved_tanh) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += size_t(gridDim.x) * blockDim.x) {
    const float x = input[i];
    const float cube = powf(x, 3.0f);
    const float value = tanhf(__fmul_rn(
        0.7978845608028654f, __fadd_rn(x, __fmul_rn(0.044715f, cube))));
    saved_tanh[i] = value;
    output[i] = __fmul_rn(__fmul_rn(0.5f, x), __fadd_rn(1.0f, value));
  }
}

__global__ void NewGELUBackwardKernel(const float* input,
                                      const float* saved_tanh,
                                      const float* gradient, size_t count,
                                      float* grad_input) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += size_t(gridDim.x) * blockDim.x) {
    const float x = input[i], value = saved_tanh[i];
    const float product = __fmul_rn(gradient[i], __fmul_rn(0.5f, x));
    const float partial =
        __fmul_rn(__fmul_rn(product, __fsub_rn(1.0f, __fmul_rn(value, value))),
                  0.7978845608028654f);
    const float cubic = __fmul_rn(__fmul_rn(partial, 0.044715f),
                                  __fmul_rn(3.0f, powf(x, 2.0f)));
    const float direct =
        __fmul_rn(__fmul_rn(gradient[i], __fadd_rn(1.0f, value)), 0.5f);
    grad_input[i] = __fadd_rn(direct, __fadd_rn(partial, cubic));
  }
}

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> NewGELUForward(
    const std::shared_ptr<Tensor>& input) {
  assert(input->Dtype() == DataType::kFLOAT32 && input->IsContiguous());
  const auto device = input->GetDevice();
  core::DeviceGuard guard(device);
  auto output = std::make_shared<Tensor>(input->Dims(), input->Dtype(), device);
  auto saved_tanh =
      std::make_shared<Tensor>(input->Dims(), input->Dtype(), device);
  const size_t count = input->NumElements();
  if (count != 0) {
    auto stream =
        dynamic_cast<core::cuda::CudaStream*>(
            core::GetDeviceGuardImpl(device.type())->GetStream(device))
            ->cuda_stream();
    const auto blocks =
        static_cast<unsigned>(std::min<size_t>((count - 1) / 256 + 1, 65535));
    NewGELUForwardKernel<<<blocks, 256, 0, stream>>>(
        static_cast<const float*>(input->DataPtr()), count,
        static_cast<float*>(output->DataPtr()),
        static_cast<float*>(saved_tanh->DataPtr()));
  }
  return {output, saved_tanh};
}

std::shared_ptr<Tensor> NewGELUBackward(
    const std::shared_ptr<Tensor>& input,
    const std::shared_ptr<Tensor>& saved_tanh,
    const std::shared_ptr<Tensor>& gradient) {
  assert(input->Dtype() == DataType::kFLOAT32 &&
         gradient->Dtype() == input->Dtype());
  assert(gradient->Dims() == input->Dims() && gradient->IsContiguous());
  const auto device = input->GetDevice();
  core::DeviceGuard guard(device);
  auto result = std::make_shared<Tensor>(input->Dims(), input->Dtype(), device);
  const size_t count = input->NumElements();
  if (count != 0) {
    auto stream =
        dynamic_cast<core::cuda::CudaStream*>(
            core::GetDeviceGuardImpl(device.type())->GetStream(device))
            ->cuda_stream();
    const auto blocks =
        static_cast<unsigned>(std::min<size_t>((count - 1) / 256 + 1, 65535));
    NewGELUBackwardKernel<<<blocks, 256, 0, stream>>>(
        static_cast<const float*>(input->DataPtr()),
        static_cast<const float*>(saved_tanh->DataPtr()),
        static_cast<const float*>(gradient->DataPtr()), count,
        static_cast<float*>(result->DataPtr()));
  }
  return result;
}

}  // namespace infini_train::kernels::cuda

REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, NewGELUForward,
                infini_train::kernels::cuda::NewGELUForward)
REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, NewGELUBackward,
                infini_train::kernels::cuda::NewGELUBackward)
