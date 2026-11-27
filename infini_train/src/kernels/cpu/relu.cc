#include <stdint.h>

#include <algorithm>
#include <memory>

#include "glog/logging.h"

#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {

std::shared_ptr<Tensor> ReLUForward(const std::shared_ptr<Tensor> &input) {
  CHECK_EQ(static_cast<int>(input->Dtype()),
           static_cast<int>(DataType::kFLOAT32));
  auto output = std::make_shared<Tensor>(input->Dims(), DataType::kFLOAT32);
  const float *in_ptr = static_cast<const float *>(input->DataPtr());
  float *out_ptr = static_cast<float *>(output->DataPtr());
  const int64_t numel = static_cast<int64_t>(input->NumElements());

#pragma omp parallel for
  for (int64_t idx = 0; idx < numel; ++idx) {
    out_ptr[idx] = in_ptr[idx] > 0.0f ? in_ptr[idx] : 0.0f;
  }
  return output;
}

std::shared_ptr<Tensor> ReLUBackward(
    const std::shared_ptr<Tensor> &input,
    const std::shared_ptr<Tensor> &grad_output) {
  CHECK_EQ(static_cast<int>(input->Dtype()),
           static_cast<int>(DataType::kFLOAT32));
  auto grad_input = std::make_shared<Tensor>(input->Dims(), DataType::kFLOAT32);
  const float *in_ptr = static_cast<const float *>(input->DataPtr());
  const float *go_ptr = static_cast<const float *>(grad_output->DataPtr());
  float *gi_ptr = static_cast<float *>(grad_input->DataPtr());
  const int64_t numel = static_cast<int64_t>(input->NumElements());

#pragma omp parallel for
  for (int64_t idx = 0; idx < numel; ++idx) {
    gi_ptr[idx] = in_ptr[idx] > 0.0f ? go_ptr[idx] : 0.0f;
  }
  return grad_input;
}

std::shared_ptr<Tensor> ReLU6Forward(const std::shared_ptr<Tensor> &input) {
  CHECK_EQ(static_cast<int>(input->Dtype()),
           static_cast<int>(DataType::kFLOAT32));
  auto output = std::make_shared<Tensor>(input->Dims(), DataType::kFLOAT32);
  const float *in_ptr = static_cast<const float *>(input->DataPtr());
  float *out_ptr = static_cast<float *>(output->DataPtr());
  const int64_t numel = static_cast<int64_t>(input->NumElements());

#pragma omp parallel for
  for (int64_t idx = 0; idx < numel; ++idx) {
    out_ptr[idx] = std::min(std::max(in_ptr[idx], 0.0f), 6.0f);
  }
  return output;
}

std::shared_ptr<Tensor> ReLU6Backward(
    const std::shared_ptr<Tensor> &input,
    const std::shared_ptr<Tensor> &grad_output) {
  CHECK_EQ(static_cast<int>(input->Dtype()),
           static_cast<int>(DataType::kFLOAT32));
  auto grad_input = std::make_shared<Tensor>(input->Dims(), DataType::kFLOAT32);
  const float *in_ptr = static_cast<const float *>(input->DataPtr());
  const float *go_ptr = static_cast<const float *>(grad_output->DataPtr());
  float *gi_ptr = static_cast<float *>(grad_input->DataPtr());
  const int64_t numel = static_cast<int64_t>(input->NumElements());

#pragma omp parallel for
  for (int64_t idx = 0; idx < numel; ++idx) {
    const float x = in_ptr[idx];
    gi_ptr[idx] = (x > 0.0f && x < 6.0f) ? go_ptr[idx] : 0.0f;
  }
  return grad_input;
}

}  // namespace infini_train::kernels::cpu

#define REGISTER_CPU_RELU_KERNEL(kernel_name)                          \
  REGISTER_KERNEL(infini_train::Device::DeviceType::kCPU, kernel_name, \
                  infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_RELU_KERNEL(ReLUForward)
REGISTER_CPU_RELU_KERNEL(ReLUBackward)
REGISTER_CPU_RELU_KERNEL(ReLU6Forward)
REGISTER_CPU_RELU_KERNEL(ReLU6Backward)

#undef REGISTER_CPU_RELU_KERNEL
