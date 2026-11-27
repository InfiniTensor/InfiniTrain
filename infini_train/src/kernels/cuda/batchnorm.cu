#include <cmath>
#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

#include <cub/block/block_reduce.cuh>

#include "glog/logging.h"

#include "infini_train/include/common/cuda/kernel_helper.cuh"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/cuda/cuda_dispatch.h"
#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

namespace infini_train::kernels::cuda {
namespace {

template <int BLOCK_SIZE, typename T>
__global__ void BatchNorm2dStatsKernel(const T *__restrict__ input,
                                       float *__restrict__ save_mean,
                                       float *__restrict__ save_invstd,
                                       float *__restrict__ running_mean,
                                       float *__restrict__ running_var,
                                       int64_t N, int64_t C, int64_t HW,
                                       float momentum, float eps) {
  using BlockReduce = cub::BlockReduce<double, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ double shared_mean;

  const int64_t c = blockIdx.x;
  if (c >= C) {
    return;
  }
  const int64_t M = N * HW;
  double sum = 0.0;
  for (int64_t idx = threadIdx.x; idx < M; idx += BLOCK_SIZE) {
    const int64_t n = idx / HW;
    const int64_t hw = idx % HW;
    sum += static_cast<double>(
        common::cuda::Cast<float>(input[(n * C + c) * HW + hw]));
  }
  const double total_sum = BlockReduce(temp_storage).Sum(sum);
  if (threadIdx.x == 0) {
    shared_mean = total_sum / static_cast<double>(M);
  }
  __syncthreads();

  const double mean = shared_mean;
  double sq_sum = 0.0;
  for (int64_t idx = threadIdx.x; idx < M; idx += BLOCK_SIZE) {
    const int64_t n = idx / HW;
    const int64_t hw = idx % HW;
    const double diff = static_cast<double>(common::cuda::Cast<float>(
                            input[(n * C + c) * HW + hw])) -
                        mean;
    sq_sum += diff * diff;
  }
  const double total_sq = BlockReduce(temp_storage).Sum(sq_sum);

  if (threadIdx.x == 0) {
    const double var_biased = total_sq / static_cast<double>(M);
    const double invstd = 1.0 / sqrt(var_biased + static_cast<double>(eps));
    save_mean[c] = static_cast<float>(mean);
    save_invstd[c] = static_cast<float>(invstd);
    if (running_mean != nullptr && running_var != nullptr) {
      const double var_unbiased =
          M > 1 ? total_sq / static_cast<double>(M - 1) : var_biased;
      const double mom = static_cast<double>(momentum);
      running_mean[c] =
          static_cast<float>((1.0 - mom) * running_mean[c] + mom * mean);
      running_var[c] =
          static_cast<float>((1.0 - mom) * running_var[c] + mom * var_unbiased);
    }
  }
}

__global__ void BatchNorm2dEvalStatsKernel(
    const float *__restrict__ running_mean,
    const float *__restrict__ running_var, float *__restrict__ save_mean,
    float *__restrict__ save_invstd, int64_t C, float eps) {
  const int64_t c = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (c < C) {
    save_mean[c] = running_mean[c];
    save_invstd[c] =
        static_cast<float>(1.0 / sqrt(static_cast<double>(running_var[c]) +
                                      static_cast<double>(eps)));
  }
}

template <typename T>
__global__ void BatchNorm2dNormalizeKernel(
    const T *__restrict__ input, const float *__restrict__ weight,
    const float *__restrict__ bias, const float *__restrict__ save_mean,
    const float *__restrict__ save_invstd, T *__restrict__ output,
    int64_t total, int64_t C, int64_t HW) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += stride) {
    const int64_t c = (idx / HW) % C;
    const float x = common::cuda::Cast<float>(input[idx]);
    const float y = (x - save_mean[c]) * (weight[c] * save_invstd[c]) + bias[c];
    output[idx] = common::cuda::Cast<T>(y);
  }
}

template <int BLOCK_SIZE, typename T>
__global__ void BatchNorm2dBackwardReduceKernel(
    const T *__restrict__ grad_output, const T *__restrict__ input,
    const float *__restrict__ save_mean, const float *__restrict__ save_invstd,
    float *__restrict__ grad_weight, float *__restrict__ grad_bias, int64_t N,
    int64_t C, int64_t HW) {
  using BlockReduce = cub::BlockReduce<double, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  const int64_t c = blockIdx.x;
  if (c >= C) {
    return;
  }
  const int64_t M = N * HW;
  const double mean = static_cast<double>(save_mean[c]);
  const double invstd = static_cast<double>(save_invstd[c]);

  double sum_dy = 0.0;
  double sum_dy_xhat = 0.0;
  for (int64_t idx = threadIdx.x; idx < M; idx += BLOCK_SIZE) {
    const int64_t n = idx / HW;
    const int64_t hw = idx % HW;
    const int64_t offset = (n * C + c) * HW + hw;
    const double dy =
        static_cast<double>(common::cuda::Cast<float>(grad_output[offset]));
    const double xhat =
        (static_cast<double>(common::cuda::Cast<float>(input[offset])) - mean) *
        invstd;
    sum_dy += dy;
    sum_dy_xhat += dy * xhat;
  }

  const double reduced_dy = BlockReduce(temp_storage).Sum(sum_dy);
  __syncthreads();
  const double reduced_dy_xhat = BlockReduce(temp_storage).Sum(sum_dy_xhat);

  if (threadIdx.x == 0) {
    grad_bias[c] = static_cast<float>(reduced_dy);
    grad_weight[c] = static_cast<float>(reduced_dy_xhat);
  }
}

template <typename T>
__global__ void BatchNorm2dBackwardInputKernel(
    const T *__restrict__ grad_output, const T *__restrict__ input,
    const float *__restrict__ weight, const float *__restrict__ save_mean,
    const float *__restrict__ save_invstd,
    const float *__restrict__ grad_weight, const float *__restrict__ grad_bias,
    T *__restrict__ grad_input, int64_t total, int64_t N, int64_t C, int64_t HW,
    bool training) {
  const int64_t M = N * HW;
  const float inv_m = 1.0f / static_cast<float>(M);
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += stride) {
    const int64_t c = (idx / HW) % C;
    const float dy = common::cuda::Cast<float>(grad_output[idx]);
    const float w_invstd = weight[c] * save_invstd[c];
    if (training) {
      const float xhat =
          (common::cuda::Cast<float>(input[idx]) - save_mean[c]) *
          save_invstd[c];
      const float dx =
          w_invstd * (dy - (grad_bias[c] + xhat * grad_weight[c]) * inv_m);
      grad_input[idx] = common::cuda::Cast<T>(dx);
    } else {
      grad_input[idx] = common::cuda::Cast<T>(dy * w_invstd);
    }
  }
}

cudaStream_t GetCudaStream(const Device &device) {
  return dynamic_cast<infini_train::core::cuda::CudaStream *>(
             infini_train::core::GetDeviceGuardImpl(device.type())
                 ->GetStream(device))
      ->cuda_stream();
}

}  // namespace

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>,
           std::shared_ptr<Tensor>>
BatchNorm2dForward(const std::shared_ptr<Tensor> &input,
                   const std::shared_ptr<Tensor> &weight,
                   const std::shared_ptr<Tensor> &bias,
                   const std::shared_ptr<Tensor> &running_mean,
                   const std::shared_ptr<Tensor> &running_var, bool training,
                   float momentum, float eps) {
  CHECK_EQ(input->Dims().size(), 4);
  const int64_t N = input->Dims()[0];
  const int64_t C = input->Dims()[1];
  const int64_t H = input->Dims()[2];
  const int64_t W = input->Dims()[3];
  const int64_t HW = H * W;
  const int64_t total = N * C * HW;

  auto device = input->GetDevice();
  auto output = std::make_shared<Tensor>(input->Dims(), input->Dtype(), device);
  auto save_mean = std::make_shared<Tensor>(std::vector<int64_t>{C},
                                            DataType::kFLOAT32, device);
  auto save_invstd = std::make_shared<Tensor>(std::vector<int64_t>{C},
                                              DataType::kFLOAT32, device);
  const auto stream = GetCudaStream(device);

  constexpr int kBlockSize = 256;
  if (training) {
    float *rm_ptr =
        running_mean ? static_cast<float *>(running_mean->DataPtr()) : nullptr;
    float *rv_ptr =
        running_var ? static_cast<float *>(running_var->DataPtr()) : nullptr;
    core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
        input->Dtype(),
        [=]<typename T>() {
          BatchNorm2dStatsKernel<kBlockSize>
              <<<static_cast<int>(C), kBlockSize, 0, stream>>>(
                  static_cast<const T *>(input->DataPtr()),
                  static_cast<float *>(save_mean->DataPtr()),
                  static_cast<float *>(save_invstd->DataPtr()), rm_ptr, rv_ptr,
                  N, C, HW, momentum, eps);
        },
        "CUDA BatchNorm2dStats");
  } else {
    CHECK(running_mean != nullptr && running_var != nullptr);
    const int c_blocks = static_cast<int>((C + kBlockSize - 1) / kBlockSize);
    BatchNorm2dEvalStatsKernel<<<c_blocks, kBlockSize, 0, stream>>>(
        static_cast<const float *>(running_mean->DataPtr()),
        static_cast<const float *>(running_var->DataPtr()),
        static_cast<float *>(save_mean->DataPtr()),
        static_cast<float *>(save_invstd->DataPtr()), C, eps);
  }

  const int norm_blocks = static_cast<int>(
      std::min<int64_t>((total + kBlockSize - 1) / kBlockSize, 65535));
  core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
      input->Dtype(),
      [=]<typename T>() {
        BatchNorm2dNormalizeKernel<<<norm_blocks, kBlockSize, 0, stream>>>(
            static_cast<const T *>(input->DataPtr()),
            static_cast<const float *>(weight->DataPtr()),
            static_cast<const float *>(bias->DataPtr()),
            static_cast<const float *>(save_mean->DataPtr()),
            static_cast<const float *>(save_invstd->DataPtr()),
            static_cast<T *>(output->DataPtr()), total, C, HW);
      },
      "CUDA BatchNorm2dNormalize");

  return {output, save_mean, save_invstd};
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>,
           std::shared_ptr<Tensor>>
BatchNorm2dBackward(const std::shared_ptr<Tensor> &grad_output,
                    const std::shared_ptr<Tensor> &input,
                    const std::shared_ptr<Tensor> &weight,
                    const std::shared_ptr<Tensor> &save_mean,
                    const std::shared_ptr<Tensor> &save_invstd, bool training) {
  CHECK_EQ(input->Dims().size(), 4);
  const int64_t N = input->Dims()[0];
  const int64_t C = input->Dims()[1];
  const int64_t H = input->Dims()[2];
  const int64_t W = input->Dims()[3];
  const int64_t HW = H * W;
  const int64_t total = N * C * HW;

  auto device = input->GetDevice();
  auto grad_input =
      std::make_shared<Tensor>(input->Dims(), input->Dtype(), device);
  auto grad_weight = std::make_shared<Tensor>(std::vector<int64_t>{C},
                                              DataType::kFLOAT32, device);
  auto grad_bias = std::make_shared<Tensor>(std::vector<int64_t>{C},
                                            DataType::kFLOAT32, device);
  const auto stream = GetCudaStream(device);

  constexpr int kBlockSize = 256;
  core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
      input->Dtype(),
      [=]<typename T>() {
        BatchNorm2dBackwardReduceKernel<kBlockSize>
            <<<static_cast<int>(C), kBlockSize, 0, stream>>>(
                static_cast<const T *>(grad_output->DataPtr()),
                static_cast<const T *>(input->DataPtr()),
                static_cast<const float *>(save_mean->DataPtr()),
                static_cast<const float *>(save_invstd->DataPtr()),
                static_cast<float *>(grad_weight->DataPtr()),
                static_cast<float *>(grad_bias->DataPtr()), N, C, HW);
      },
      "CUDA BatchNorm2dBackwardReduce");

  const int dx_blocks = static_cast<int>(
      std::min<int64_t>((total + kBlockSize - 1) / kBlockSize, 65535));
  core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
      input->Dtype(),
      [=]<typename T>() {
        BatchNorm2dBackwardInputKernel<<<dx_blocks, kBlockSize, 0, stream>>>(
            static_cast<const T *>(grad_output->DataPtr()),
            static_cast<const T *>(input->DataPtr()),
            static_cast<const float *>(weight->DataPtr()),
            static_cast<const float *>(save_mean->DataPtr()),
            static_cast<const float *>(save_invstd->DataPtr()),
            static_cast<const float *>(grad_weight->DataPtr()),
            static_cast<const float *>(grad_bias->DataPtr()),
            static_cast<T *>(grad_input->DataPtr()), total, N, C, HW, training);
      },
      "CUDA BatchNorm2dBackwardInput");

  return {grad_input, grad_weight, grad_bias};
}

}  // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_BATCHNORM_KERNEL(kernel_name)                     \
  REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, \
                  infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_BATCHNORM_KERNEL(BatchNorm2dForward)
REGISTER_CUDA_BATCHNORM_KERNEL(BatchNorm2dBackward)

#undef REGISTER_CUDA_BATCHNORM_KERNEL
