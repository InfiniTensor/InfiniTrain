#include <cfloat>
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

__device__ inline int64_t AdaptiveStartIndexDev(int64_t out_idx,
                                                int64_t out_size,
                                                int64_t in_size) {
  return (out_idx * in_size) / out_size;
}

__device__ inline int64_t AdaptiveEndIndexDev(int64_t out_idx, int64_t out_size,
                                              int64_t in_size) {
  return ((out_idx + 1) * in_size + out_size - 1) / out_size;
}

template <typename T>
__global__ void MaxPool2dForwardKernel(
    const T *__restrict__ input, T *__restrict__ output,
    int64_t *__restrict__ indices, int64_t NC, int64_t H, int64_t W,
    int64_t H_out, int64_t W_out, int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w) {
  const int64_t total = NC * H_out * W_out;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += stride) {
    const int64_t ow = idx % W_out;
    const int64_t tmp = idx / W_out;
    const int64_t oh = tmp % H_out;
    const int64_t nc = tmp / H_out;

    const int64_t raw_h0 = oh * stride_h - pad_h;
    const int64_t raw_h1 = raw_h0 + kernel_h;
    const int64_t h_start = raw_h0 > 0 ? raw_h0 : 0;
    const int64_t h_end = raw_h1 < H ? raw_h1 : H;
    const int64_t raw_w0 = ow * stride_w - pad_w;
    const int64_t raw_w1 = raw_w0 + kernel_w;
    const int64_t w_start = raw_w0 > 0 ? raw_w0 : 0;
    const int64_t w_end = raw_w1 < W ? raw_w1 : W;

    const T *in_nc = input + nc * H * W;
    float max_val = -FLT_MAX;
    int64_t max_idx = h_start * W + w_start;
    for (int64_t ih = h_start; ih < h_end; ++ih) {
      for (int64_t iw = w_start; iw < w_end; ++iw) {
        const float v = common::cuda::Cast<float>(in_nc[ih * W + iw]);
        if (v > max_val) {
          max_val = v;
          max_idx = ih * W + iw;
        }
      }
    }
    output[idx] = common::cuda::Cast<T>(max_val);
    indices[idx] = max_idx;
  }
}

__global__ void MaxPool2dBackwardKernelFP32(const float *__restrict__ grad_out,
                                            const int64_t *__restrict__ indices,
                                            float *__restrict__ grad_in,
                                            int64_t NC, int64_t HW,
                                            int64_t HW_out) {
  const int64_t total = NC * HW_out;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += stride) {
    const int64_t nc = idx / HW_out;
    const int64_t in_spatial = indices[idx];
    atomicAdd(grad_in + nc * HW + in_spatial, grad_out[idx]);
  }
}

template <int BLOCK_SIZE, typename T>
__global__ void GlobalAvgPool2dForwardKernel(const T *__restrict__ input,
                                             T *__restrict__ output, int64_t NC,
                                             int64_t HW) {
  using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  const int64_t nc = blockIdx.x;
  if (nc >= NC) {
    return;
  }
  const T *in_nc = input + nc * HW;
  float sum = 0.0f;
  for (int64_t i = threadIdx.x; i < HW; i += BLOCK_SIZE) {
    sum += common::cuda::Cast<float>(in_nc[i]);
  }
  const float reduced = BlockReduce(temp_storage).Sum(sum);
  if (threadIdx.x == 0) {
    output[nc] = common::cuda::Cast<T>(reduced / static_cast<float>(HW));
  }
}

template <typename T>
__global__ void GlobalAvgPool2dBackwardKernel(const T *__restrict__ grad_out,
                                              T *__restrict__ grad_in,
                                              int64_t NC, int64_t HW) {
  const int64_t total = NC * HW;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const float inv_hw = 1.0f / static_cast<float>(HW);
  for (int64_t idx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += stride) {
    const int64_t nc = idx / HW;
    grad_in[idx] =
        common::cuda::Cast<T>(common::cuda::Cast<float>(grad_out[nc]) * inv_hw);
  }
}

template <typename T>
__global__ void AdaptiveAvgPool2dForwardKernel(const T *__restrict__ input,
                                               T *__restrict__ output,
                                               int64_t NC, int64_t H, int64_t W,
                                               int64_t out_h, int64_t out_w) {
  const int64_t total = NC * out_h * out_w;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += stride) {
    const int64_t ow = idx % out_w;
    const int64_t tmp = idx / out_w;
    const int64_t oh = tmp % out_h;
    const int64_t nc = tmp / out_h;

    const int64_t h0 = AdaptiveStartIndexDev(oh, out_h, H);
    const int64_t h1 = AdaptiveEndIndexDev(oh, out_h, H);
    const int64_t w0 = AdaptiveStartIndexDev(ow, out_w, W);
    const int64_t w1 = AdaptiveEndIndexDev(ow, out_w, W);

    const T *in_nc = input + nc * H * W;
    float sum = 0.0f;
    for (int64_t ih = h0; ih < h1; ++ih) {
      for (int64_t iw = w0; iw < w1; ++iw) {
        sum += common::cuda::Cast<float>(in_nc[ih * W + iw]);
      }
    }
    output[idx] =
        common::cuda::Cast<T>(sum / static_cast<float>((h1 - h0) * (w1 - w0)));
  }
}

__global__ void AdaptiveAvgPool2dBackwardKernelFP32(
    const float *__restrict__ grad_out, float *__restrict__ grad_in, int64_t NC,
    int64_t H, int64_t W, int64_t out_h, int64_t out_w) {
  const int64_t total = NC * H * W;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += stride) {
    const int64_t iw = idx % W;
    const int64_t tmp = idx / W;
    const int64_t ih = tmp % H;
    const int64_t nc = tmp / H;

    const float *go_nc = grad_out + nc * out_h * out_w;
    float sum = 0.0f;
    for (int64_t oh = 0; oh < out_h; ++oh) {
      const int64_t h0 = AdaptiveStartIndexDev(oh, out_h, H);
      const int64_t h1 = AdaptiveEndIndexDev(oh, out_h, H);
      if (ih < h0 || ih >= h1) {
        continue;
      }
      for (int64_t ow = 0; ow < out_w; ++ow) {
        const int64_t w0 = AdaptiveStartIndexDev(ow, out_w, W);
        const int64_t w1 = AdaptiveEndIndexDev(ow, out_w, W);
        if (iw < w0 || iw >= w1) {
          continue;
        }
        sum +=
            go_nc[oh * out_w + ow] / static_cast<float>((h1 - h0) * (w1 - w0));
      }
    }
    grad_in[idx] = sum;
  }
}

cudaStream_t GetCudaStream(const Device &device) {
  return dynamic_cast<infini_train::core::cuda::CudaStream *>(
             infini_train::core::GetDeviceGuardImpl(device.type())
                 ->GetStream(device))
      ->cuda_stream();
}

}  // namespace

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> MaxPool2dForward(
    const std::shared_ptr<Tensor> &input, int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w) {
  CHECK_EQ(input->Dims().size(), 4);
  const int64_t N = input->Dims()[0];
  const int64_t C = input->Dims()[1];
  const int64_t H = input->Dims()[2];
  const int64_t W = input->Dims()[3];

  const int64_t H_out = (H + 2 * pad_h - kernel_h) / stride_h + 1;
  const int64_t W_out = (W + 2 * pad_w - kernel_w) / stride_w + 1;
  CHECK_GT(H_out, 0);
  CHECK_GT(W_out, 0);

  auto device = input->GetDevice();
  auto output = std::make_shared<Tensor>(
      std::vector<int64_t>{N, C, H_out, W_out}, input->Dtype(), device);
  auto indices = std::make_shared<Tensor>(
      std::vector<int64_t>{N, C, H_out, W_out}, DataType::kINT64, device);

  const int64_t total = N * C * H_out * W_out;
  const int threads = 256;
  const int blocks = static_cast<int>(
      std::min<int64_t>((total + threads - 1) / threads, 65535));
  const auto stream = GetCudaStream(device);

  core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
      input->Dtype(),
      [=]<typename T>() {
        MaxPool2dForwardKernel<<<blocks, threads, 0, stream>>>(
            static_cast<const T *>(input->DataPtr()),
            static_cast<T *>(output->DataPtr()),
            static_cast<int64_t *>(indices->DataPtr()), N * C, H, W, H_out,
            W_out, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
      },
      "CUDA MaxPool2dForward");
  return {output, indices};
}

std::shared_ptr<Tensor> MaxPool2dBackward(
    const std::shared_ptr<Tensor> &grad_output,
    const std::shared_ptr<Tensor> &indices,
    const std::vector<int64_t> &input_dims) {
  CHECK_EQ(input_dims.size(), 4);
  CHECK_EQ(static_cast<int>(grad_output->Dtype()),
           static_cast<int>(DataType::kFLOAT32));
  const int64_t N = input_dims[0];
  const int64_t C = input_dims[1];
  const int64_t H = input_dims[2];
  const int64_t W = input_dims[3];
  const int64_t H_out = grad_output->Dims()[2];
  const int64_t W_out = grad_output->Dims()[3];

  auto device = grad_output->GetDevice();
  auto grad_input =
      std::make_shared<Tensor>(input_dims, grad_output->Dtype(), device);
  grad_input->Fill(0.0f);

  const int64_t total = N * C * H_out * W_out;
  const int threads = 256;
  const int blocks = static_cast<int>(
      std::min<int64_t>((total + threads - 1) / threads, 65535));
  const auto stream = GetCudaStream(device);

  MaxPool2dBackwardKernelFP32<<<blocks, threads, 0, stream>>>(
      static_cast<const float *>(grad_output->DataPtr()),
      static_cast<const int64_t *>(indices->DataPtr()),
      static_cast<float *>(grad_input->DataPtr()), N * C, H * W, H_out * W_out);
  return grad_input;
}

std::shared_ptr<Tensor> AdaptiveAvgPool2dForward(
    const std::shared_ptr<Tensor> &input, int64_t out_h, int64_t out_w) {
  CHECK_EQ(input->Dims().size(), 4);
  const int64_t N = input->Dims()[0];
  const int64_t C = input->Dims()[1];
  const int64_t H = input->Dims()[2];
  const int64_t W = input->Dims()[3];
  CHECK_GT(out_h, 0);
  CHECK_GT(out_w, 0);

  auto device = input->GetDevice();
  auto output = std::make_shared<Tensor>(
      std::vector<int64_t>{N, C, out_h, out_w}, input->Dtype(), device);
  const auto stream = GetCudaStream(device);

  if (out_h == 1 && out_w == 1) {
    constexpr int kBlockSize = 256;
    core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
        input->Dtype(),
        [=]<typename T>() {
          GlobalAvgPool2dForwardKernel<kBlockSize>
              <<<static_cast<int>(N * C), kBlockSize, 0, stream>>>(
                  static_cast<const T *>(input->DataPtr()),
                  static_cast<T *>(output->DataPtr()), N * C, H * W);
        },
        "CUDA GlobalAvgPool2dForward");
    return output;
  }

  const int64_t total = N * C * out_h * out_w;
  const int threads = 256;
  const int blocks = static_cast<int>(
      std::min<int64_t>((total + threads - 1) / threads, 65535));
  core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
      input->Dtype(),
      [=]<typename T>() {
        AdaptiveAvgPool2dForwardKernel<<<blocks, threads, 0, stream>>>(
            static_cast<const T *>(input->DataPtr()),
            static_cast<T *>(output->DataPtr()), N * C, H, W, out_h, out_w);
      },
      "CUDA AdaptiveAvgPool2dForward");
  return output;
}

std::shared_ptr<Tensor> AdaptiveAvgPool2dBackward(
    const std::shared_ptr<Tensor> &grad_output,
    const std::vector<int64_t> &input_dims) {
  CHECK_EQ(input_dims.size(), 4);
  const int64_t N = input_dims[0];
  const int64_t C = input_dims[1];
  const int64_t H = input_dims[2];
  const int64_t W = input_dims[3];
  const int64_t out_h = grad_output->Dims()[2];
  const int64_t out_w = grad_output->Dims()[3];

  auto device = grad_output->GetDevice();
  auto grad_input =
      std::make_shared<Tensor>(input_dims, grad_output->Dtype(), device);
  const auto stream = GetCudaStream(device);
  const int64_t total = N * C * H * W;
  const int threads = 256;
  const int blocks = static_cast<int>(
      std::min<int64_t>((total + threads - 1) / threads, 65535));

  if (out_h == 1 && out_w == 1) {
    core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
        grad_output->Dtype(),
        [=]<typename T>() {
          GlobalAvgPool2dBackwardKernel<<<blocks, threads, 0, stream>>>(
              static_cast<const T *>(grad_output->DataPtr()),
              static_cast<T *>(grad_input->DataPtr()), N * C, H * W);
        },
        "CUDA GlobalAvgPool2dBackward");
    return grad_input;
  }

  CHECK_EQ(static_cast<int>(grad_output->Dtype()),
           static_cast<int>(DataType::kFLOAT32));
  AdaptiveAvgPool2dBackwardKernelFP32<<<blocks, threads, 0, stream>>>(
      static_cast<const float *>(grad_output->DataPtr()),
      static_cast<float *>(grad_input->DataPtr()), N * C, H, W, out_h, out_w);
  return grad_input;
}

}  // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_POOLING_KERNEL(kernel_name)                       \
  REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, \
                  infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_POOLING_KERNEL(MaxPool2dForward)
REGISTER_CUDA_POOLING_KERNEL(MaxPool2dBackward)
REGISTER_CUDA_POOLING_KERNEL(AdaptiveAvgPool2dForward)
REGISTER_CUDA_POOLING_KERNEL(AdaptiveAvgPool2dBackward)

#undef REGISTER_CUDA_POOLING_KERNEL
