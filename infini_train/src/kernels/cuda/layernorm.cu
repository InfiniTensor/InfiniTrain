#include <cub/block/block_reduce.cuh>

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/common/cuda/kernel_helper.cuh"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/cuda/cuda_dispatch.h"
#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"
#include "infini_train/src/kernels/cuda/layernorm/kernel.cuh"
#include "infini_train/src/kernels/cuda/layernorm/kernel.h"

namespace infini_train::kernels::cuda {

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>,
           std::shared_ptr<Tensor>>
StableLayerNormForward32(const std::shared_ptr<Tensor> &x,
                         const std::shared_ptr<Tensor> &w,
                         const std::shared_ptr<Tensor> &b, float eps) {
  CHECK(x->Dtype() == DataType::kFLOAT32 && w->Dtype() == x->Dtype() &&
        b->Dtype() == x->Dtype());
  CHECK_EQ(x->Dims().size(), 3);
  const int tokens = x->Dims()[0] * x->Dims()[1], width = x->Dims()[2];
  CHECK_GT(width, 0);
  CHECK_LE(width, w->NumElements());
  CHECK_LE(width, b->NumElements());
  auto device = x->GetDevice();
  auto y = std::make_shared<Tensor>(x->Dims(), DataType::kFLOAT32, device);
  auto mean =
      std::make_shared<Tensor>(std::vector<int64_t>{x->Dims()[0], x->Dims()[1]},
                               DataType::kFLOAT32, device);
  auto rstd =
      std::make_shared<Tensor>(mean->Dims(), DataType::kFLOAT32, device);
  auto stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                    core::GetDeviceGuardImpl(device.type())->GetStream(device))
                    ->cuda_stream();
  StableLNForward32<256><<<tokens, 256, 0, stream>>>(
      static_cast<const float *>(x->DataPtr()),
      static_cast<const float *>(w->DataPtr()),
      static_cast<const float *>(b->DataPtr()), width, eps,
      static_cast<float *>(mean->DataPtr()),
      static_cast<float *>(rstd->DataPtr()),
      static_cast<float *>(y->DataPtr()));
  return {y, mean, rstd};
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>,
           std::shared_ptr<Tensor>>
StableLayerNormBackward32(const std::shared_ptr<Tensor> &x,
                          const std::shared_ptr<Tensor> &w,
                          const std::shared_ptr<Tensor> &b,
                          const std::shared_ptr<Tensor> &rstd,
                          const std::shared_ptr<Tensor> &dy) {
  CHECK(x->Dtype() == DataType::kFLOAT32 && w->Dtype() == x->Dtype() &&
        b->Dtype() == x->Dtype() && dy->Dtype() == x->Dtype());
  CHECK_EQ(x->Dims().size(), 3);
  const int tokens = x->Dims()[0] * x->Dims()[1], width = x->Dims()[2];
  CHECK_GT(width, 0);
  CHECK_LE(width, w->NumElements());
  CHECK_LE(width, b->NumElements());
  auto device = x->GetDevice();
  auto dx = std::make_shared<Tensor>(x->Dims(), DataType::kFLOAT32, device);
  auto dw = std::make_shared<Tensor>(w->Dims(), DataType::kFLOAT32, device);
  auto db = std::make_shared<Tensor>(b->Dims(), DataType::kFLOAT32, device);
  auto normalized =
      std::make_shared<Tensor>(x->Dims(), DataType::kFLOAT32, device);
  dw->Fill(0);
  db->Fill(0);
  auto stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                    core::GetDeviceGuardImpl(device.type())->GetStream(device))
                    ->cuda_stream();
  StableLNInputGrad32<256><<<tokens, 256, 0, stream>>>(
      static_cast<const float *>(x->DataPtr()),
      static_cast<const float *>(dy->DataPtr()),
      static_cast<const float *>(w->DataPtr()),
      static_cast<const float *>(rstd->DataPtr()), width,
      static_cast<float *>(normalized->DataPtr()),
      static_cast<float *>(dx->DataPtr()));
  StableLNParameterGrad32<256><<<width, 256, 0, stream>>>(
      static_cast<const float *>(normalized->DataPtr()),
      static_cast<const float *>(dy->DataPtr()), tokens, width,
      static_cast<float *>(dw->DataPtr()), static_cast<float *>(db->DataPtr()));
  // Keep the temporary workspace alive until both kernels have completed.
  core::GetDeviceGuardImpl(device.type())->SynchronizeDevice(device);
  return {dx, dw, db};
}

template <int BLOCK_SIZE, typename T>
__global__ void LayerNormForwardKernel(const T *input, const T *weight,
                                       const T *bias, float *mean_out,
                                       float *rstd_out, T *output, float eps,
                                       int embed_dim) {
  using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage temp_storage_mean;
  __shared__ typename BlockReduce::TempStorage temp_storage_rstd;
  __shared__ float shared_mean;
  __shared__ float shared_rstd;

  const int token_idx = blockIdx.x;
  const T *x = input + token_idx * embed_dim;
  T *y = output + token_idx * embed_dim;

  float sum = 0.0f;
  float sqsum = 0.0f;

  for (int i = threadIdx.x; i < embed_dim; i += BLOCK_SIZE) {
    float val = common::cuda::Cast<float>(x[i]);
    sum += val;
    sqsum += val * val;
  }

  float total_sum = BlockReduce(temp_storage_mean).Sum(sum);
  float total_sqsum = BlockReduce(temp_storage_rstd).Sum(sqsum);

  if (threadIdx.x == 0) {
    float mean = total_sum / embed_dim;
    float var = total_sqsum / embed_dim - mean * mean;
    float rstd = rsqrtf(var + eps);
    shared_mean = mean;
    shared_rstd = rstd;
    if (mean_out) {
      mean_out[token_idx] = mean;
    }
    if (rstd_out) {
      rstd_out[token_idx] = rstd;
    }
  }
  __syncthreads();

  for (int i = threadIdx.x; i < embed_dim; i += BLOCK_SIZE) {
    float norm = (common::cuda::Cast<float>(x[i]) - shared_mean) * shared_rstd;
    y[i] = common::cuda::Cast<T>(norm * common::cuda::Cast<float>(weight[i]) +
                                 common::cuda::Cast<float>(bias[i]));
  }
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>,
           std::shared_ptr<Tensor>>
LayerNormForward(const std::shared_ptr<Tensor> &input,
                 const std::shared_ptr<Tensor> &weight,
                 const std::shared_ptr<Tensor> &bias, const float eps) {
  if (input->Dtype() == DataType::kFLOAT32) {
    return StableLayerNormForward32(input, weight, bias, eps);
  }

  CHECK_EQ(input->Dims().size(), 3);
  CHECK_LE(input->Dims()[2], weight->Dims()[0]);
  CHECK_LE(input->Dims()[2], bias->Dims()[0]);

  const int batch_size = input->Dims()[0];
  const int max_seqlen = input->Dims()[1];
  const int embed_dim = input->Dims()[2];

  auto dtype = input->Dtype();

  auto output =
      std::make_shared<Tensor>(input->Dims(), dtype, input->GetDevice());
  auto mean =
      std::make_shared<Tensor>(std::vector<int64_t>{batch_size, max_seqlen},
                               DataType::kFLOAT32, input->GetDevice());
  auto rstd =
      std::make_shared<Tensor>(std::vector<int64_t>{batch_size, max_seqlen},
                               DataType::kFLOAT32, input->GetDevice());

  constexpr int BLOCK_SIZE = 256;
  int threads_per_block = BLOCK_SIZE;
  int num_blocks = batch_size * max_seqlen;

  auto device = input->GetDevice();
  const auto &cuda_stream =
      dynamic_cast<infini_train::core::cuda::CudaStream *>(
          infini_train::core::GetDeviceGuardImpl(device.type())
              ->GetStream(device))
          ->cuda_stream();

  core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
      dtype,
      [=]<typename T>() {
        // Each token block writes its mean and rstd exactly once; no Fill is
        // needed.
        LayerNormForwardKernel<BLOCK_SIZE>
            <<<num_blocks, threads_per_block, 0, cuda_stream>>>(
                static_cast<const T *>(input->DataPtr()),
                static_cast<const T *>(weight->DataPtr()),
                static_cast<const T *>(bias->DataPtr()),
                static_cast<float *>(mean->DataPtr()),
                static_cast<float *>(rstd->DataPtr()),
                static_cast<T *>(output->DataPtr()), eps, embed_dim);
      },
      "CUDA LayerNormForward");

  return {output, mean, rstd};
}

template <int BLOCK_SIZE, typename T>
__global__ void LayerNormInputGradKernel(
    const T *__restrict__ input, const T *__restrict__ grad_output,
    const float *__restrict__ mean, const float *__restrict__ rstd,
    const T *__restrict__ weight, T *__restrict__ grad_input, int embed_dim) {
  using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage temp_storage_mean;
  __shared__ typename BlockReduce::TempStorage temp_storage_norm;
  __shared__ float shared_mean;
  __shared__ float shared_norm;

  int tid = threadIdx.x;
  int token_idx = blockIdx.x;

  const T *input_ptr = input + token_idx * embed_dim;
  const T *grad_output_ptr = grad_output + token_idx * embed_dim;
  T *grad_input_ptr = grad_input + token_idx * embed_dim;

  float mean_val = mean[token_idx];
  float rstd_val = rstd[token_idx];

  float dnorm_mean = 0.f;
  float dnorm_norm_mean = 0.f;

  for (int i = tid; i < embed_dim; i += BLOCK_SIZE) {
    float dnorm = common::cuda::Cast<float>(weight[i]) *
                  common::cuda::Cast<float>(grad_output_ptr[i]);
    float norm =
        (common::cuda::Cast<float>(input_ptr[i]) - mean_val) * rstd_val;
    dnorm_mean += dnorm;
    dnorm_norm_mean += dnorm * norm;
  }

  dnorm_mean = BlockReduce(temp_storage_mean).Sum(dnorm_mean);
  dnorm_norm_mean = BlockReduce(temp_storage_norm).Sum(dnorm_norm_mean);

  if (tid == 0) {
    float mean_d = dnorm_mean / embed_dim;
    float norm_d = dnorm_norm_mean / embed_dim;
    shared_mean = mean_d;
    shared_norm = norm_d;
  }
  __syncthreads();

  for (int i = tid; i < embed_dim; i += BLOCK_SIZE) {
    float norm =
        (common::cuda::Cast<float>(input_ptr[i]) - mean_val) * rstd_val;
    float grad_output_val = common::cuda::Cast<float>(grad_output_ptr[i]);

    grad_input_ptr[i] = common::cuda::Cast<T>(
        (common::cuda::Cast<float>(weight[i]) * grad_output_val - shared_mean -
         norm * shared_norm) *
        rstd_val);
  }
}

template <int BLOCK_SIZE, typename T>
__global__ void LayerNormParameterGradKernel(
    const T *__restrict__ input, const T *__restrict__ grad_output,
    const float *__restrict__ mean, const float *__restrict__ rstd,
    T *__restrict__ grad_weight, T *__restrict__ grad_bias, int64_t num_tokens,
    int embed_dim) {
  using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage temp_storage_weight;
  __shared__ typename BlockReduce::TempStorage temp_storage_bias;

  int feature_idx = blockIdx.x;
  float grad_weight_sum = 0.0f;
  float grad_bias_sum = 0.0f;

  for (int64_t token_idx = threadIdx.x; token_idx < num_tokens;
       token_idx += BLOCK_SIZE) {
    int64_t offset = token_idx * embed_dim + feature_idx;
    float grad_output_val = common::cuda::Cast<float>(grad_output[offset]);
    float norm = (common::cuda::Cast<float>(input[offset]) - mean[token_idx]) *
                 rstd[token_idx];
    grad_weight_sum += grad_output_val * norm;
    grad_bias_sum += grad_output_val;
  }

  float grad_weight_reduced =
      BlockReduce(temp_storage_weight).Sum(grad_weight_sum);
  float grad_bias_reduced = BlockReduce(temp_storage_bias).Sum(grad_bias_sum);

  if (threadIdx.x == 0) {
    grad_weight[feature_idx] = common::cuda::Cast<T>(grad_weight_reduced);
    grad_bias[feature_idx] = common::cuda::Cast<T>(grad_bias_reduced);
  }
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>,
           std::shared_ptr<Tensor>>
LayerNormBackward(const std::shared_ptr<Tensor> &input,
                  const std::shared_ptr<Tensor> &weight,
                  const std::shared_ptr<Tensor> &bias,
                  const std::shared_ptr<Tensor> &mean,
                  const std::shared_ptr<Tensor> &rstd,
                  const std::shared_ptr<Tensor> &grad_output) {
  if (input->Dtype() == DataType::kFLOAT32) {
    return StableLayerNormBackward32(input, weight, bias, rstd, grad_output);
  }

  CHECK_EQ(input->Dims().size(), 3);
  CHECK_LE(input->Dims()[2], weight->Dims()[0]);
  CHECK_LE(input->Dims()[2], bias->Dims()[0]);

  const int batch_size = input->Dims()[0];
  const int max_seqlen = input->Dims()[1];
  const int embed_dim = input->Dims()[2];
  const int64_t num_tokens = static_cast<int64_t>(batch_size) * max_seqlen;

  auto dtype = input->Dtype();
  CHECK(dtype == weight->Dtype() && dtype == bias->Dtype() &&
        dtype == grad_output->Dtype() && mean->Dtype() == DataType::kFLOAT32 &&
        rstd->Dtype() == DataType::kFLOAT32);

  auto grad_input =
      std::make_shared<Tensor>(input->Dims(), dtype, grad_output->GetDevice());
  auto grad_weight =
      std::make_shared<Tensor>(weight->Dims(), dtype, grad_output->GetDevice());
  auto grad_bias =
      std::make_shared<Tensor>(bias->Dims(), dtype, grad_output->GetDevice());

  constexpr int BLOCK_SIZE = 256;
  int threads_per_block = BLOCK_SIZE;
  int num_blocks = batch_size * max_seqlen;

  auto device = input->GetDevice();
  const auto &cuda_stream =
      dynamic_cast<infini_train::core::cuda::CudaStream *>(
          infini_train::core::GetDeviceGuardImpl(device.type())
              ->GetStream(device))
          ->cuda_stream();
  core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
      dtype,
      [=]<typename T>() {
        grad_weight->Fill(0);
        grad_bias->Fill(0);
        LayerNormInputGradKernel<BLOCK_SIZE>
            <<<num_blocks, threads_per_block, 0, cuda_stream>>>(
                static_cast<const T *>(input->DataPtr()),
                static_cast<const T *>(grad_output->DataPtr()),
                static_cast<const float *>(mean->DataPtr()),
                static_cast<const float *>(rstd->DataPtr()),
                static_cast<const T *>(weight->DataPtr()),
                static_cast<T *>(grad_input->DataPtr()), embed_dim);
        LayerNormParameterGradKernel<BLOCK_SIZE>
            <<<embed_dim, threads_per_block, 0, cuda_stream>>>(
                static_cast<const T *>(input->DataPtr()),
                static_cast<const T *>(grad_output->DataPtr()),
                static_cast<const float *>(mean->DataPtr()),
                static_cast<const float *>(rstd->DataPtr()),
                static_cast<T *>(grad_weight->DataPtr()),
                static_cast<T *>(grad_bias->DataPtr()), num_tokens, embed_dim);
      },
      "CUDA LayerNormBackward");

  return {grad_input, grad_weight, grad_bias};
}
}  // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_LAYERNORM_KERNEL(kernel_name)                     \
  REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, \
                  infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_LAYERNORM_KERNEL(LayerNormForward)
REGISTER_CUDA_LAYERNORM_KERNEL(LayerNormBackward)

#undef REGISTER_CUDA_LAYERNORM_KERNEL
