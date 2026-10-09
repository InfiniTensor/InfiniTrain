#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {

__global__ void Conv2dForwardKernel(const float *input, const float *weight, const float *bias,
                                    float *output, int64_t N, int64_t C_in, int64_t H, int64_t W,
                                    int64_t C_out, int64_t K, int64_t H_out, int64_t W_out,
                                    int64_t stride, int64_t padding) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = N * C_out * H_out * W_out;
    if (idx >= total) return;

    int64_t ow = idx % W_out;
    int64_t oh = (idx / W_out) % H_out;
    int64_t co = (idx / (W_out * H_out)) % C_out;
    int64_t n = idx / (W_out * H_out * C_out);

    float sum = bias ? bias[co] : 0.0f;
    for (int64_t ci = 0; ci < C_in; ++ci) {
        for (int64_t kh = 0; kh < K; ++kh) {
            for (int64_t kw = 0; kw < K; ++kw) {
                int64_t ih = oh * stride + kh - padding;
                int64_t iw = ow * stride + kw - padding;
                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                sum += input[((n * C_in + ci) * H + ih) * W + iw]
                     * weight[((co * C_in + ci) * K + kh) * K + kw];
            }
        }
    }
    output[idx] = sum;
}

std::shared_ptr<Tensor> Conv2dForward(const std::shared_ptr<Tensor> &input,
                                      const std::shared_ptr<Tensor> &weight,
                                      const std::shared_ptr<Tensor> &bias,
                                      int64_t stride, int64_t padding) {
    const auto &input_dims = input->Dims();
    const auto &weight_dims = weight->Dims();
    int64_t N = input_dims[0], C_in = input_dims[1], H = input_dims[2], W = input_dims[3];
    int64_t C_out = weight_dims[0], K = weight_dims[2];
    int64_t H_out = (H + 2 * padding - K) / stride + 1;
    int64_t W_out = (W + 2 * padding - K) / stride + 1;

    auto output = std::make_shared<Tensor>(std::vector<int64_t>{N, C_out, H_out, W_out}, DataType::kFLOAT32, input->GetDevice());

    int64_t total = N * C_out * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    Conv2dForwardKernel<<<blocks, threads>>>(
        static_cast<const float *>(input->DataPtr()),
        static_cast<const float *>(weight->DataPtr()),
        bias ? static_cast<const float *>(bias->DataPtr()) : nullptr,
        static_cast<float *>(output->DataPtr()),
        N, C_in, H, W, C_out, K, H_out, W_out, stride, padding);

    return output;
}

__global__ void Conv2dBackwardInputKernel(const float *weight, const float *grad_output,
                                          float *grad_input, int64_t N, int64_t C_in, int64_t H,
                                          int64_t W, int64_t C_out, int64_t K, int64_t H_out,
                                          int64_t W_out, int64_t stride, int64_t padding) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = N * C_in * H * W;
    if (idx >= total) return;

    int64_t iw = idx % W;
    int64_t ih = (idx / W) % H;
    int64_t ci = (idx / (W * H)) % C_in;
    int64_t n = idx / (W * H * C_in);

    float sum = 0.0f;
    for (int64_t co = 0; co < C_out; ++co) {
        for (int64_t kh = 0; kh < K; ++kh) {
            for (int64_t kw = 0; kw < K; ++kw) {
                int64_t oh = (ih + padding - kh);
                int64_t ow = (iw + padding - kw);
                if (oh % stride != 0 || ow % stride != 0) continue;
                oh /= stride;
                ow /= stride;
                if (oh < 0 || oh >= H_out || ow < 0 || ow >= W_out) continue;
                sum += grad_output[((n * C_out + co) * H_out + oh) * W_out + ow]
                     * weight[((co * C_in + ci) * K + kh) * K + kw];
            }
        }
    }
    grad_input[idx] = sum;
}

std::shared_ptr<Tensor> Conv2dBackwardInput(const std::shared_ptr<Tensor> &weight,
                                            const std::shared_ptr<Tensor> &grad_output,
                                            const std::vector<int64_t> &input_dims,
                                            int64_t stride, int64_t padding) {
    int64_t N = input_dims[0], C_in = input_dims[1], H = input_dims[2], W = input_dims[3];
    const auto &weight_dims = weight->Dims();
    int64_t C_out = weight_dims[0], K = weight_dims[2];
    int64_t H_out = (H + 2 * padding - K) / stride + 1;
    int64_t W_out = (W + 2 * padding - K) / stride + 1;

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32, weight->GetDevice());

    int64_t total = N * C_in * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    Conv2dBackwardInputKernel<<<blocks, threads>>>(
        static_cast<const float *>(weight->DataPtr()),
        static_cast<const float *>(grad_output->DataPtr()),
        static_cast<float *>(grad_input->DataPtr()),
        N, C_in, H, W, C_out, K, H_out, W_out, stride, padding);

    return grad_input;
}

__global__ void Conv2dBackwardWeightKernel(const float *input, const float *grad_output,
                                           float *grad_weight, int64_t N, int64_t C_in, int64_t H,
                                           int64_t W, int64_t C_out, int64_t K, int64_t H_out,
                                           int64_t W_out, int64_t stride, int64_t padding) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = C_out * C_in * K * K;
    if (idx >= total) return;

    int64_t kw = idx % K;
    int64_t kh = (idx / K) % K;
    int64_t ci = (idx / (K * K)) % C_in;
    int64_t co = idx / (K * K * C_in);

    float sum = 0.0f;
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t oh = 0; oh < H_out; ++oh) {
            for (int64_t ow = 0; ow < W_out; ++ow) {
                int64_t ih = oh * stride + kh - padding;
                int64_t iw = ow * stride + kw - padding;
                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                sum += grad_output[((n * C_out + co) * H_out + oh) * W_out + ow]
                     * input[((n * C_in + ci) * H + ih) * W + iw];
            }
        }
    }
    grad_weight[idx] = sum;
}

std::shared_ptr<Tensor> Conv2dBackwardWeight(const std::shared_ptr<Tensor> &input,
                                             const std::shared_ptr<Tensor> &grad_output,
                                             const std::vector<int64_t> &weight_dims,
                                             int64_t stride, int64_t padding) {
    int64_t N = input->Dims()[0], C_in = input->Dims()[1], H = input->Dims()[2], W = input->Dims()[3];
    int64_t C_out = weight_dims[0], K = weight_dims[2];
    int64_t H_out = (H + 2 * padding - K) / stride + 1;
    int64_t W_out = (W + 2 * padding - K) / stride + 1;

    auto grad_weight = std::make_shared<Tensor>(weight_dims, DataType::kFLOAT32, input->GetDevice());

    int64_t total = C_out * C_in * K * K;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    Conv2dBackwardWeightKernel<<<blocks, threads>>>(
        static_cast<const float *>(input->DataPtr()),
        static_cast<const float *>(grad_output->DataPtr()),
        static_cast<float *>(grad_weight->DataPtr()),
        N, C_in, H, W, C_out, K, H_out, W_out, stride, padding);

    return grad_weight;
}

__global__ void Conv2dBackwardBiasKernel(const float *grad_output, float *grad_bias,
                                         int64_t N, int64_t C_out, int64_t H_out, int64_t W_out) {
    int64_t co = blockIdx.x * blockDim.x + threadIdx.x;
    if (co >= C_out) return;

    float sum = 0.0f;
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t oh = 0; oh < H_out; ++oh) {
            for (int64_t ow = 0; ow < W_out; ++ow) {
                sum += grad_output[((n * C_out + co) * H_out + oh) * W_out + ow];
            }
        }
    }
    grad_bias[co] = sum;
}

std::shared_ptr<Tensor> Conv2dBackwardBias(const std::shared_ptr<Tensor> &grad_output, int64_t out_channels) {
    const auto &grad_dims = grad_output->Dims();
    int64_t N = grad_dims[0], H_out = grad_dims[2], W_out = grad_dims[3];

    auto grad_bias = std::make_shared<Tensor>(std::vector<int64_t>{out_channels}, DataType::kFLOAT32, grad_output->GetDevice());

    int threads = 256;
    int blocks = (out_channels + threads - 1) / threads;

    Conv2dBackwardBiasKernel<<<blocks, threads>>>(
        static_cast<const float *>(grad_output->DataPtr()),
        static_cast<float *>(grad_bias->DataPtr()),
        N, out_channels, H_out, W_out);

    return grad_bias;
}

} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_CONV2D_KERNEL(kernel_name) \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_CONV2D_KERNEL(Conv2dForward)
REGISTER_CUDA_CONV2D_KERNEL(Conv2dBackwardInput)
REGISTER_CUDA_CONV2D_KERNEL(Conv2dBackwardWeight)
REGISTER_CUDA_CONV2D_KERNEL(Conv2dBackwardBias)

#undef REGISTER_CUDA_CONV2D_KERNEL