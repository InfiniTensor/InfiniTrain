#include "infini_train/include/device.h"
#include "infini_train/include/kernels/cuda/layernorm.h"

#include <cmath>
#include <memory>
#include <tuple>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {

#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t status = call;                                                                                     \
        if (status != cudaSuccess) {                                                                                   \
            LOG(FATAL) << "CUDA Error: " << cudaGetErrorString(status) << " at " << __FILE__ << ":" << __LINE__;       \
        }                                                                                                              \
    } while (0)

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

__forceinline__ __device__ float warpReduceSum(float val) {
    unsigned mask = __activemask();
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) { val += __shfl_down_sync(mask, val, offset); }
    return val;
}

__global__ void LayerNormForwardKernel(const float *input, const float *weight, const float *bias, float *mean,
                                       float *rstd, float *output, float eps, int batch_size, int max_seqlen,
                                       int embed_dim) {
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int idx = blockIdx.x * (blockDim.x / WARP_SIZE) + warp_id;
    if (idx >= batch_size * max_seqlen) {
        return;
    }

    float sum = 0.0f;
    for (int i = lane_id; i < embed_dim; i += WARP_SIZE) { sum += (float)input[idx * embed_dim + i]; }
    float m = warpReduceSum(sum) / embed_dim;
    if (lane_id == 0 && mean) {
        mean[idx] = m;
    }

    sum = 0.0f;
    for (int i = lane_id; i < embed_dim; i += WARP_SIZE) {
        float diff = (float)input[idx * embed_dim + i] - m;
        sum += diff * diff;
    }
    float s = rsqrtf(warpReduceSum(sum) / embed_dim + eps);
    if (lane_id == 0 && rstd) {
        rstd[idx] = s;
    }

    for (int c = lane_id; c < embed_dim; c += WARP_SIZE) {
        float n = s * ((float)input[idx * embed_dim + c] - m);
        output[idx * embed_dim + c] = (float)(n * (float)weight[c] + (float)bias[c]);
    }
}

std::shared_ptr<Tensor> LayerNormForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                                         const std::shared_ptr<Tensor> &bias, std::shared_ptr<Tensor> &mean,
                                         std::shared_ptr<Tensor> &rstd, const float eps) {
    CHECK_EQ(input->Dims().size(), 3);
    CHECK_LE(input->Dims()[2], weight->Dims()[0]);
    CHECK_LE(input->Dims()[2], bias->Dims()[0]);
    CHECK_EQ(mean, nullptr);
    CHECK_EQ(rstd, nullptr);

    const int batch_size = input->Dims()[0];
    const int max_seqlen = input->Dims()[1];
    const int embed_dim = input->Dims()[2];

    auto output = std::make_shared<Tensor>(std::vector<int64_t>{batch_size, max_seqlen, embed_dim}, DataType::kFLOAT32,
                                           Device(DeviceType::kCUDA, 0));
    mean = std::make_unique<Tensor>(std::vector<int64_t>{batch_size, max_seqlen}, DataType::kFLOAT32,
                                    Device(DeviceType::kCUDA, 0));
    rstd = std::make_unique<Tensor>(std::vector<int64_t>{batch_size, max_seqlen}, DataType::kFLOAT32,
                                    Device(DeviceType::kCUDA, 0));
    mean->Fill<float>(0.0f);
    rstd->Fill<float>(0.0f);

    int threads_per_block = 256;
    int warps_per_block = threads_per_block / WARP_SIZE;
    int num_blocks = (batch_size * max_seqlen + warps_per_block - 1) / warps_per_block;

    LayerNormForwardKernel<<<num_blocks, threads_per_block>>>(
        reinterpret_cast<const float *>(input->DataPtr()), reinterpret_cast<const float *>(weight->DataPtr()),
        reinterpret_cast<const float *>(bias->DataPtr()), reinterpret_cast<float *>(mean->DataPtr()),
        reinterpret_cast<float *>(rstd->DataPtr()), reinterpret_cast<float *>(output->DataPtr()), eps, batch_size,
        max_seqlen, embed_dim);

    return {output};
}

__global__ void LayerNormBackwardKernel(float *grad_input, float *grad_weight, float *grad_bias,
                                        const float *grad_output, const float *input, const float *weight,
                                        const float *mean, const float *rstd, int batch_size, int max_seqlen,
                                        int embed_dim) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;

    __shared__ float shared_grad_weight[32];
    __shared__ float shared_grad_bias[32];

    float grad_weight_sum = 0.0f;
    float grad_bias_sum = 0.0f;
    float grad_input_val = 0.0f;

    const int N = batch_size * max_seqlen * embed_dim;

    for (int i = bid * embed_dim + tid; i < N; i += gridDim.x * embed_dim) {
        int idx = i % embed_dim;
        float val_x = input[i];
        float val_grad_output = grad_output[i];
        float norm_x = (val_x - mean[i / embed_dim]) * rstd[i / embed_dim];

        grad_weight_sum += val_grad_output * norm_x;
        grad_bias_sum += val_grad_output;

        // Compute grad_input using grad_output
        grad_input_val = val_grad_output * weight[idx] * rstd[i / embed_dim];
        grad_input[i] = grad_input_val;
    }

    grad_weight_sum = warpReduceSum(grad_weight_sum);
    grad_bias_sum = warpReduceSum(grad_bias_sum);

    if (lane == 0) {
        shared_grad_weight[warp_id] = grad_weight_sum;
        shared_grad_bias[warp_id] = grad_bias_sum;
    }
    __syncthreads();

    if (warp_id == 0 && lane < warps_per_block) {
        grad_weight_sum = shared_grad_weight[lane];
        grad_bias_sum = shared_grad_bias[lane];
        grad_weight_sum = warpReduceSum(grad_weight_sum);
        grad_bias_sum = warpReduceSum(grad_bias_sum);

        if (lane == 0) {
            atomicAdd(&grad_weight[bid], grad_weight_sum);
            atomicAdd(&grad_bias[bid], grad_bias_sum);
        }
    }
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
LayerNormBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                  const std::shared_ptr<Tensor> &bias, const std::shared_ptr<Tensor> &mean,
                  const std::shared_ptr<Tensor> &rstd, const std::shared_ptr<Tensor> &grad_output) {
    CHECK_EQ(input->Dims().size(), 3);
    CHECK_LE(input->Dims()[2], weight->Dims()[0]);
    CHECK_LE(input->Dims()[2], bias->Dims()[0]);
    CHECK_NE(mean, nullptr);
    CHECK_NE(rstd, nullptr);

    const int batch_size = input->Dims()[0];
    const int max_seqlen = input->Dims()[1];
    const int embed_dim = input->Dims()[2];

    auto grad_input = std::make_shared<Tensor>(input->Dims(), DataType::kFLOAT32, Device(DeviceType::kCUDA, 0));
    auto grad_weight = std::make_shared<Tensor>(weight->Dims(), DataType::kFLOAT32, Device(DeviceType::kCUDA, 0));
    auto grad_bias = std::make_shared<Tensor>(bias->Dims(), DataType::kFLOAT32, Device(DeviceType::kCUDA, 0));

    grad_weight->Fill<float>(0.0f);
    grad_bias->Fill<float>(0.0f);

    int threads_per_block = 256;
    int num_blocks = (batch_size * max_seqlen * embed_dim + threads_per_block - 1) / threads_per_block;

    // TODO(zbl): check correctness
    LayerNormBackwardKernel<<<num_blocks, threads_per_block>>>(
        reinterpret_cast<float *>(grad_input->DataPtr()), reinterpret_cast<float *>(grad_weight->DataPtr()),
        reinterpret_cast<float *>(grad_bias->DataPtr()), reinterpret_cast<const float *>(grad_output->DataPtr()),
        reinterpret_cast<const float *>(input->DataPtr()), reinterpret_cast<const float *>(weight->DataPtr()),
        reinterpret_cast<const float *>(mean->DataPtr()), reinterpret_cast<const float *>(rstd->DataPtr()), batch_size,
        max_seqlen, embed_dim);
    CUDA_CHECK(cudaGetLastError());

    return {grad_input, grad_weight, grad_bias};
}
} // namespace infini_train::kernels::cuda
