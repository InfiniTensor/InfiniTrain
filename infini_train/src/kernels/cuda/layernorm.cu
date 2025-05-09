#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {

#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t status = call;                                                                                     \
        if (status != cudaSuccess) {                                                                                   \
            LOG(FATAL) << "CUDA Error: " << cudaGetErrorString(status) << " at " << __FILE__ << ":" << __LINE__;       \
        }                                                                                                              \
    } while (0)

__global__ void LayerNormForwardKernel(const float *input, const float *weight, const float *bias, float *mean_out,
                                       float *rstd_out, float *output, float eps, int batch_size, int max_seqlen,
                                       int embed_dim) {
    int idx = blockIdx.x; // token idx
    const float *x = input + idx * embed_dim;
    float *y = output + idx * embed_dim;

    extern __shared__ float smem[]; // smem[0:threadnum] for sum, next for sqsum
    float sum = 0.0f;
    float sqsum = 0.0f;

    for (int i = threadIdx.x; i < embed_dim; i += blockDim.x) {
        float val = x[i];
        sum += val;
        sqsum += val * val;
    }

    // block reduce sum
    smem[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float mean = smem[0] / embed_dim;
    if (threadIdx.x == 0 && mean_out) {
        mean_out[idx] = mean;
    }
    __syncthreads();

    // block reduce sqsum
    smem[threadIdx.x] = sqsum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float var = smem[0] / embed_dim - mean * mean;
    float rstd = rsqrtf(var + eps);
    if (threadIdx.x == 0 && rstd_out) {
        rstd_out[idx] = rstd;
    }
    __syncthreads();

    // normalize
    for (int i = threadIdx.x; i < embed_dim; i += blockDim.x) {
        float norm = (x[i] - mean) * rstd;
        y[i] = norm * weight[i] + bias[i];
    }
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
LayerNormForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                 const std::shared_ptr<Tensor> &bias, const float eps) {
    CHECK_EQ(input->Dims().size(), 3);
    CHECK_LE(input->Dims()[2], weight->Dims()[0]);
    CHECK_LE(input->Dims()[2], bias->Dims()[0]);

    const int batch_size = input->Dims()[0];
    const int max_seqlen = input->Dims()[1];
    const int embed_dim = input->Dims()[2];

    auto output = std::make_shared<Tensor>(input->Dims(), DataType::kFLOAT32, input->GetDevice());
    auto mean = std::make_shared<Tensor>(std::vector<int64_t>{batch_size, max_seqlen}, DataType::kFLOAT32,
                                         input->GetDevice());
    auto rstd = std::make_shared<Tensor>(std::vector<int64_t>{batch_size, max_seqlen}, DataType::kFLOAT32,
                                         input->GetDevice());
    mean->Fill<float>(0.0f);
    rstd->Fill<float>(0.0f);

    int threads_per_block = 256;
    int num_blocks = batch_size * max_seqlen;
    int shared_mem_size = threads_per_block * sizeof(float);

    LayerNormForwardKernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
        static_cast<const float *>(input->DataPtr()), static_cast<const float *>(weight->DataPtr()),
        static_cast<const float *>(bias->DataPtr()), static_cast<float *>(mean->DataPtr()),
        static_cast<float *>(rstd->DataPtr()), static_cast<float *>(output->DataPtr()), eps, batch_size, max_seqlen,
        embed_dim);
    return {output, mean, rstd};
}

// Helper function for warp/block-wide reduction
__inline__ __device__ float BlockReduceSum(float val) {
    static __shared__ float shared_sum[32]; // assuming <= 1024 threads

    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    // Warp reduction
    for (int offset = 16; offset > 0; offset /= 2) { val += __shfl_down_sync(0xffffffff, val, offset); }

    // Write reduced warp result to shared memory
    if (lane == 0) {
        shared_sum[warp_id] = val;
    }

    __syncthreads();

    // Final reduction among warps
    val = (threadIdx.x < (blockDim.x + 31) / 32) ? shared_sum[lane] : 0.0f;
    if (warp_id == 0) {
        for (int offset = 16; offset > 0; offset /= 2) { val += __shfl_down_sync(0xffffffff, val, offset); }
    }
    return val;
}

__global__ void LayerNormBackwardKernel(const float *__restrict__ input, const float *__restrict__ grad_output,
                                        const float *__restrict__ mean, const float *__restrict__ rstd,
                                        const float *__restrict__ weight, float *__restrict__ grad_input,
                                        float *__restrict__ grad_weight, float *__restrict__ grad_bias, int embed_dim) {
    extern __shared__ float shared[]; // shared[0] = dnorm_mean, shared[1] = dnorm_norm_mean

    int tid = threadIdx.x;
    int token_idx = blockIdx.x;
    int stride = blockDim.x;

    const float *input_ptr = input + token_idx * embed_dim;
    const float *grad_output_ptr = grad_output + token_idx * embed_dim;
    float *grad_input_ptr = grad_input + token_idx * embed_dim;

    float mean_val = mean[token_idx];
    float rstd_val = rstd[token_idx];

    float dnorm_mean = 0.f;
    float dnorm_norm_mean = 0.f;

    // Step 1: accumulate dnorm_mean and dnorm_norm_mean
    for (int i = tid; i < embed_dim; i += stride) {
        float gi = grad_output_ptr[i];
        float wi = weight[i];
        float xi = input_ptr[i];
        float dnorm = wi * gi;
        dnorm_mean += dnorm;
        dnorm_norm_mean += dnorm * (xi - mean_val);
    }

    // Block-wide reductions
    dnorm_mean = BlockReduceSum(dnorm_mean);
    dnorm_norm_mean = BlockReduceSum(dnorm_norm_mean);

    __syncthreads();

    // Write reduced dnorm_mean and dnorm_norm_mean to shared memory
    if (tid == 0) {
        shared[0] = dnorm_mean / embed_dim;
        shared[1] = (dnorm_norm_mean / embed_dim) * rstd_val - shared[0] * mean_val * rstd_val;
    }
    __syncthreads();

    dnorm_mean = shared[0];
    dnorm_norm_mean = shared[1];

    // Step 2: compute grad_input and accumulate grad_weight, grad_bias per dimension
    for (int i = tid; i < embed_dim; i += stride) {
        float go = grad_output_ptr[i];
        float w = weight[i];
        float x = input_ptr[i];

        float norm = (x - mean_val) * rstd_val;

        // compute grad_input
        float dval = w * go;
        dval -= dnorm_mean;
        dval -= norm * dnorm_norm_mean;
        dval *= rstd_val;
        grad_input_ptr[i] = dval;

        // accumulate grad_weight and grad_bias
        atomicAdd(&grad_weight[i], go * norm);
        atomicAdd(&grad_bias[i], go);
    }
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
LayerNormBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                  const std::shared_ptr<Tensor> &bias, const std::shared_ptr<Tensor> &mean,
                  const std::shared_ptr<Tensor> &rstd, const std::shared_ptr<Tensor> &grad_output) {
    const int batch = input->Dims()[0];
    const int seqlen = input->Dims()[1];
    const int embed_dim = input->Dims()[2];

    auto grad_input = std::make_shared<Tensor>(input->Dims(), DataType::kFLOAT32, grad_output->GetDevice());
    auto grad_weight = std::make_shared<Tensor>(weight->Dims(), DataType::kFLOAT32, grad_output->GetDevice());
    auto grad_bias = std::make_shared<Tensor>(bias->Dims(), DataType::kFLOAT32, grad_output->GetDevice());
    grad_input->Fill<float>(0.0f);
    grad_weight->Fill<float>(0.0f);
    grad_bias->Fill<float>(0.0f);

    int num_blocks = batch * seqlen;
    int threads_per_block = 256;
    int shared_mem = 2 * sizeof(float); // for the block-wide `dnorm_mean` & `dnorm_norm_mean`

    LayerNormBackwardKernel<<<num_blocks, threads_per_block, shared_mem>>>(
        static_cast<const float *>(input->DataPtr()), static_cast<const float *>(grad_output->DataPtr()),
        static_cast<const float *>(mean->DataPtr()), static_cast<const float *>(rstd->DataPtr()),
        static_cast<const float *>(weight->DataPtr()), static_cast<float *>(grad_input->DataPtr()),
        static_cast<float *>(grad_weight->DataPtr()), static_cast<float *>(grad_bias->DataPtr()), embed_dim);
    return {grad_input, grad_weight, grad_bias};
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_LAYERNORM_KERNEL(kernel_name)                                                                    \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_LAYERNORM_KERNEL(LayerNormForward)
REGISTER_CUDA_LAYERNORM_KERNEL(LayerNormBackward)

#undef REGISTER_CUDA_LAYERNORM_KERNEL
