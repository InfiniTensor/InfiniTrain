#include "glog/logging.h"
#include <cub/block/block_reduce.cuh>

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {

template <int BLOCK_SIZE>
__global__ void LayerNormForwardKernel(const float *input, const float *weight, const float *bias, float *mean_out,
                                       float *rstd_out, float *output, float eps, int embed_dim) {
    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage_mean;
    __shared__ typename BlockReduce::TempStorage temp_storage_rstd;
    __shared__ float shared_mean;
    __shared__ float shared_rstd;

    const int token_idx = blockIdx.x;
    const float *x = input + token_idx * embed_dim;
    float *y = output + token_idx * embed_dim;

    float sum = 0.0f;
    float sqsum = 0.0f;
    for (int i = threadIdx.x; i < embed_dim; i += BLOCK_SIZE) {
        float val = x[i];
        sum += val;
        sqsum += val * val;
    }

    float total_sum = BlockReduce(temp_storage_mean).Sum(sum);
    __syncthreads();

    float total_sqsum = BlockReduce(temp_storage_rstd).Sum(sqsum);
    __syncthreads();

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
        float norm = (x[i] - shared_mean) * shared_rstd;
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

    constexpr int BLOCK_SIZE = 256;
    int threads_per_block = BLOCK_SIZE;
    int num_blocks = batch_size * max_seqlen;

    const auto *cuda_device = dynamic_cast<const CudaDevice *>(input->GetDevice());
    LayerNormForwardKernel<BLOCK_SIZE><<<num_blocks, threads_per_block, 0, cuda_device->Stream()>>>(
        static_cast<const float *>(input->DataPtr()), static_cast<const float *>(weight->DataPtr()),
        static_cast<const float *>(bias->DataPtr()), static_cast<float *>(mean->DataPtr()),
        static_cast<float *>(rstd->DataPtr()), static_cast<float *>(output->DataPtr()), eps, embed_dim);
    return {output, mean, rstd};
}

template <int BLOCK_SIZE>
__global__ void LayerNormBackwardKernel(const float *__restrict__ input, const float *__restrict__ grad_output,
                                        const float *__restrict__ mean, const float *__restrict__ rstd,
                                        const float *__restrict__ weight, float *__restrict__ grad_input,
                                        float *__restrict__ grad_weight, float *__restrict__ grad_bias, int embed_dim) {
    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage_mean;
    __shared__ typename BlockReduce::TempStorage temp_storage_norm;
    __shared__ float shared_mean;
    __shared__ float shared_norm;

    int tid = threadIdx.x;
    int token_idx = blockIdx.x;

    const float *input_ptr = input + token_idx * embed_dim;
    const float *grad_output_ptr = grad_output + token_idx * embed_dim;
    float *grad_input_ptr = grad_input + token_idx * embed_dim;

    float mean_val = mean[token_idx];
    float rstd_val = rstd[token_idx];

    float dnorm_mean = 0.f;
    float dnorm_norm_mean = 0.f;

    for (int i = tid; i < embed_dim; i += BLOCK_SIZE) {
        float dnorm = weight[i] * grad_output_ptr[i];
        dnorm_mean += dnorm;
        dnorm_norm_mean += dnorm * (input_ptr[i] - mean_val);
    }

    dnorm_mean = BlockReduce(temp_storage_mean).Sum(dnorm_mean);
    __syncthreads();
    dnorm_norm_mean = BlockReduce(temp_storage_norm).Sum(dnorm_norm_mean);
    __syncthreads();

    if (tid == 0) {
        float mean_d = dnorm_mean / embed_dim;
        float norm_d = (dnorm_norm_mean / embed_dim) * rstd_val - mean_d * mean_val * rstd_val;
        shared_mean = mean_d;
        shared_norm = norm_d;
    }
    __syncthreads();

    for (int i = tid; i < embed_dim; i += BLOCK_SIZE) {
        float norm = (input_ptr[i] - mean_val) * rstd_val;

        grad_input_ptr[i] = (weight[i] * grad_output_ptr[i] - shared_mean - norm * shared_norm) * rstd_val;

        atomicAdd(&grad_weight[i], grad_output_ptr[i] * norm);
        atomicAdd(&grad_bias[i], grad_output_ptr[i]);
    }
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
LayerNormBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                  const std::shared_ptr<Tensor> &bias, const std::shared_ptr<Tensor> &mean,
                  const std::shared_ptr<Tensor> &rstd, const std::shared_ptr<Tensor> &grad_output) {
    const int batch_size = input->Dims()[0];
    const int max_seqlen = input->Dims()[1];
    const int embed_dim = input->Dims()[2];

    auto grad_input = std::make_shared<Tensor>(input->Dims(), DataType::kFLOAT32, grad_output->GetDevice());
    auto grad_weight = std::make_shared<Tensor>(weight->Dims(), DataType::kFLOAT32, grad_output->GetDevice());
    auto grad_bias = std::make_shared<Tensor>(bias->Dims(), DataType::kFLOAT32, grad_output->GetDevice());
    grad_input->Fill<float>(0.0f);
    grad_weight->Fill<float>(0.0f);
    grad_bias->Fill<float>(0.0f);

    constexpr int BLOCK_SIZE = 256;
    int threads_per_block = BLOCK_SIZE;
    int num_blocks = batch_size * max_seqlen;

    const auto *cuda_device = dynamic_cast<const CudaDevice *>(input->GetDevice());
    LayerNormBackwardKernel<BLOCK_SIZE><<<num_blocks, threads_per_block, 0, cuda_device->Stream()>>>(
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
