#include <hipcub/block/block_reduce.hpp>

#include "infini_train/include/common/dcu/common_dcu.h"
#include "infini_train/include/common/dcu/kernel_helper.cuh"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/dcu/dcu_dispatch.h"
#include "infini_train/src/core/runtime/dcu/dcu_runtime_common.h"

namespace infini_train::kernels::dcu {

template <int BLOCK_SIZE, typename T>
__global__ void LayerNormForwardKernel(const T *input, const T *weight, const T *bias, float *mean_out, float *rstd_out,
                                       T *output, float eps, int embed_dim) {
    using BlockReduce = hipcub::BlockReduce<float, BLOCK_SIZE>;
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
        float val = common::dcu::Cast<float>(x[i]);
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
        float norm = (common::dcu::Cast<float>(x[i]) - shared_mean) * shared_rstd;
        y[i] = common::dcu::Cast<T>(norm * common::dcu::Cast<float>(weight[i]) + common::dcu::Cast<float>(bias[i]));
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

    auto dtype = input->Dtype();

    auto output = std::make_shared<Tensor>(input->Dims(), dtype, input->GetDevice());
    auto mean = std::make_shared<Tensor>(std::vector<int64_t>{batch_size, max_seqlen}, DataType::kFLOAT32,
                                         input->GetDevice());
    auto rstd = std::make_shared<Tensor>(std::vector<int64_t>{batch_size, max_seqlen}, DataType::kFLOAT32,
                                         input->GetDevice());

    constexpr int BLOCK_SIZE = 256;
    int threads_per_block = BLOCK_SIZE;
    int num_blocks = batch_size * max_seqlen;

    auto device = input->GetDevice();
    const auto &hip_stream = dynamic_cast<infini_train::core::dcu::DcuStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->hip_stream();

    core::dcu::DispatchDcuFunc<INFINI_ALL_FLOATING_TYPES>(
        dtype,
        [=]<typename T>() {
            // Each token block writes mean and rstd exactly once.
            LayerNormForwardKernel<BLOCK_SIZE><<<num_blocks, threads_per_block, 0, hip_stream>>>(
                static_cast<const T *>(input->DataPtr()), static_cast<const T *>(weight->DataPtr()),
                static_cast<const T *>(bias->DataPtr()), static_cast<float *>(mean->DataPtr()),
                static_cast<float *>(rstd->DataPtr()), static_cast<T *>(output->DataPtr()), eps, embed_dim);
        },
        "HIP LayerNormForward");

    return {output, mean, rstd};
}

template <int BLOCK_SIZE, typename T>
__global__ void LayerNormGradInputKernel(const T *__restrict__ input, const T *__restrict__ grad_output,
                                         const float *__restrict__ mean, const float *__restrict__ rstd,
                                         const T *__restrict__ weight, T *__restrict__ grad_input, int embed_dim) {
    using BlockReduce = hipcub::BlockReduce<float, BLOCK_SIZE>;
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
        float dnorm = common::dcu::Cast<float>(common::dcu::Mul(weight[i], grad_output_ptr[i]));
        dnorm_mean += dnorm;
        dnorm_norm_mean += dnorm * (common::dcu::Cast<float>(input_ptr[i]) - mean_val);
    }

    dnorm_mean = BlockReduce(temp_storage_mean).Sum(dnorm_mean);
    dnorm_norm_mean = BlockReduce(temp_storage_norm).Sum(dnorm_norm_mean);

    if (tid == 0) {
        float mean_d = dnorm_mean / embed_dim;
        // dnorm_norm_mean is accumulated against (x - mean), so multiplying by
        // rstd already produces mean(dnorm * x_hat). Subtracting the mean term
        // again would double-center the input and corrupt grad_input.
        float norm_d = (dnorm_norm_mean / embed_dim) * rstd_val;
        shared_mean = mean_d;
        shared_norm = norm_d;
    }
    __syncthreads();

    for (int i = tid; i < embed_dim; i += BLOCK_SIZE) {
        float norm = (common::dcu::Cast<float>(input_ptr[i]) - mean_val) * rstd_val;
        float grad_output_val = common::dcu::Cast<float>(grad_output_ptr[i]);

        grad_input_ptr[i] = common::dcu::Cast<T>(
            (common::dcu::Cast<float>(weight[i]) * grad_output_val - shared_mean - norm * shared_norm) * rstd_val);

    }
}

template <int BLOCK_SIZE, typename T>
__global__ void LayerNormParamGradKernel(const T *__restrict__ input, const T *__restrict__ grad_output,
                                         const float *__restrict__ mean, const float *__restrict__ rstd,
                                         T *__restrict__ grad_weight, T *__restrict__ grad_bias, int num_tokens,
                                         int embed_dim, size_t weight_num_elements, size_t bias_num_elements) {
    using BlockReduce = hipcub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage_weight;
    __shared__ typename BlockReduce::TempStorage temp_storage_bias;

    const int feature_idx = blockIdx.x;
    float grad_weight_sum = 0.0f;
    float grad_bias_sum = 0.0f;
    if (feature_idx < embed_dim) {
        for (int token_idx = threadIdx.x; token_idx < num_tokens; token_idx += BLOCK_SIZE) {
            const int64_t offset = static_cast<int64_t>(token_idx) * embed_dim + feature_idx;
            const float grad_output_val = common::dcu::Cast<float>(grad_output[offset]);
            const float norm = (common::dcu::Cast<float>(input[offset]) - mean[token_idx]) * rstd[token_idx];
            grad_weight_sum += grad_output_val * norm;
            grad_bias_sum += grad_output_val;
        }
    }

    grad_weight_sum = BlockReduce(temp_storage_weight).Sum(grad_weight_sum);
    grad_bias_sum = BlockReduce(temp_storage_bias).Sum(grad_bias_sum);
    if (threadIdx.x == 0) {
        if (static_cast<size_t>(feature_idx) < weight_num_elements) {
            grad_weight[feature_idx] = common::dcu::Cast<T>(grad_weight_sum);
        }
        if (static_cast<size_t>(feature_idx) < bias_num_elements) {
            grad_bias[feature_idx] = common::dcu::Cast<T>(grad_bias_sum);
        }
    }
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
LayerNormBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                  const std::shared_ptr<Tensor> &bias, const std::shared_ptr<Tensor> &mean,
                  const std::shared_ptr<Tensor> &rstd, const std::shared_ptr<Tensor> &grad_output) {
    const int batch_size = input->Dims()[0];
    const int max_seqlen = input->Dims()[1];
    const int embed_dim = input->Dims()[2];

    auto dtype = input->Dtype();
    CHECK(dtype == weight->Dtype() && dtype == bias->Dtype() && dtype == grad_output->Dtype()
          && mean->Dtype() == DataType::kFLOAT32 && rstd->Dtype() == DataType::kFLOAT32);

    auto grad_input = std::make_shared<Tensor>(input->Dims(), dtype, grad_output->GetDevice());
    auto grad_weight = std::make_shared<Tensor>(weight->Dims(), dtype, grad_output->GetDevice());
    auto grad_bias = std::make_shared<Tensor>(bias->Dims(), dtype, grad_output->GetDevice());

    constexpr int BLOCK_SIZE = 256;
    int threads_per_block = BLOCK_SIZE;
    int num_tokens = batch_size * max_seqlen;
    int param_blocks = std::max(weight->NumElements(), bias->NumElements());

    auto device = input->GetDevice();
    const auto &hip_stream = dynamic_cast<infini_train::core::dcu::DcuStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->hip_stream();
    core::dcu::DispatchDcuFunc<INFINI_ALL_FLOATING_TYPES>(
        dtype,
        [=]<typename T>() {
            LayerNormGradInputKernel<BLOCK_SIZE><<<num_tokens, threads_per_block, 0, hip_stream>>>(
                static_cast<const T *>(input->DataPtr()), static_cast<const T *>(grad_output->DataPtr()),
                static_cast<const float *>(mean->DataPtr()), static_cast<const float *>(rstd->DataPtr()),
                static_cast<const T *>(weight->DataPtr()), static_cast<T *>(grad_input->DataPtr()), embed_dim);
            LayerNormParamGradKernel<BLOCK_SIZE><<<param_blocks, threads_per_block, 0, hip_stream>>>(
                static_cast<const T *>(input->DataPtr()), static_cast<const T *>(grad_output->DataPtr()),
                static_cast<const float *>(mean->DataPtr()), static_cast<const float *>(rstd->DataPtr()),
                static_cast<T *>(grad_weight->DataPtr()), static_cast<T *>(grad_bias->DataPtr()), num_tokens,
                embed_dim, grad_weight->NumElements(), grad_bias->NumElements());
        },
        "HIP LayerNormBackward");

    return {grad_input, grad_weight, grad_bias};
}
} // namespace infini_train::kernels::dcu

#define REGISTER_HIP_LAYERNORM_KERNEL(kernel_name)                                                                    \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kDCU, kernel_name, infini_train::kernels::dcu::kernel_name)

REGISTER_HIP_LAYERNORM_KERNEL(LayerNormForward)
REGISTER_HIP_LAYERNORM_KERNEL(LayerNormBackward)

#undef REGISTER_HIP_LAYERNORM_KERNEL
