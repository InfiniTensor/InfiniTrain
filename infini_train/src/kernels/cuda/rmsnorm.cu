#include <cub/block/block_reduce.cuh>

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/common/cuda/kernel_helper.cuh"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/cuda/cuda_dispatch.h"
#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

namespace infini_train::kernels::cuda {

template <int BLOCK_SIZE, typename T>
__global__ void RMSNormForwardKernel(const T *__restrict__ input, const T *__restrict__ weight, float eps,
                                     T *__restrict__ output, float *__restrict__ rstd_out, int embed_dim) {
    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage_rstd;
    __shared__ float shared_rstd;

    const int token_idx = blockIdx.x;
    const T *x = input + token_idx * embed_dim;
    T *y = output + token_idx * embed_dim;

    float sqsum = 0.0f;

    for (int i = threadIdx.x; i < embed_dim; i += BLOCK_SIZE) {
        float val = common::cuda::Cast<float>(x[i]);
        sqsum += val * val;
    }

    float total_sqsum = BlockReduce(temp_storage_rstd).Sum(sqsum);

    if (threadIdx.x == 0) {
        float var = total_sqsum / embed_dim;
        float rstd = rsqrtf(var + eps);
        shared_rstd = rstd;
        if (rstd_out) {
            rstd_out[token_idx] = rstd;
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < embed_dim; i += BLOCK_SIZE) {
        // Keep the two multiplications separate: normalize by rstd first, then scale by weight.
        // A fused x * (rstd * weight) would change the rounding order of the composite path.
        float norm = common::cuda::Cast<float>(x[i]) * shared_rstd;
        y[i] = common::cuda::Cast<T>(norm * common::cuda::Cast<float>(weight[i]));
    }
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
RMSNormForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight, const float eps) {
    /*
        x: [..., embed_dim]
        -> RMSNorm (w: [embed_dim])
        -> o: [..., embed_dim]
    */
    // The composite path (Mean(-1)/Pow/Rsqrt/Mul) supports any rank, so the fused kernel keeps the
    // same generality: one block per leading-index row, reducing over the last dimension.
    CHECK_GE(input->Dims().size(), 2);
    CHECK_EQ(input->Dims().back(), weight->Dims()[0]);
    auto input_c = input->IsContiguous() ? input : input->Contiguous();

    const int embed_dim = static_cast<int>(input_c->Dims().back());
    const int64_t rows = input_c->NumElements() / embed_dim;

    auto dtype = input_c->Dtype();
    CHECK(dtype == weight->Dtype());

    auto output = std::make_shared<Tensor>(input_c->Dims(), dtype, input_c->GetDevice());
    auto rstd = std::make_shared<Tensor>(std::vector<int64_t>(input_c->Dims().begin(), input_c->Dims().end() - 1),
                                         DataType::kFLOAT32, input_c->GetDevice());

    constexpr int BLOCK_SIZE = 256;
    int threads_per_block = BLOCK_SIZE;
    int num_blocks = static_cast<int>(rows);

    auto device = input_c->GetDevice();
    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();

    core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
        dtype,
        [=]<typename T>() {
            // Each token block writes its rstd exactly once; no Fill is needed.
            RMSNormForwardKernel<BLOCK_SIZE><<<num_blocks, threads_per_block, 0, cuda_stream>>>(
                static_cast<const T *>(input_c->DataPtr()), static_cast<const T *>(weight->DataPtr()), eps,
                static_cast<T *>(output->DataPtr()), static_cast<float *>(rstd->DataPtr()), embed_dim);
        },
        "CUDA RMSNormForward");

    return {output, rstd};
}

template <int BLOCK_SIZE, typename T>
__global__ void RMSNormBackwardKernel(const T *__restrict__ input, const T *__restrict__ grad_output,
                                      const T *__restrict__ weight, const float *__restrict__ rstd,
                                      T *__restrict__ grad_input, T *__restrict__ grad_weight, int embed_dim,
                                      size_t weight_num_elements) {
    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float shared_K;

    const int token_idx = blockIdx.x;
    const T *x = input + token_idx * embed_dim;
    const T *g_out = grad_output + token_idx * embed_dim;
    T *g_in = grad_input + token_idx * embed_dim;
    const float rstd_val = rstd[token_idx];

    // S1 = sum_i(g_i * w_i * x_i); K = (S1 / H) * rstd, shared by the whole row.
    float S1 = 0.0f;
    for (int i = threadIdx.x; i < embed_dim; i += BLOCK_SIZE) {
        float xv = common::cuda::Cast<float>(x[i]);
        float wv = common::cuda::Cast<float>(weight[i]);
        float go = common::cuda::Cast<float>(g_out[i]);
        S1 += go * wv * xv;
    }
    S1 = BlockReduce(temp_storage).Sum(S1);
    if (threadIdx.x == 0) {
        shared_K = (S1 / embed_dim) * rstd_val;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < embed_dim; i += BLOCK_SIZE) {
        float xv = common::cuda::Cast<float>(x[i]);
        float wv = common::cuda::Cast<float>(weight[i]);
        float go = common::cuda::Cast<float>(g_out[i]);
        float norm = xv * rstd_val;
        g_in[i] = common::cuda::Cast<T>((go * wv - shared_K * norm) * rstd_val);
        common::cuda::fastAtomicAdd<T, size_t>(grad_weight, i, weight_num_elements, common::cuda::Cast<T>(go * norm),
                                               true);
    }
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
RMSNormBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                const std::shared_ptr<Tensor> &rstd, const std::shared_ptr<Tensor> &grad_output) {
    auto input_c = input->IsContiguous() ? input : input->Contiguous();

    const int embed_dim = static_cast<int>(input_c->Dims().back());
    const int64_t rows = input_c->NumElements() / embed_dim;

    auto dtype = input_c->Dtype();
    CHECK(dtype == weight->Dtype() && dtype == grad_output->Dtype() && rstd->Dtype() == DataType::kFLOAT32);

    auto grad_input = std::make_shared<Tensor>(input_c->Dims(), dtype, grad_output->GetDevice());
    auto grad_weight = std::make_shared<Tensor>(weight->Dims(), dtype, grad_output->GetDevice());

    constexpr int BLOCK_SIZE = 256;
    int threads_per_block = BLOCK_SIZE;
    int num_blocks = static_cast<int>(rows);

    auto device = input_c->GetDevice();
    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();
    core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
        dtype,
        [=]<typename T>() {
            // grad_weight accumulates across token blocks via atomics and must start at zero;
            // each token block fully overwrites its own grad_input slice, so no Fill is needed there.
            grad_weight->Fill(0.0);
            RMSNormBackwardKernel<BLOCK_SIZE><<<num_blocks, threads_per_block, 0, cuda_stream>>>(
                static_cast<const T *>(input_c->DataPtr()), static_cast<const T *>(grad_output->DataPtr()),
                static_cast<const T *>(weight->DataPtr()), static_cast<const float *>(rstd->DataPtr()),
                static_cast<T *>(grad_input->DataPtr()), static_cast<T *>(grad_weight->DataPtr()), embed_dim,
                grad_weight->NumElements());
        },
        "CUDA RMSNormBackward");

    return {grad_input, grad_weight};
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_RMSNORM_KERNEL(kernel_name)                                                                      \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_RMSNORM_KERNEL(RMSNormForward)
REGISTER_CUDA_RMSNORM_KERNEL(RMSNormBackward)

#undef REGISTER_CUDA_RMSNORM_KERNEL
