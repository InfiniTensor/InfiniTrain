#include <cstdint>
#include <memory>
#include <type_traits>

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/cuda/cuda_dispatch.h"
#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

namespace infini_train::kernels::cuda {

template <typename T>
__global__ void EmbeddingForwardKernel(const int64_t *input, T *output, const T *weight, int batch_size, int max_seqlen,
                                       int embed_dim, int vocab_size) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= batch_size * max_seqlen * embed_dim) {
        return;
    }

    int bt = idx / embed_dim;
    int b = bt / max_seqlen;
    int t = bt % max_seqlen;
    int c = idx % embed_dim;

    int ix = static_cast<int>(input[b * max_seqlen + t]);
    if (ix < 0 || ix >= vocab_size) {
        return;
    }
    output[b * max_seqlen * embed_dim + t * embed_dim + c] = weight[ix * embed_dim + c];
}

std::shared_ptr<Tensor> EmbeddingForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight) {
    CHECK(input->Dtype() == DataType::kINT64);
    CHECK_EQ(weight->Dims().size(), 2);

    auto device = input->GetDevice();
    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();

    const int batch_size = input->Dims().size() == 2 ? input->Dims()[0] : 1;
    const int max_seqlen = input->Dims().size() == 2 ? input->Dims()[1] : input->Dims()[0];
    const int vocab_size = weight->Dims()[0];
    const int embed_dim = weight->Dims()[1];
    auto output_dims = input->Dims();
    output_dims.push_back(embed_dim);

    auto dtype = weight->Dtype();
    auto output = std::make_shared<Tensor>(output_dims, dtype, input->GetDevice());
    int threads_per_block = 256;
    int num_blocks = (batch_size * max_seqlen * embed_dim + threads_per_block - 1) / threads_per_block;

    core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
        dtype,
        [=]<typename T>() {
            EmbeddingForwardKernel<<<num_blocks, threads_per_block, 0, cuda_stream>>>(
                static_cast<const int64_t *>(input->DataPtr()), static_cast<T *>(output->DataPtr()),
                static_cast<const T *>(weight->DataPtr()), batch_size, max_seqlen, embed_dim, vocab_size);
        },
        "CUDA EmbeddingForward");

    return output;
}

// One thread block per token: threads in the block split the embedding dimension and atomically accumulate the
// token's gradient row into grad_weight. This lifts the grid-level parallelism from ceil(num_tokens/256) blocks
// to num_tokens blocks. Rows untouched by any token keep the zero value written by the Fill below.
template <typename T>
__global__ void EmbeddingBackwardKernel(const int64_t *input_ptr, const T *grad_output_ptr, T *grad_weight_ptr,
                                        int num_tokens, int embedding_dim, int vocab_size) {
    const int idx = blockIdx.x;
    if (idx >= num_tokens) {
        return;
    }

    const int token_id = static_cast<int>(input_ptr[idx]);
    if (token_id < 0 || token_id >= vocab_size) {
        return;
    }

    const T *grad_row = grad_output_ptr + static_cast<int64_t>(idx) * embedding_dim;
    T *weight_row = grad_weight_ptr + static_cast<int64_t>(token_id) * embedding_dim;

    if constexpr (std::is_same_v<T, float>) {
        // 128-bit fast path for fp32: vectorized loads plus one vector atomicAdd per 16B chunk (sm_90+).
        if ((embedding_dim & 3) == 0 && (reinterpret_cast<uintptr_t>(grad_row) & 0xF) == 0
            && (reinterpret_cast<uintptr_t>(weight_row) & 0xF) == 0) {
            const float4 *grad_row4 = reinterpret_cast<const float4 *>(grad_row);
            float4 *weight_row4 = reinterpret_cast<float4 *>(weight_row);
            for (int j = threadIdx.x; j < embedding_dim / 4; j += blockDim.x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
                atomicAdd(&weight_row4[j], grad_row4[j]);
#else
                const float4 grad = grad_row4[j];
                float *weight = &weight_row[j * 4];
                atomicAdd(&weight[0], grad.x);
                atomicAdd(&weight[1], grad.y);
                atomicAdd(&weight[2], grad.z);
                atomicAdd(&weight[3], grad.w);
#endif
            }
            return;
        }
    }

    for (int j = threadIdx.x; j < embedding_dim; j += blockDim.x) { atomicAdd(&weight_row[j], grad_row[j]); }
}

std::shared_ptr<Tensor> EmbeddingBackward(const std::shared_ptr<Tensor> &input, const std::vector<int64_t> &weight_dims,
                                          const std::shared_ptr<Tensor> &grad_output) {
    CHECK(input->Dtype() == DataType::kINT64);
    CHECK_EQ(weight_dims.size(), 2);
    auto device = input->GetDevice();
    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();

    const int vocab_size = weight_dims[0];
    const int embedding_dim = weight_dims[1];
    CHECK_EQ(input->Dims().size() + 1, grad_output->Dims().size());
    for (int idx = 0; idx < input->Dims().size(); ++idx) { CHECK_EQ(input->Dims()[idx], grad_output->Dims()[idx]); }
    CHECK_EQ(*grad_output->Dims().rbegin(), embedding_dim);

    auto dtype = grad_output->Dtype();
    auto grad_weight = std::make_shared<Tensor>(weight_dims, dtype, grad_output->GetDevice());
    const int num_tokens = input->NumElements();
    const int threads_per_block = 256;
    // One block per token; each block cooperatively accumulates a whole gradient row.
    const int num_blocks = num_tokens;

    core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
        dtype,
        [=]<typename T>() {
            grad_weight->Fill(0.0);
            if (num_tokens > 0) {
                EmbeddingBackwardKernel<<<num_blocks, threads_per_block, 0, cuda_stream>>>(
                    static_cast<const int64_t *>(input->DataPtr()), static_cast<const T *>(grad_output->DataPtr()),
                    static_cast<T *>(grad_weight->DataPtr()), num_tokens, embedding_dim, vocab_size);
            }
        },
        "CUDA EmbeddingBackward");

    return grad_weight;
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_EMBEDDING_KERNEL(kernel_name)                                                                    \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_EMBEDDING_KERNEL(EmbeddingForward)
REGISTER_CUDA_EMBEDDING_KERNEL(EmbeddingBackward)

#undef REGISTER_CUDA_EMBEDDING_KERNEL
