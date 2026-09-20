#include <memory>

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/sparse_row_grad.h"
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

template <typename T>
__global__ void EmbeddingBackwardKernel(const int64_t *input_ptr, const T *grad_output_ptr, T *grad_weight_ptr,
                                        int num_tokens, int embedding_dim, int vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tokens) {
        return;
    }

    int token_id = static_cast<int>(input_ptr[idx]);
    if (token_id < 0 || token_id >= vocab_size) {
        return;
    }

    for (int j = 0; j < embedding_dim; ++j) {
        atomicAdd(&grad_weight_ptr[token_id * embedding_dim + j], grad_output_ptr[idx * embedding_dim + j]);
    }
}

// Sparse backward, part 1: scatter-add the per-token grad rows into the persistent gradient
// buffer and claim (deduplicate) every hit row in row_list, so later passes can touch exactly the
// rows that are dirty instead of the whole [vocab, dim] tensor.
// grid = (num_tokens, ceil(dim / blockDim.x)): block (t, c) covers the c-th chunk of token t. The
// first thread of every c == 0 block claims row tokens[t] with a CAS against the stamp array, so
// each row lands in row_list exactly once per accumulation cycle (generation); every thread then
// scatter-adds its element. Claim and scatter write disjoint memory and both complete before any
// later kernel (Adam, clear) observes them.
template <typename T>
__global__ void EmbeddingBackwardSparseKernel(const int64_t *__restrict__ tokens, const T *__restrict__ grad_output,
                                              T *__restrict__ grad_buffer, int32_t *__restrict__ stamp,
                                              int32_t *__restrict__ row_list, int32_t *__restrict__ count, int gen,
                                              int vocab_size, int embedding_dim) {
    const int64_t token = tokens[blockIdx.x];
    if (token < 0 || token >= vocab_size) {
        return;
    }
    const int row = static_cast<int>(token);

    if (blockIdx.y == 0 && threadIdx.x == 0) {
        int old = stamp[row];
        while (old != gen) {
            const int prev = atomicCAS(&stamp[row], old, gen);
            if (prev == old) {
                row_list[atomicAdd(count, 1)] = row;
                break;
            }
            old = prev;
        }
    }

    const int elem = blockIdx.y * blockDim.x + threadIdx.x;
    if (elem < embedding_dim) {
        atomicAdd(&grad_buffer[static_cast<size_t>(row) * embedding_dim + elem],
                  grad_output[static_cast<size_t>(blockIdx.x) * embedding_dim + elem]);
    }
}

// Sparse backward, part 2 (driven by the optimizer): zero exactly the rows claimed since the last
// clear. The row count lives on the device, so the host never needs to synchronize to size this.
template <typename T>
__global__ void SparseRowClearRowsKernel(T *__restrict__ grad_buffer, const int32_t *__restrict__ row_list,
                                         const int32_t *__restrict__ count, int embedding_dim) {
    const int num_rows = *count;
    for (int r = blockIdx.x; r < num_rows; r += gridDim.x) {
        T *row_base = grad_buffer + static_cast<size_t>(row_list[r]) * embedding_dim;
        for (int e = threadIdx.x; e < embedding_dim; e += blockDim.x) { row_base[e] = static_cast<T>(0.0f); }
    }
}

__global__ void SparseRowResetCountKernel(int32_t *count) { *count = 0; }

void SparseRowClearRows(const std::shared_ptr<Tensor> &grad_buffer, const std::shared_ptr<Tensor> &row_list,
                        const std::shared_ptr<Tensor> &count) {
    auto device = grad_buffer->GetDevice();
    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();
    const int embedding_dim = static_cast<int>(grad_buffer->Dims()[1]);

    constexpr int kThreadsPerBlock = 256;
    constexpr int kNumBlocks = 1024; // grid-strided; idle blocks only read *count and exit
    core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
        grad_buffer->Dtype(),
        [=]<typename T>() {
            SparseRowClearRowsKernel<T><<<kNumBlocks, kThreadsPerBlock, 0, cuda_stream>>>(
                static_cast<T *>(grad_buffer->DataPtr()), static_cast<const int32_t *>(row_list->DataPtr()),
                static_cast<const int32_t *>(count->DataPtr()), embedding_dim);
        },
        "CUDA SparseRowClearRows");
}

void SparseRowResetCount(const std::shared_ptr<Tensor> &count) {
    auto device = count->GetDevice();
    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();
    SparseRowResetCountKernel<<<1, 1, 0, cuda_stream>>>(static_cast<int32_t *>(count->DataPtr()));
}

std::shared_ptr<Tensor> EmbeddingBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                                          const std::shared_ptr<Tensor> &grad_output) {
    CHECK(input->Dtype() == DataType::kINT64);
    CHECK_EQ(weight->Dims().size(), 2);
    auto device = input->GetDevice();
    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();

    const int64_t vocab_size = weight->Dims()[0];
    const int64_t embedding_dim = weight->Dims()[1];
    CHECK_EQ(input->Dims().size() + 1, grad_output->Dims().size());
    for (int idx = 0; idx < input->Dims().size(); ++idx) { CHECK_EQ(input->Dims()[idx], grad_output->Dims()[idx]); }
    CHECK_EQ(*grad_output->Dims().rbegin(), embedding_dim);
    const int64_t num_tokens = input->NumElements();

    // Dtype mismatch (a grad produced outside autocast against a cast weight): the sparse buffer
    // lives in the weight's dtype, so fall back to the legacy dense grad for that rare case.
    if (grad_output->Dtype() != weight->Dtype()) {
        auto grad_weight = std::make_shared<Tensor>(weight->Dims(), grad_output->Dtype(), grad_output->GetDevice());
        const int threads_per_block = 256;
        const int num_blocks = (num_tokens + threads_per_block - 1) / threads_per_block;
        core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
            grad_output->Dtype(),
            [=]<typename T>() {
                grad_weight->Fill(0.0);
                EmbeddingBackwardKernel<<<num_blocks, threads_per_block, 0, cuda_stream>>>(
                    static_cast<const int64_t *>(input->DataPtr()), static_cast<const T *>(grad_output->DataPtr()),
                    static_cast<T *>(grad_weight->DataPtr()), static_cast<int>(num_tokens),
                    static_cast<int>(embedding_dim), static_cast<int>(vocab_size));
            },
            "CUDA EmbeddingBackward");
        return grad_weight;
    }

    auto *state = SparseRowGradRegistry::Instance().GetOrCreate(weight);
    if (!state->initialized) {
        // One-time init: the "zero everywhere except rows claimed since the last clear" invariant
        // has to start from an actually zero buffer. Everything afterwards only touches dirty rows.
        state->grad_buffer->Fill(0.0);
        cudaMemsetAsync(state->stamp->DataPtr(), 0, state->stamp->NumElements() * sizeof(int32_t), cuda_stream);
        cudaMemsetAsync(state->count->DataPtr(), 0, sizeof(int32_t), cuda_stream);
        state->initialized = true;
    }

    if (num_tokens > 0) {
        constexpr int kThreadsPerBlock = 256;
        const dim3 grid(static_cast<unsigned>(num_tokens),
                        static_cast<unsigned>((embedding_dim + kThreadsPerBlock - 1) / kThreadsPerBlock));
        core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
            weight->Dtype(),
            [=]<typename T>() {
                EmbeddingBackwardSparseKernel<T><<<grid, kThreadsPerBlock, 0, cuda_stream>>>(
                    static_cast<const int64_t *>(input->DataPtr()), static_cast<const T *>(grad_output->DataPtr()),
                    static_cast<T *>(state->grad_buffer->DataPtr()), static_cast<int32_t *>(state->stamp->DataPtr()),
                    static_cast<int32_t *>(state->row_list->DataPtr()), static_cast<int32_t *>(state->count->DataPtr()),
                    state->generation, static_cast<int>(vocab_size), static_cast<int>(embedding_dim));
            },
            "CUDA EmbeddingBackward");
    }

    // A view sharing the persistent buffer: AccumulateGrad notices that the grad storage is the
    // scattered buffer and skips its own accumulation, so these rows are what ends up in
    // param->grad() — with no per-call allocation or fill of a vocab-sized tensor.
    return std::make_shared<Tensor>(*state->grad_buffer.get(), 0, state->grad_buffer->Dims());
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_EMBEDDING_KERNEL(kernel_name)                                                                    \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_EMBEDDING_KERNEL(EmbeddingForward)
REGISTER_CUDA_EMBEDDING_KERNEL(EmbeddingBackward)
REGISTER_CUDA_EMBEDDING_KERNEL(SparseRowClearRows)
REGISTER_CUDA_EMBEDDING_KERNEL(SparseRowResetCount)

#undef REGISTER_CUDA_EMBEDDING_KERNEL
