#include <memory>

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/core/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"
#include "infini_train/src/core/cuda/cuda_stream.h"

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

    DispatchFunc<INFINI_ALL_FLOATING_TYPES>(
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
    const int num_blocks = (num_tokens + threads_per_block - 1) / threads_per_block;

    DispatchFunc<INFINI_ALL_FLOATING_TYPES>(
        dtype,
        [=]<typename T>() {
            grad_weight->Fill<T>(0);
            EmbeddingBackwardKernel<<<num_blocks, threads_per_block, 0, cuda_stream>>>(
                static_cast<const int64_t *>(input->DataPtr()), static_cast<const T *>(grad_output->DataPtr()),
                static_cast<T *>(grad_weight->DataPtr()), num_tokens, embedding_dim, vocab_size);
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
