#include "glog/logging.h"
#include <cuda_bf16.h>

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

template <typename T>
__global__ void EmbeddingForwardKernel(const int64_t *input, T *output, const T *weight, int batch_size, int max_seqlen,
                                       int embed_dim) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= batch_size * max_seqlen * embed_dim) {
        return;
    }

    int bt = idx / embed_dim;
    int b = bt / max_seqlen;
    int t = bt % max_seqlen;
    int c = idx % embed_dim;

    int ix = static_cast<int>(input[b * max_seqlen + t]);

    output[b * max_seqlen * embed_dim + t * embed_dim + c] = weight[ix * embed_dim + c];
}

std::shared_ptr<Tensor> EmbeddingForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight) {
    CHECK(input->Dtype() == DataType::kINT64);
    CHECK_EQ(weight->Dims().size(), 2);

    const int batch_size = input->Dims().size() == 2 ? input->Dims()[0] : 1;
    const int max_seqlen = input->Dims().size() == 2 ? input->Dims()[1] : input->Dims()[0];
    const int embed_dim = weight->Dims()[1];
    auto output_dims = input->Dims();
    output_dims.push_back(embed_dim);

    auto dtype = weight->Dtype();
    auto output = std::make_shared<Tensor>(output_dims, dtype, input->GetDevice());
    int threads_per_block = 256;
    int num_blocks = (batch_size * max_seqlen * embed_dim + threads_per_block - 1) / threads_per_block;

    switch (dtype) {
        DISPATCH_CASE(DataType::kFLOAT32,
                      WRAP(EmbeddingForwardKernel<<<num_blocks, threads_per_block>>>(
                               static_cast<const int64_t *>(input->DataPtr()), static_cast<float *>(output->DataPtr()),
                               static_cast<const float *>(weight->DataPtr()), batch_size, max_seqlen, embed_dim);))
        DISPATCH_CASE(
            DataType::kBFLOAT16,
            WRAP(EmbeddingForwardKernel<<<num_blocks, threads_per_block>>>(
                     static_cast<const int64_t *>(input->DataPtr()), static_cast<nv_bfloat16 *>(output->DataPtr()),
                     static_cast<const nv_bfloat16 *>(weight->DataPtr()), batch_size, max_seqlen, embed_dim);))
    default:
        LOG(FATAL) << "CUDA Embedding forward: 'Unsupported data type' at " << __FILE__ << ":" << __LINE__;
    }

    return output;
}

template <typename T>
__global__ void EmbeddingBackwardKernel(const int64_t *input_ptr, const T *grad_output_ptr, T *grad_weight_ptr,
                                        int num_tokens, int embedding_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tokens) {
        return;
    }

    int token_id = static_cast<int>(input_ptr[idx]);
    if (token_id < 0) {
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
    const int embedding_dim = weight_dims[1];
    CHECK_EQ(input->Dims().size() + 1, grad_output->Dims().size());
    for (int idx = 0; idx < input->Dims().size(); ++idx) { CHECK_EQ(input->Dims()[idx], grad_output->Dims()[idx]); }
    CHECK_EQ(*grad_output->Dims().rbegin(), embedding_dim);

    auto dtype = grad_output->Dtype();
    auto grad_weight = std::make_shared<Tensor>(weight_dims, dtype, grad_output->GetDevice());
    const int num_tokens = input->NumElements();
    const int threads_per_block = 256;
    const int num_blocks = (num_tokens + threads_per_block - 1) / threads_per_block;

    switch (dtype) {
        DISPATCH_CASE(DataType::kFLOAT32, WRAP({
                          grad_weight->Fill<float>(0.0f);
                          EmbeddingBackwardKernel<<<num_blocks, threads_per_block>>>(
                              static_cast<const int64_t *>(input->DataPtr()),
                              static_cast<const float *>(grad_output->DataPtr()),
                              static_cast<float *>(grad_weight->DataPtr()), num_tokens, embedding_dim);
                      }))
        DISPATCH_CASE(DataType::kBFLOAT16, WRAP({
                          grad_weight->Fill<nv_bfloat16>(0);
                          EmbeddingBackwardKernel<<<num_blocks, threads_per_block>>>(
                              static_cast<const int64_t *>(input->DataPtr()),
                              static_cast<const nv_bfloat16 *>(grad_output->DataPtr()),
                              static_cast<nv_bfloat16 *>(grad_weight->DataPtr()), num_tokens, embedding_dim);
                      }))
    default:
        LOG(FATAL) << "CUDA Embedding backward: 'Unsupported data type' at " << __FILE__ << ":" << __LINE__;
    }

    return grad_weight;
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_EMBEDDING_KERNEL(kernel_name)                                                                    \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_EMBEDDING_KERNEL(EmbeddingForward)
REGISTER_CUDA_EMBEDDING_KERNEL(EmbeddingBackward)

#undef REGISTER_CUDA_EMBEDDING_KERNEL
