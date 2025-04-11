#include "infini_train/include/device.h"
#include "infini_train/include/kernels/cuda/embedding.h"

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

__global__ void EmbeddingForwardKernel(const uint16_t *input, float *output, const float *weight, int batch_size,
                                       int max_seqlen, int embed_dim) {
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
    CHECK_EQ(input->Dims().size(), 2);
    CHECK_EQ(weight->Dims().size(), 2);

    const int batch_size = input->Dims()[0];
    const int max_seqlen = input->Dims()[1];
    const int embed_dim = weight->Dims()[1];

    auto output = std::make_shared<Tensor>(std::vector<int64_t>{batch_size, max_seqlen, embed_dim}, DataType::kFLOAT32,
                                           Device(DeviceType::kCUDA, 0));

    int threads_per_block = 256;
    int num_blocks = (batch_size * max_seqlen * embed_dim + threads_per_block - 1) / threads_per_block;
    EmbeddingForwardKernel<<<num_blocks, threads_per_block>>>(
        reinterpret_cast<const uint16_t *>(input->DataPtr()), reinterpret_cast<float *>(output->DataPtr()),
        reinterpret_cast<const float *>(weight->DataPtr()), batch_size, max_seqlen, embed_dim);

    return {output};
}

__global__ void WeightBackwardKernel(float *grad_weight, const float *grad_output, const uint16_t *input,
                                     int batch_size, int max_seqlen, int embed_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * max_seqlen) {
        return;
    }

    int token = static_cast<int>(input[idx]);
    if (token < 0) {
        return;
    }

    int c = threadIdx.x % embed_dim;
    float grad = grad_output[idx * embed_dim + c];

    atomicAdd(&grad_weight[token * embed_dim + c], grad);
}

std::shared_ptr<Tensor> EmbeddingBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                                          const std::shared_ptr<Tensor> &grad_output) {
    CHECK_EQ(input->Dims().size(), 2);
    CHECK_EQ(weight->Dims().size(), 2);

    const int batch_size = input->Dims()[0];
    const int max_seqlen = input->Dims()[1];
    const int embed_dim = weight->Dims()[1];

    auto grad_weight = std::make_shared<Tensor>(weight->Dims(), DataType::kFLOAT32, Device(DeviceType::kCUDA, 0));
    grad_weight->Fill<float>(0.0f);

    int threads_per_block = 256;
    int num_blocks = ((batch_size * max_seqlen * embed_dim) + threads_per_block - 1) / threads_per_block;

    WeightBackwardKernel<<<num_blocks, threads_per_block>>>(
        reinterpret_cast<float *>(grad_weight->DataPtr()), reinterpret_cast<const float *>(grad_output->DataPtr()),
        reinterpret_cast<const uint16_t *>(input->DataPtr()), batch_size, max_seqlen, embed_dim);

    return {grad_weight};
}
} // namespace infini_train::kernels::cuda
