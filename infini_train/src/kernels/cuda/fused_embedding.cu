#include "infini_train/include/device.h"
#include "infini_train/include/kernels/cuda/fused_embedding.h"

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

__global__ void FusedEmbeddingForwardKernel(const uint16_t *input, float *output, const float *wte, const float *wpe,
                                            int batch_size, int max_seqlen, int embed_dim) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= batch_size * max_seqlen * embed_dim) {
        return;
    }

    int bt = idx / embed_dim;
    int b = bt / max_seqlen;
    int t = bt % max_seqlen;
    int c = idx % embed_dim;

    int ix = static_cast<int>(input[b * max_seqlen + t]);

    output[b * max_seqlen * embed_dim + t * embed_dim + c] = wte[ix * embed_dim + c] + wpe[t * embed_dim + c];
}

std::shared_ptr<Tensor> FusedEmbeddingForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &wte,
                                              const std::shared_ptr<Tensor> &wpe) {
    CHECK_EQ(input->Dims().size(), 2);
    CHECK_EQ(wte->Dims().size(), 2);
    CHECK_EQ(wpe->Dims().size(), 2);
    CHECK_LE(input->Dims()[1], wpe->Dims()[0]);
    CHECK_EQ(wte->Dims()[1], wpe->Dims()[1]);

    const int batch_size = input->Dims()[0];
    const int max_seqlen = wpe->Dims()[0];
    const int embed_dim = wpe->Dims()[1];

    auto output = std::make_shared<Tensor>(std::vector<int64_t>{batch_size, max_seqlen, embed_dim}, DataType::kFLOAT32,
                                           Device(DeviceType::kCUDA, 0));

    int threads_per_block = 256;
    int num_blocks = (batch_size * max_seqlen * embed_dim + threads_per_block - 1) / threads_per_block;
    FusedEmbeddingForwardKernel<<<num_blocks, threads_per_block>>>(
        reinterpret_cast<const uint16_t *>(input->DataPtr()), reinterpret_cast<float *>(output->DataPtr()),
        reinterpret_cast<const float *>(wte->DataPtr()), reinterpret_cast<const float *>(wpe->DataPtr()), batch_size,
        max_seqlen, embed_dim);
    return {output};
}

__global__ void WTEBackwardKernel(float *grad_wte, const float *grad_output, const uint16_t *input, int batch_size,
                                  int max_seqlen, int embed_dim) {
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

    atomicAdd(&grad_wte[token * embed_dim + c], grad);
}

__global__ void WPEBackwardKernel(float *grad_wpe, const float *grad_output, const uint16_t *input, int batch_size,
                                  int max_seqlen, int embed_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_seqlen * embed_dim) {
        return;
    }

    int t = idx / embed_dim;
    int c = idx % embed_dim;
    float accum = 0.0f;

    for (int b = 0; b < batch_size; b++) { accum += grad_output[b * max_seqlen * embed_dim + t * embed_dim + c]; }

    atomicAdd(&grad_wpe[t * embed_dim + c], accum);
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
FusedEmbeddingBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &wte,
                       const std::shared_ptr<Tensor> &wpe, const std::shared_ptr<Tensor> &grad_output) {
    CHECK_EQ(input->Dims().size(), 2);
    CHECK_EQ(wte->Dims().size(), 2);
    CHECK_EQ(wpe->Dims().size(), 2);
    CHECK_LE(input->Dims()[1], wpe->Dims()[0]);
    CHECK_EQ(wte->Dims()[1], wpe->Dims()[1]);

    const int batch_size = input->Dims()[0];
    const int max_seqlen = wpe->Dims()[0];
    const int embed_dim = wpe->Dims()[1];

    auto grad_input = std::make_shared<Tensor>(input->Dims(), DataType::kFLOAT32, Device(DeviceType::kCUDA, 0));
    auto grad_wte = std::make_shared<Tensor>(wte->Dims(), DataType::kFLOAT32, Device(DeviceType::kCUDA, 0));
    auto grad_wpe = std::make_shared<Tensor>(wpe->Dims(), DataType::kFLOAT32, Device(DeviceType::kCUDA, 0));
    grad_wte->Fill<float>(0.0f);
    grad_wpe->Fill<float>(0.0f);

    int threads_per_block = 256;
    int num_blocks = ((max_seqlen * embed_dim) + threads_per_block - 1) / threads_per_block;
    WPEBackwardKernel<<<num_blocks, threads_per_block>>>(
        reinterpret_cast<float *>(grad_wpe->DataPtr()), reinterpret_cast<const float *>(grad_output->DataPtr()),
        reinterpret_cast<const uint16_t *>(input->DataPtr()), batch_size, max_seqlen, embed_dim);

    num_blocks = ((batch_size * max_seqlen) + threads_per_block - 1) / threads_per_block;
    WTEBackwardKernel<<<num_blocks, threads_per_block>>>(
        reinterpret_cast<float *>(grad_wte->DataPtr()), reinterpret_cast<const float *>(grad_output->DataPtr()),
        reinterpret_cast<const uint16_t *>(input->DataPtr()), batch_size, max_seqlen, embed_dim);

    return {grad_input, grad_wte, grad_wpe};
}
} // namespace infini_train::kernels::cuda
