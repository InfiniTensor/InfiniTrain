#include <algorithm>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

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

__global__ void StackForwardKernel(const float **inputs, float *output, int64_t N, int64_t D, int64_t num_inputs) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = N * num_inputs * D;

    if (idx >= total) {
        return;
    }

    int64_t d = idx % D;
    int64_t s = (idx / D) % num_inputs;
    int64_t n = idx / (D * num_inputs);

    const float *input = inputs[s];
    output[idx] = input[n * D + d];
}

std::shared_ptr<Tensor> StackForward(const std::vector<std::shared_ptr<Tensor>> &inputs, int64_t dim) {
    CHECK(!inputs.empty());

    const auto &base_dims = inputs[0]->Dims();
    if (dim < 0) {
        dim += base_dims.size() + 1;
    }
    CHECK_GE(dim, 0);
    CHECK_LE(dim, base_dims.size());
    for (const auto &input : inputs) { CHECK(input->Dims() == base_dims); }

    std::vector<int64_t> out_dims = base_dims;
    out_dims.insert(out_dims.begin() + dim, inputs.size());
    auto output = std::make_shared<Tensor>(out_dims, DataType::kFLOAT32, inputs[0]->GetDevice());

    const int64_t N = std::accumulate(base_dims.begin(), base_dims.begin() + dim, 1, std::multiplies<int64_t>());
    const int64_t D = std::accumulate(base_dims.begin() + dim, base_dims.end(), 1, std::multiplies<int64_t>());
    const int64_t num_inputs = inputs.size();

    std::vector<const float *> host_input_ptrs;
    for (const auto &t : inputs) { host_input_ptrs.push_back(static_cast<const float *>(t->DataPtr())); }

    const auto *cuda_device = dynamic_cast<const CudaDevice *>(output->GetDevice());
    const auto &stream = cuda_device->Stream();

    const float **device_input_ptrs;
    cudaMallocAsync(&device_input_ptrs, sizeof(float *) * num_inputs, stream);
    cudaMemcpyAsync(device_input_ptrs, host_input_ptrs.data(), sizeof(float *) * num_inputs, cudaMemcpyHostToDevice,
                    stream);

    int64_t total = N * num_inputs * D;
    int threads_per_block = 256;
    int num_blocks = (total + threads_per_block - 1) / threads_per_block;
    StackForwardKernel<<<num_blocks, threads_per_block, 0, stream>>>(
        device_input_ptrs, static_cast<float *>(output->DataPtr()), N, D, num_inputs);

    cudaFreeAsync(device_input_ptrs, stream);
    return output;
}

__global__ void StackBackwardKernel(const float *grad_output, float **grad_inputs, int64_t N, int64_t D,
                                    int64_t num_inputs) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = N * num_inputs * D;

    if (idx >= total) {
        return;
    }

    int64_t d = idx % D;
    int64_t s = (idx / D) % num_inputs;
    int64_t n = idx / (D * num_inputs);

    if (s < num_inputs) {
        grad_inputs[s][n * D + d] = grad_output[idx];
    }
}

std::vector<std::shared_ptr<Tensor>> StackBackward(const std::vector<int64_t> &input_dims, int64_t dim,
                                                   const std::shared_ptr<Tensor> &grad_output) {
    if (dim < 0) {
        dim += input_dims.size() + 1;
    }
    const int64_t num_inputs = grad_output->Dims()[dim];
    std::vector<int64_t> base_dims = grad_output->Dims();
    base_dims.erase(base_dims.begin() + dim);

    std::vector<std::shared_ptr<Tensor>> grads;
    for (int i = 0; i < num_inputs; ++i) {
        auto t = std::make_shared<Tensor>(base_dims, DataType::kFLOAT32, grad_output->GetDevice());
        t->Fill<float>(0.0f);
        grads.push_back(t);
    }

    int64_t N = std::accumulate(input_dims.begin(), input_dims.begin() + dim, 1, std::multiplies<int64_t>());
    int64_t D = std::accumulate(input_dims.begin() + dim, input_dims.end(), 1, std::multiplies<int64_t>());

    std::vector<float *> host_ptrs;
    for (auto &t : grads) { host_ptrs.push_back(static_cast<float *>(t->DataPtr())); }

    const auto *cuda_device = dynamic_cast<const CudaDevice *>(grad_output->GetDevice());
    const auto &stream = cuda_device->Stream();

    float **device_ptrs;
    cudaMallocAsync(&device_ptrs, sizeof(float *) * num_inputs, stream);
    cudaMemcpyAsync(device_ptrs, host_ptrs.data(), sizeof(float *) * num_inputs, cudaMemcpyHostToDevice, stream);

    int64_t total = N * num_inputs * D;
    int threads_per_block = 256;
    int num_blocks = (total + threads_per_block - 1) / threads_per_block;

    StackBackwardKernel<<<num_blocks, threads_per_block, 0, stream>>>(
        static_cast<const float *>(grad_output->DataPtr()), device_ptrs, N, D, num_inputs);

    cudaFreeAsync(device_ptrs, stream);
    return grads;
}

} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_STACK_KERNEL(kernel_name)                                                                        \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_STACK_KERNEL(StackForward)
REGISTER_CUDA_STACK_KERNEL(StackBackward)

#undef REGISTER_CUDA_STACK_KERNEL
