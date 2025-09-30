#include <algorithm>
#include <numeric>
#include <vector>

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {
template <typename T>
__global__ void SplitForwardKernel(const T *input, T *output, int64_t N, int64_t H_in, int64_t H_out, int64_t W,
                                   int64_t start_idx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H_out * W;

    if (idx < total) {
        int w = idx % W;
        int h = (idx / W) % H_out;
        int n = idx / (H_out * W);

        int input_h = h + start_idx;
        int input_idx = n * H_in * W + input_h * W + w;
        int output_idx = n * H_out * W + h * W + w;

        output[output_idx] = input[input_idx];
    }
}

std::vector<std::shared_ptr<Tensor>> SplitForward(const std::shared_ptr<Tensor> &input, int64_t split_size, int dim) {
    CHECK_GT(split_size, 0);
    CHECK_GE(dim, 0) << "Currently we do not support negative dimension";
    const auto &input_dims = input->Dims();
    CHECK_LT(dim, input_dims.size());

    std::vector<std::shared_ptr<Tensor>> outputs;
    auto dtype = input->Dtype();

    const int64_t N = std::accumulate(input_dims.begin(), input_dims.begin() + dim, 1, std::multiplies<int64_t>());
    const int64_t W = std::accumulate(input_dims.begin() + dim + 1, input_dims.end(), 1, std::multiplies<int64_t>());
    const int64_t H_in = input_dims[dim];

    for (int64_t start = 0; start < H_in; start += split_size) {
        auto output_dims = input_dims;
        const int64_t H_out = std::min(split_size, H_in - start);
        output_dims[dim] = H_out;

        auto output = std::make_shared<Tensor>(output_dims, dtype, input->GetDevice());

        int64_t total = N * H_out * W;
        int threads_per_block = 256;
        int num_blocks = (total + threads_per_block - 1) / threads_per_block;

        const auto *cuda_device = dynamic_cast<const CudaDevice *>(input->GetDevice());

        DispatchFunc<INFINI_ALL_TYPES>(
            dtype,
            [=]<typename T>() {
                SplitForwardKernel<<<num_blocks, threads_per_block, 0, cuda_device->Stream()>>>(
                    static_cast<const T *>(input->DataPtr()), static_cast<T *>(output->DataPtr()), N, H_in, H_out, W,
                    start);
            },
            "CUDA SplitForward");

        outputs.push_back(std::move(output));
    }

    return outputs;
}

template <typename T>
__global__ void SplitBackwardKernel(const T *const *grad_outputs, T *grad_input, int64_t N, int64_t H_in, int64_t W,
                                    int64_t split_size, int64_t num_splits, const int64_t *H_outs) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = N * H_in * W;
    if (idx >= total) {
        return;
    }

    int64_t w = idx % W;
    int64_t h = (idx / W) % H_in;
    int64_t n = idx / (H_in * W);

    int64_t split_idx = h / split_size;
    if (split_idx >= num_splits) {
        return;
    }

    int64_t H_out = H_outs[split_idx];
    int64_t local_h = h - split_idx * split_size;

    if (local_h >= H_out) {
        return;
    }

    const T *grad_output = grad_outputs[split_idx];
    T value = grad_output[(n * H_out + local_h) * W + w];
    grad_input[(n * H_in + h) * W + w] = value;
}

template <typename T>
std::shared_ptr<Tensor> LaunchSplitBackward(const std::vector<int64_t> &input_dims, int64_t split_size, int dim,
                                            const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_GT(split_size, 0);
    CHECK_GE(dim, 0) << "Currently we do not support negative dimension";
    CHECK_LT(dim, input_dims.size());

    const auto &grad = grad_outputs[0];
    auto dtype = grad->Dtype();
    auto grad_input = std::make_shared<Tensor>(input_dims, dtype, grad->GetDevice());
    grad_input->Fill<T>(0);

    int64_t N = std::accumulate(input_dims.begin(), input_dims.begin() + dim, 1, std::multiplies<int64_t>());
    int64_t W = std::accumulate(input_dims.begin() + dim + 1, input_dims.end(), 1, std::multiplies<int64_t>());
    int64_t H_in = input_dims[dim];
    int64_t num_splits = grad_outputs.size();

    const auto *cuda_device = dynamic_cast<const CudaDevice *>(grad->GetDevice());
    const auto &stream = cuda_device->Stream();
    // init the array of grad_output ptrs
    std::vector<const T *> host_grad_output_ptrs;
    for (const auto &grad_output : grad_outputs) {
        host_grad_output_ptrs.push_back(static_cast<const T *>(grad_output->DataPtr()));
    }

    void *device_ptr;
    const T **device_grad_output_ptrs;
    int64_t *device_H_outs;
    cudaMallocAsync(&device_ptr, (sizeof(T *) + sizeof(int64_t)) * num_splits, stream);
    device_grad_output_ptrs = (const T **)(device_ptr);
    device_H_outs = reinterpret_cast<int64_t *>(device_grad_output_ptrs + num_splits);

    cudaMemcpyAsync(device_grad_output_ptrs, host_grad_output_ptrs.data(), sizeof(T *) * num_splits,
                    cudaMemcpyHostToDevice, stream);

    // init H_out for each split
    std::vector<int64_t> H_outs(num_splits);
    for (int i = 0; i < num_splits; ++i) { H_outs[i] = std::min(split_size, H_in - i * split_size); }

    cudaMemcpyAsync(device_H_outs, H_outs.data(), sizeof(int64_t) * num_splits, cudaMemcpyHostToDevice, stream);

    int64_t total_elements = N * H_in * W;
    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    SplitBackwardKernel<<<num_blocks, threads_per_block, 0, stream>>>(device_grad_output_ptrs,
                                                                      static_cast<T *>(grad_input->DataPtr()), N, H_in,
                                                                      W, split_size, num_splits, device_H_outs);

    cudaFreeAsync(device_ptr, stream);

    return grad_input;
}

std::shared_ptr<Tensor> SplitBackward(const std::vector<int64_t> &input_dims, int64_t split_size, int dim,
                                      const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_GT(split_size, 0);
    CHECK_GE(dim, 0) << "Currently we do not support negative dimension";
    CHECK_LT(dim, input_dims.size());

    return DispatchFunc<INFINI_ALL_TYPES>(
        grad_outputs[0]->Dtype(),
        [=]<typename T>() { return LaunchSplitBackward<T>(input_dims, split_size, dim, grad_outputs); },
        "CUDA SplitBackward");
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_SPLIT_KERNEL(kernel_name)                                                                        \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_SPLIT_KERNEL(SplitForward)
REGISTER_CUDA_SPLIT_KERNEL(SplitBackward)

#undef REGISTER_CUDA_SPLIT_KERNEL
