#include <cmath>
#include <cstddef>

#include "cub/block/block_reduce.cuh"
#include "glog/logging.h"

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/common/cuda/kernel_helper.cuh"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {
template <size_t BLOCK_SIZE, typename T>
__global__ void SoftmaxForwardKernel(T *output, const T *input, int64_t outer_size, int64_t axis_size,
                                     int64_t inner_size) {
    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;

    __shared__ typename BlockReduce::TempStorage temp_storage_max;
    __shared__ typename BlockReduce::TempStorage temp_storage_sum;
    __shared__ float row_max;
    __shared__ float row_sum;

    const int64_t group = blockIdx.x;     // row of the grid
    const int64_t inner_idx = blockIdx.y; // column of the grid
    const int tid = threadIdx.x;

    // calculate the maximum for each group
    float thread_max = -INFINITY;
    for (int64_t axis = tid; axis < axis_size; axis += BLOCK_SIZE) {
        int64_t idx = (group * axis_size + axis) * inner_size + inner_idx;
        thread_max = max(thread_max, common::cuda::Cast<float>(input[idx]));
    }
    float block_max = BlockReduce(temp_storage_max).Reduce(thread_max, cub::Max());

    if (tid == 0) {
        row_max = block_max;
    }
    __syncthreads();

    // calculate the sum of exponents
    float thread_sum = 0;
    for (int64_t axis = tid; axis < axis_size; axis += BLOCK_SIZE) {
        int64_t idx = (group * axis_size + axis) * inner_size + inner_idx;
        float exp_val = exp(common::cuda::Cast<float>(input[idx]) - row_max);
        output[idx] = common::cuda::Cast<T>(exp_val);
        thread_sum += exp_val;
    }
    float block_sum = BlockReduce(temp_storage_sum).Sum(thread_sum);

    if (tid == 0) {
        row_sum = block_sum;
    }
    __syncthreads();

    // normalize
    for (int64_t axis = tid; axis < axis_size; axis += BLOCK_SIZE) {
        int64_t idx = (group * axis_size + axis) * inner_size + inner_idx;
        output[idx] = common::cuda::Cast<T>(common::cuda::Cast<float>(output[idx]) / row_sum);
    }
}

template <size_t BLOCK_SIZE, typename T>
void LaunchForward(const std::shared_ptr<Tensor> &output, const std::shared_ptr<Tensor> &input, int64_t dim) {
    const auto &input_dims = input->Dims();
    int64_t outer_size = 1;
    int64_t axis_size = input_dims[dim];
    int64_t inner_size = 1;

    for (int i = 0; i < dim; ++i) { outer_size *= input_dims[i]; };
    for (int i = dim + 1; i < input_dims.size(); ++i) { inner_size *= input_dims[i]; };
    if (axis_size == 0) {
        LOG_LOC(INFO, "CUDA softmax forward: 'input_dims[dim] == 0'");
        return;
    }
    if (outer_size == 0) {
        return;
    }

    T *output_ptr = static_cast<T *>(output->DataPtr());
    const T *input_ptr = static_cast<const T *>(input->DataPtr());

    if (BLOCK_SIZE > 1024) {
        LOG_LOC(FATAL, "CUDA softmax forward: 'BLOCK_SIZE used is larger than the max number of thread per block'");
    }
    dim3 block_dims(BLOCK_SIZE);
    dim3 grid_dims(outer_size, inner_size);

    const auto *cuda_device = dynamic_cast<const CudaDevice *>(output->GetDevice());
    SoftmaxForwardKernel<BLOCK_SIZE, T>
        <<<grid_dims, block_dims, 0, cuda_device->Stream()>>>(output_ptr, input_ptr, outer_size, axis_size, inner_size);
}

std::shared_ptr<Tensor> SoftmaxForward(const std::shared_ptr<Tensor> &input, int64_t dim) {
    auto dtype = input->Dtype();
    const auto &input_dims = input->Dims();
    dim = dim < 0 ? dim + input_dims.size() : dim;
    CHECK(dim >= 0 && dim < input_dims.size());
    auto output = std::make_shared<Tensor>(input_dims, dtype, input->GetDevice());

    switch (dtype) {
        DISPATCH_CASE(WRAP(LaunchForward<256, float>(output, input, dim);), DataType::kFLOAT32)
        DISPATCH_CASE(WRAP(LaunchForward<256, nv_bfloat16>(output, input, dim);), DataType::kBFLOAT16)
    default:
        LOG_LOC(FATAL, "CUDA softmax forward: 'Unsupported data type'");
    }
    return output;
}

template <size_t BLOCK_SIZE, typename T>
__global__ void SoftmaxBackwardKernel(T *grad_input, const T *grad_output, const T *output, int64_t outer_size,
                                      int64_t axis_size, int64_t inner_size) {
    using BlockReduce = cub::BlockReduce<T, BLOCK_SIZE>;

    __shared__ typename BlockReduce::TempStorage temp_storage_sum;
    __shared__ T row_sum;

    const int64_t group = blockIdx.x;
    const int64_t inner_idx = blockIdx.y;
    const int tid = threadIdx.x;

    // calculate the sum of the dot product of gradients
    T thread_sum = 0;
    for (int64_t axis = tid; axis < axis_size; axis += BLOCK_SIZE) {
        const int64_t idx = (group * axis_size + axis) * inner_size + inner_idx;
        thread_sum += grad_output[idx] * output[idx];
    }
    T block_sum = BlockReduce(temp_storage_sum).Sum(thread_sum);

    if (tid == 0) {
        row_sum = block_sum;
    }
    __syncthreads();

    // update the input gradient
    for (int64_t axis = tid; axis < axis_size; axis += BLOCK_SIZE) {
        const int64_t idx = (group * axis_size + axis) * inner_size + inner_idx;
        grad_input[idx] = output[idx] * (grad_output[idx] - row_sum);
    }
}

template <size_t BLOCK_SIZE, typename T>
void LaunchBackward(const std::shared_ptr<Tensor> &grad_input, const std::shared_ptr<Tensor> &grad_output,
                    const std::shared_ptr<Tensor> &output, int64_t dim) {
    const auto &output_dims = output->Dims();
    int64_t outer_size = 1;
    int64_t axis_size = output_dims[dim];
    int64_t inner_size = 1;

    for (int i = 0; i < dim; ++i) { outer_size *= output_dims[i]; };
    for (int i = dim + 1; i < output_dims.size(); ++i) { inner_size *= output_dims[i]; };
    if (axis_size == 0) {
        LOG_LOC(INFO, "CUDA softmax backward: 'output_dims[dim] == 0'");
        return;
    }
    if (outer_size == 0) {
        return;
    }

    T *grad_input_ptr = static_cast<T *>(grad_input->DataPtr());
    const T *grad_output_ptr = static_cast<const T *>(grad_output->DataPtr());
    const T *output_ptr = static_cast<const T *>(output->DataPtr());

    if (BLOCK_SIZE > 1024) {
        LOG_LOC(FATAL, "CUDA softmax backward: 'BLOCK_SIZE used is larger than the max number of thread per block'");
    }
    dim3 block(BLOCK_SIZE);
    dim3 grid(outer_size, inner_size);

    const auto *cuda_device = dynamic_cast<const CudaDevice *>(output->GetDevice());
    SoftmaxBackwardKernel<BLOCK_SIZE, T><<<grid, block, 0, cuda_device->Stream()>>>(
        grad_input_ptr, grad_output_ptr, output_ptr, outer_size, axis_size, inner_size);
}

std::shared_ptr<Tensor> SoftmaxBackward(const std::shared_ptr<Tensor> &grad_output,
                                        const std::shared_ptr<Tensor> &output, int64_t dim) {
    auto dtype = output->Dtype();
    const auto &output_dims = output->Dims();
    dim = dim < 0 ? dim + output->Dims().size() : dim;
    CHECK(dim >= 0 && dim < output->Dims().size());

    auto grad_input = std::make_shared<Tensor>(output_dims, dtype, output->GetDevice());
    grad_input->Fill<float>(0.0f);

    switch (dtype) {
        DISPATCH_CASE(WRAP(LaunchBackward<256, float>(grad_input, grad_output, output, dim);), DataType::kFLOAT32)
        DISPATCH_CASE(WRAP(LaunchBackward<256, nv_bfloat16>(grad_input, grad_output, output, dim);),
                      DataType::kBFLOAT16)
    default:
        LOG_LOC(FATAL, "CUDA softmax backward: 'Unsupported data type'");
    }

    return grad_input;
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_SOFTMAX_KERNEL(kernel_name)                                                                      \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_SOFTMAX_KERNEL(SoftmaxForward)
REGISTER_CUDA_SOFTMAX_KERNEL(SoftmaxBackward)

#undef REGISTER_CUDA_SOFTMAX_KERNEL
