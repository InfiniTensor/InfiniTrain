#include <cfloat>
#include <cmath>
#include <cstdint>
#include <memory>

#include "cublas_v2.h"
#include "glog/logging.h"
#include <curand_kernel.h>

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

// CUDA kernel for LogSoftmax forward pass with shared memory reduction
template <int BLOCK_SIZE>
__global__ void LogSoftmaxForwardKernel(const float *input, float *output, int outer, int axis, int inner) {
    int o = blockIdx.z;
    int i = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int tx = threadIdx.x;

    if (o >= outer || i >= inner) {
        return;
    }

    __shared__ float max_vals[BLOCK_SIZE];
    __shared__ float exp_sums[BLOCK_SIZE];

    // Calculate base offset for current outer and inner indices
    int base_offset = o * axis * inner + i;

    // Step 1: Find max value in the axis dimension
    float local_max = -FLT_MAX;
    for (int j = tx; j < axis; j += BLOCK_SIZE) {
        float val = input[base_offset + j * inner];
        if (val > local_max) {
            local_max = val;
        }
    }
    max_vals[threadIdx.x] = local_max;
    __syncthreads();

    // Reduce to find the maximum value in the block
    float block_max = local_max;
    for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            float other = max_vals[threadIdx.x + offset];
            if (other > block_max) {
                block_max = other;
            }
        }
        __syncthreads();
    }

    // Step 2: Compute exp(x - max) and sum
    float local_exp_sum = 0.0f;
    for (int j = tx; j < axis; j += BLOCK_SIZE) { local_exp_sum += expf(input[base_offset + j * inner] - block_max); }
    exp_sums[threadIdx.x] = local_exp_sum;
    __syncthreads();

    // Reduce to find the sum of exp(x - max) in the block
    float block_exp_sum = local_exp_sum;
    for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            block_exp_sum += exp_sums[threadIdx.x + offset];
        }
        __syncthreads();
    }

    // Step 3: Compute log(softmax)
    if (tx < axis) {
        float log_sum_exp = block_max + logf(block_exp_sum);
        output[base_offset + tx * inner] = input[base_offset + tx * inner] - log_sum_exp;
    }
}

// CUDA kernel for LogSoftmax backward pass
template <int BLOCK_SIZE>
__global__ void LogSoftmaxBackwardKernel(const float *grad_output, const float *output, float *grad_input, int outer,
                                         int axis, int inner) {
    int o = blockIdx.z;
    int i = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int tx = threadIdx.x;

    if (o >= outer || i >= inner) {
        return;
    }

    __shared__ float grad_sums[BLOCK_SIZE];

    // Calculate base offset for current outer and inner indices
    int base_offset = o * axis * inner + i;

    // Step 1: Sum gradients in the axis dimension
    float local_grad_sum = 0.0f;
    for (int j = tx; j < axis; j += BLOCK_SIZE) { local_grad_sum += grad_output[base_offset + j * inner]; }
    grad_sums[threadIdx.x] = local_grad_sum;
    __syncthreads();

    // Reduce to find the sum of gradients in the block
    float block_grad_sum = local_grad_sum;
    for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            block_grad_sum += grad_sums[threadIdx.x + offset];
        }
        __syncthreads();
    }

    // Step 2: Compute gradient input
    if (tx < axis) {
        float exp_output = expf(output[base_offset + tx * inner]);
        grad_input[base_offset + tx * inner] = grad_output[base_offset + tx * inner] - exp_output * block_grad_sum;
    }
}

std::shared_ptr<Tensor> LogSoftmaxForward(const std::shared_ptr<Tensor> &input, int64_t dim) {
    dim = dim < 0 ? input->Dims().size() + dim : dim;
    auto output = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());

    const float *input_data = static_cast<const float *>(input->DataPtr());
    float *output_data = static_cast<float *>(output->DataPtr());

    const auto &dims = input->Dims();
    int outer = 1;
    int axis = dims[dim];
    int inner = 1;

    for (int i = 0; i < dim; ++i) { outer *= dims[i]; }
    for (int i = dim + 1; i < dims.size(); ++i) { inner *= dims[i]; }

    // CUDA configuration
    constexpr int BLOCK_SIZE = 256;
    dim3 block(BLOCK_SIZE, 1, 1);
    dim3 grid((inner + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, outer);

    // Launch LogSoftmax forward kernel
    LogSoftmaxForwardKernel<BLOCK_SIZE><<<grid, block>>>(input_data, output_data, outer, axis, inner);
    CUDA_CHECK(cudaDeviceSynchronize());

    return output;
}

std::shared_ptr<Tensor> LogSoftmaxBackward(const std::shared_ptr<Tensor> &grad_output,
                                           const std::shared_ptr<Tensor> &output, int64_t dim) {
    dim = dim < 0 ? output->Dims().size() + dim : dim;
    auto grad_input = std::make_shared<Tensor>(output->Dims(), output->Dtype(), output->GetDevice());

    const float *grad_output_data = static_cast<const float *>(grad_output->DataPtr());
    const float *output_data = static_cast<const float *>(output->DataPtr());
    float *grad_input_data = static_cast<float *>(grad_input->DataPtr());

    const auto &dims = output->Dims();
    int outer = 1;
    int axis = dims[dim];
    int inner = 1;

    for (int i = 0; i < dim; ++i) { outer *= dims[i]; }
    for (int i = dim + 1; i < dims.size(); ++i) { inner *= dims[i]; }

    // CUDA configuration
    constexpr int BLOCK_SIZE = 256;
    dim3 block(BLOCK_SIZE, 1, 1);
    dim3 grid((inner + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, outer);

    // Launch LogSoftmax backward kernel
    LogSoftmaxBackwardKernel<BLOCK_SIZE>
        <<<grid, block>>>(grad_output_data, output_data, grad_input_data, outer, axis, inner);
    CUDA_CHECK(cudaDeviceSynchronize());

    return grad_input;
}

} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_LOGSOFTMAX_KERNEL(kernel_name)                                                                   \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_LOGSOFTMAX_KERNEL(LogSoftmaxForward)
REGISTER_CUDA_LOGSOFTMAX_KERNEL(LogSoftmaxBackward)

#undef REGISTER_CUDA_LOGSOFTMAX_KERNEL