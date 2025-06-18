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

// CUDA kernel for Dropout forward pass
template <typename T>
__global__ void DropoutForwardKernel(T *input, T *output, float *mask, int numel, float p, curandState_t *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) {
        return;
    }

    // Generate random number [0, 1)
    float r = curand_uniform(&states[idx]);
    float scale = 1.0f / (1.0f - p);

    if (r < p) {
        mask[idx] = 0.0f;
        output[idx] = 0.0f;
    } else {
        mask[idx] = scale;
        output[idx] = input[idx] * scale;
    }
}

// CUDA kernel for Dropout backward pass
template <typename T>
__global__ void DropoutBackwardKernel(T *grad_output, T *grad_input, const float *mask, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) {
        return;
    }

    grad_input[idx] = grad_output[idx] * mask[idx];
}

// CUDA kernel for generating curand states
__global__ void InitCurandStates(curandState_t *states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> DropoutForward(const std::shared_ptr<Tensor> input,
                                                                            float p, bool training, bool inplace) {
    CHECK(p >= 0.0f && p < 1.0f) << "Dropout probability must be in [0, 1)";
    CHECK(input != nullptr) << "Input tensor cannot be null";

    const auto &input_dims = input->Dims();
    const int64_t numel = input->NumElements();
    const Device &device = input->GetDevice();

    if (!training || p == 0.0f) {
        auto mask = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32, device);
        mask->Fill(1.0f);

        if (inplace) {
            return {input, mask};
        } else {
            auto output = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32, device);
            CUDA_CHECK(
                cudaMemcpy(output->DataPtr(), input->DataPtr(), numel * sizeof(float), cudaMemcpyDeviceToDevice));
            return {output, mask};
        }
    }

    const float scale = 1.0f / (1.0f - p);
    std::shared_ptr<Tensor> output;
    auto mask = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32, device);

    if (inplace) {
        output = input;
    } else {
        output = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32, device);
    }

    // Allocate memory for curand states
    curandState_t *states;
    CUDA_CHECK(cudaMalloc(&states, numel * sizeof(curandState_t)));

    // Initialize curand states with a seed (could be based on time or a fixed value)
    const unsigned long seed = 12345ULL;
    const int threads_per_block = 256;
    const int num_blocks = (numel + threads_per_block - 1) / threads_per_block;
    InitCurandStates<<<num_blocks, threads_per_block>>>(states, seed);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Launch dropout forward kernel
    if (input->Dtype() == DataType::kFLOAT32) {
        DropoutForwardKernel<float><<<num_blocks, threads_per_block>>>(
            static_cast<float *>(output->DataPtr()), static_cast<float *>(output->DataPtr()),
            static_cast<float *>(mask->DataPtr()), numel, p, states);
    } else if (input->Dtype() == DataType::kFLOAT16) {
        // For float16, we can either use native operations or convert to float32
        // This is a simplified example; a real implementation would handle float16 properly
        DropoutForwardKernel<float><<<num_blocks, threads_per_block>>>(
            reinterpret_cast<float *>(output->DataPtr()), reinterpret_cast<float *>(output->DataPtr()),
            static_cast<float *>(mask->DataPtr()), numel, p, states);
    } else {
        LOG(FATAL) << "Unsupported data type for Dropout";
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Free curand states
    CUDA_CHECK(cudaFree(states));

    return {output, mask};
}

std::shared_ptr<Tensor> DropoutBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &mask,
                                        float p, bool training, bool inplace) {
    CHECK(grad_output != nullptr) << "Gradient output tensor cannot be null";
    CHECK(mask != nullptr) << "Mask tensor cannot be null";
    CHECK(p >= 0.0f && p < 1.0f) << "Dropout probability must be in [0, 1)";

    const auto &grad_dims = grad_output->Dims();
    const int64_t numel = grad_output->NumElements();
    const auto &mask_dims = mask->Dims();
    CHECK(grad_dims == mask_dims) << "Gradient and mask dimension mismatch";
    const Device &device = grad_output->GetDevice();

    if (!training || p == 0.0f) {
        if (inplace) {
            return grad_output;
        } else {
            auto grad_input = std::make_shared<Tensor>(grad_dims, DataType::kFLOAT32, device);
            CUDA_CHECK(cudaMemcpy(grad_input->DataPtr(), grad_output->DataPtr(), numel * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
            return grad_input;
        }
    }

    std::shared_ptr<Tensor> grad_input;
    if (inplace) {
        grad_input = grad_output;
    } else {
        grad_input = std::make_shared<Tensor>(grad_dims, DataType::kFLOAT32, device);
    }

    const int threads_per_block = 256;
    const int num_blocks = (numel + threads_per_block - 1) / threads_per_block;

    // Launch dropout backward kernel
    if (grad_output->Dtype() == DataType::kFLOAT32) {
        DropoutBackwardKernel<float><<<num_blocks, threads_per_block>>>(
            static_cast<float *>(grad_output->DataPtr()), static_cast<float *>(grad_input->DataPtr()),
            static_cast<const float *>(mask->DataPtr()), numel);
    } else if (grad_output->Dtype() == DataType::kFLOAT16) {
        // For float16, we can either use native operations or convert to float32
        // This is a simplified example; a real implementation would handle float16 properly
        DropoutBackwardKernel<float><<<num_blocks, threads_per_block>>>(
            reinterpret_cast<float *>(grad_output->DataPtr()), reinterpret_cast<float *>(grad_input->DataPtr()),
            static_cast<const float *>(mask->DataPtr()), numel);
    } else {
        LOG(FATAL) << "Unsupported data type for Dropout backward";
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    return grad_input;
}

} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_DROPOUT_KERNEL(kernel_name)                                                                      \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_DROPOUT_KERNEL(DropoutForward)
REGISTER_CUDA_DROPOUT_KERNEL(DropoutBackward)

#undef REGISTER_CUDA_DROPOUT_KERNEL