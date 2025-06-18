#include "cublas_v2.h"
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

// CUDA kernel for ReLU forward pass
__global__ void ReLUForwardKernel(const float *input, float *output, float *mask, size_t num_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float val = input[idx];
        if (val > 0.0f) {
            output[idx] = val;
            mask[idx] = 1.0f;
        } else {
            output[idx] = 0.0f;
            mask[idx] = 0.0f;
        }
    }
}

// CUDA kernel for ReLU backward pass
__global__ void ReLUBackwardKernel(const float *grad_output, const float *mask, float *grad_input,
                                   size_t num_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        grad_input[idx] = grad_output[idx] * mask[idx];
    }
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> ReluForward(const std::shared_ptr<Tensor> &input) {
    // Save input dimensions
    const auto &input_dims = input->Dims();
    const size_t num_elements = input->NumElements();
    const Device &device = input->GetDevice();

    // Create output tensor and mask tensor
    auto output = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32, device);
    auto mask = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32, device);

    // Get data pointers
    const float *input_data = static_cast<const float *>(input->DataPtr());
    float *output_data = static_cast<float *>(output->DataPtr());
    float *mask_data = static_cast<float *>(mask->DataPtr());

    // CUDA configuration
    const int threads_per_block = 256;
    const int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    // Launch ReLU forward kernel
    ReLUForwardKernel<<<blocks_per_grid, threads_per_block>>>(input_data, output_data, mask_data, num_elements);
    CUDA_CHECK(cudaDeviceSynchronize());

    return {output, mask};
}

std::shared_ptr<Tensor> ReluBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &mask) {
    // Validate input dimensions
    const auto &go_dims = grad_output->Dims();
    const auto &mask_dims = mask->Dims();
    CHECK_EQ(go_dims.size(), mask_dims.size());

    const size_t num_elements = grad_output->NumElements();
    const Device &device = grad_output->GetDevice();

    // Create gradient tensor for input
    auto grad_input = std::make_shared<Tensor>(go_dims, DataType::kFLOAT32, device);
    float *grad_input_data = static_cast<float *>(grad_input->DataPtr());
    const float *go_data = static_cast<const float *>(grad_output->DataPtr());
    const float *mask_data = static_cast<const float *>(mask->DataPtr());

    // CUDA configuration
    const int threads_per_block = 256;
    const int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    // Launch ReLU backward kernel
    ReLUBackwardKernel<<<blocks_per_grid, threads_per_block>>>(go_data, mask_data, grad_input_data, num_elements);
    CUDA_CHECK(cudaDeviceSynchronize());

    return grad_input;
}

} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_RELU_KERNEL(kernel_name)                                                                         \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_RELU_KERNEL(ReluForward)
REGISTER_CUDA_RELU_KERNEL(ReluBackward)

#undef REGISTER_CUDA_RELU_KERNEL