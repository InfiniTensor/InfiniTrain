#include "cublas_v2.h"
#include "glog/logging.h"
#include <cub/block/block_reduce.cuh>

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

#define CUBLAS_CHECK(call)                                                                                             \
    do {                                                                                                               \
        cublasStatus_t status = call;                                                                                  \
        if (status != CUBLAS_STATUS_SUCCESS) {                                                                         \
            LOG(FATAL) << "CUBLAS Error: " << cublasGetStatusString(status) << " at " << __FILE__ << ":" << __LINE__;  \
        }                                                                                                              \
    } while (0)

// CUDA kernel with shared memory optimization for convolution
template <int TILE_SIZE, int GROUP_SIZE>
__global__ void Conv2DKernelSharedMem(const float *input, const float *kernel, float *output, int batch_size,
                                      int in_channels, int input_rows, int input_cols, int out_channels,
                                      int kernel_size, int stride, int padding, int dilation, int groups,
                                      int output_rows, int output_cols) {
    // Calculate batch, output channel and spatial position for current thread
    int b = blockIdx.z;
    int oc = blockIdx.y * TILE_SIZE + threadIdx.y;
    int oh = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (b >= batch_size || oc >= out_channels || oh >= output_rows) {
        return;
    }

    // Grouped convolution parameters
    int group = oc / (out_channels / groups);
    int group_channels = in_channels / groups;
    int group_start = group * group_channels;

    // Shared memory for caching input and kernel data
    __shared__ float input_tile[TILE_SIZE + 2 * TILE_SIZE][TILE_SIZE + 2 * TILE_SIZE];
    __shared__ float kernel_tile[TILE_SIZE][TILE_SIZE * GROUP_SIZE];

    float conv_sum = 0.0f;
    int group_channel = threadIdx.z; // Channel index within the group

    if (group_channel < group_channels) {
        // Calculate actual input channel within the current group
        int ic = group_start + group_channel;

        // Iterate over kernel spatial positions
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                // Map to input feature map position
                int ih = oh * stride + kh * dilation - padding;
                int iw = threadIdx.x * stride + kw * dilation - padding;

                // Check if within input boundaries
                if (ih >= 0 && ih < input_rows && iw >= 0 && iw < input_cols) {
                    // Calculate input data offset
                    int input_offset = b * in_channels * input_rows * input_cols + ic * input_rows * input_cols
                                     + ih * input_cols + iw;
                    input_tile[threadIdx.y][threadIdx.x] = input[input_offset];
                } else {
                    input_tile[threadIdx.y][threadIdx.x] = 0.0f; // Padding with 0
                }

                // Calculate kernel data offset
                int kernel_offset = oc * group_channels * kernel_size * kernel_size
                                  + group_channel * kernel_size * kernel_size + kh * kernel_size + kw;
                kernel_tile[threadIdx.y][threadIdx.x * GROUP_SIZE + group_channel] = kernel[kernel_offset];
            }
        }
        __syncthreads();

        // Perform convolution calculation
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                conv_sum += input_tile[kh][kw] * kernel_tile[kh][kw * GROUP_SIZE + group_channel];
            }
        }
    }
    __syncthreads();

    // Use block reduction to sum
    using BlockReduce = cub::BlockReduce<float, TILE_SIZE * TILE_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float sum = BlockReduce(temp_storage).Sum(conv_sum);

    // Write to output
    if (threadIdx.x < TILE_SIZE && threadIdx.y < TILE_SIZE) {
        int ow = blockIdx.x * TILE_SIZE + threadIdx.x;
        if (ow < output_cols) {
            int output_offset
                = b * out_channels * output_rows * output_cols + oc * output_rows * output_cols + oh * output_cols + ow;
            output[output_offset] = sum;
        }
    }
}

std::shared_ptr<Tensor> Conv2DForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &kernel,
                                      const std::shared_ptr<Tensor> &bias, const size_t kernel_size,
                                      const int64_t in_channels, const int64_t out_channels, const int64_t stride,
                                      const int64_t padding, const int64_t dilation, const int64_t groups,
                                      const bool use_bias) {
    // Verify input dimensions
    const auto &input_dims = input->Dims();
    const auto &kernel_dims = kernel->Dims();
    CHECK_EQ(input_dims.size(), 4) << "Input must be 4D tensor";
    CHECK_EQ(kernel_dims.size(), 4) << "Kernel must be 4D tensor";

    const auto batch_size = input_dims[0];
    const auto input_channels = input_dims[1];
    const auto input_rows = input_dims[2];
    const auto input_cols = input_dims[3];
    CHECK_EQ(input_channels, in_channels) << "Input channel count mismatch";

    CHECK_EQ(kernel_dims[0], out_channels) << "Kernel output channel mismatch";
    CHECK_EQ(kernel_dims[1], in_channels / groups) << "Kernel input channel mismatch";
    CHECK_EQ(kernel_dims[2], kernel_size) << "Kernel height mismatch";
    CHECK_EQ(kernel_dims[3], kernel_size) << "Kernel width mismatch";

    // Calculate output dimensions
    const int64_t output_rows = (input_rows + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int64_t output_cols = (input_cols + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    CHECK_GT(output_rows, 0) << "Invalid output height";
    CHECK_GT(output_cols, 0) << "Invalid output width";

    // Create output tensor
    auto output = std::make_shared<Tensor>(std::vector<int64_t>{batch_size, out_channels, output_rows, output_cols},
                                           DataType::kFLOAT32, input->GetDevice());
    output->Fill(0.0f);

    // Get data pointers
    float *output_data = static_cast<float *>(output->DataPtr());
    const float *input_data = static_cast<const float *>(input->DataPtr());
    const float *kernel_data = static_cast<const float *>(kernel->DataPtr());
    const float *bias_data = use_bias ? static_cast<const float *>(bias->DataPtr()) : nullptr;

    // CUDA configuration
    constexpr int TILE_SIZE = 16;
    // constexpr int BLOCK_SIZE = TILE_SIZE * TILE_SIZE;
    dim3 block(TILE_SIZE, TILE_SIZE, 1); // Each block processes a TILE_SIZExTILE_SIZE output region
    dim3 grid((output_cols + TILE_SIZE - 1) / TILE_SIZE, (out_channels + TILE_SIZE - 1) / TILE_SIZE, batch_size);

    // Execute convolution kernel
    Conv2DKernelSharedMem<TILE_SIZE, 4><<<grid, block>>>(input_data, kernel_data, output_data, batch_size, in_channels,
                                                         input_rows, input_cols, out_channels, kernel_size, stride,
                                                         padding, dilation, groups, output_rows, output_cols);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Add bias if enabled
    if (use_bias) {
        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));

        const float alpha = 1.0f;
        // const float beta = 1.0f;
        int n = out_channels * output_rows * output_cols;

        for (int b = 0; b < batch_size; b++) {
            CUBLAS_CHECK(cublasSaxpy(handle, n, &alpha, bias_data, 1, output_data + b * n, 1));
        }

        CUBLAS_CHECK(cublasDestroy(handle));
    }

    return output;
}

// CUDA kernel for computing input gradient
template <int TILE_SIZE>
__global__ void Conv2DBackwardInputKernel(const float *grad_output, const float *grad_kernel, float *grad_input,
                                          int batch_size, int in_channels, int input_rows, int input_cols,
                                          int out_channels, int kernel_size, int stride, int padding, int dilation,
                                          int groups, int output_rows, int output_cols) {
    int b = blockIdx.z;
    int ic = blockIdx.y;
    int ih = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (b >= batch_size || ic >= in_channels || ih >= input_rows) {
        return;
    }

    // Grouped convolution parameters
    int group = ic / (in_channels / groups);
    int group_channels = in_channels / groups;
    int group_start = group * group_channels;
    int oc_per_group = out_channels / groups;

    // Shared memory
    __shared__ float grad_output_tile[TILE_SIZE + 2 * TILE_SIZE][TILE_SIZE + 2 * TILE_SIZE];
    __shared__ float grad_kernel_tile[TILE_SIZE][TILE_SIZE];

    float grad_sum = 0.0f;
    int ow = threadIdx.y; // Output column index

    for (int oc_group = 0; oc_group < oc_per_group; oc_group++) {
        int oc = group * oc_per_group + oc_group;
        int kc = ic - group_start; // Kernel channel index within the group

        if (kc >= 0 && kc < group_channels) {
            // Iterate over kernel spatial positions
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    // Map to output feature map position
                    int oh = (ih - kh * dilation + padding) / stride;
                    if (oh < 0 || oh >= output_rows) {
                        continue;
                    }

                    int iw = (ow - kw * dilation + padding) / stride;
                    if (iw < 0 || iw >= output_cols) {
                        continue;
                    }

                    // Fill shared memory
                    int go_offset = b * out_channels * output_rows * output_cols + oc * output_rows * output_cols
                                  + oh * output_cols + iw;
                    grad_output_tile[kh][kw] = grad_output[go_offset];

                    int gk_offset = oc * group_channels * kernel_size * kernel_size + kc * kernel_size * kernel_size
                                  + kh * kernel_size + kw;
                    grad_kernel_tile[kh][kw] = grad_kernel[gk_offset];
                }
            }
            __syncthreads();

            // Calculate gradient
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    grad_sum += grad_output_tile[kh][kw] * grad_kernel_tile[kh][kw];
                }
            }
        }
    }
    __syncthreads();

    // Write to input gradient
    if (ow < input_cols) {
        int gi_offset = b * in_channels * input_rows * input_cols + ic * input_rows * input_cols + ih * input_cols + ow;
        grad_input[gi_offset] = grad_sum;
    }
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
Conv2DBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &input,
               const size_t kernel_size, const int64_t in_channels, const int64_t out_channels, const int64_t stride,
               const int64_t padding, const int64_t dilation, const int64_t groups, const bool use_bias) {
    // Verify input dimensions
    const auto &input_dims = input->Dims();
    CHECK_EQ(input_dims.size(), 4) << "Input must be 4D tensor";

    const auto batch_size = input_dims[0];
    const auto input_channels = input_dims[1];
    const auto input_rows = input_dims[2];
    const auto input_cols = input_dims[3];
    CHECK_EQ(input_channels, in_channels) << "Input channel mismatch";

    const auto &grad_output_dims = grad_output->Dims();
    CHECK_EQ(grad_output_dims.size(), 4) << "Gradient output must be 4D tensor";
    const auto grad_output_channels = grad_output_dims[1];
    const auto grad_output_rows = grad_output_dims[2];
    const auto grad_output_cols = grad_output_dims[3];
    CHECK_EQ(grad_output_channels, out_channels) << "Gradient output channel mismatch";

    // Create kernel gradient tensor
    auto grad_kernel = std::make_shared<Tensor>(
        std::vector<int64_t>{out_channels, in_channels / groups, (int64_t)kernel_size, (int64_t)kernel_size},
        DataType::kFLOAT32, grad_output->GetDevice());
    grad_kernel->Fill(0.0f);

    // Create bias gradient tensor
    std::shared_ptr<Tensor> grad_bias = nullptr;
    if (use_bias) {
        grad_bias = std::make_shared<Tensor>(std::vector<int64_t>{out_channels}, DataType::kFLOAT32,
                                             grad_output->GetDevice());
        grad_bias->Fill(0.0f);
    }

    // Create input gradient tensor
    auto grad_input = std::make_shared<Tensor>(std::vector<int64_t>{batch_size, in_channels, input_rows, input_cols},
                                               DataType::kFLOAT32, grad_output->GetDevice());
    grad_input->Fill(0.0f);

    // Get data pointers
    const float *grad_output_data = static_cast<const float *>(grad_output->DataPtr());
    const float *input_data = static_cast<const float *>(input->DataPtr());
    float *grad_kernel_data = static_cast<float *>(grad_kernel->DataPtr());
    float *grad_input_data = static_cast<float *>(grad_input->DataPtr());

    // Use CUBLAS to compute kernel gradient (optimized version)
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Adjust dimensions for matrix multiplication
    int batch = batch_size;
    int m = grad_output_rows * grad_output_cols;
    int k = kernel_size * kernel_size;
    int n = in_channels / groups;

    // Reshape tensors for matrix multiplication
    auto grad_output_reshaped = std::make_shared<Tensor>(std::vector<int64_t>{batch, out_channels, m},
                                                         DataType::kFLOAT32, grad_output->GetDevice());
    auto input_reshaped = std::make_shared<Tensor>(std::vector<int64_t>{batch, in_channels, input_rows * input_cols},
                                                   DataType::kFLOAT32, input->GetDevice());

    // Data rearrangement (actual project needs to implement rearrangement function)
    // ...

    // Perform matrix multiplication to compute kernel gradient
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, out_channels, m, &alpha,
                                            input_reshaped->DataPtr(), CUDA_R_32F, input_rows * input_cols,
                                            in_channels * input_rows * input_cols, grad_output_reshaped->DataPtr(),
                                            CUDA_R_32F, m, out_channels * m, &beta, grad_kernel_data, CUDA_R_32F, k,
                                            out_channels * k, batch, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));

    // Compute bias gradient (sum)
    if (use_bias) {
        // constexpr int BLOCK_SIZE = 256;
        // int threads_per_block = BLOCK_SIZE;
        // int num_blocks = (out_channels + threads_per_block - 1) / threads_per_block;

        // Parallel sum (actual implementation requires kernel function)
        // ...
    }

    // Compute input gradient using CUDA kernel
    constexpr int TILE_SIZE = 16;
    dim3 block(TILE_SIZE, TILE_SIZE, 1);
    dim3 grid((input_rows + TILE_SIZE - 1) / TILE_SIZE, in_channels, batch_size);

    Conv2DBackwardInputKernel<TILE_SIZE><<<grid, block>>>(
        grad_output_data, grad_kernel_data, grad_input_data, batch_size, in_channels, input_rows, input_cols,
        out_channels, kernel_size, stride, padding, dilation, groups, grad_output_rows, grad_output_cols);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUBLAS_CHECK(cublasDestroy(handle));
    return {grad_input, grad_kernel, grad_bias};
}

} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_CONVOLUTION_KERNEL(kernel_name)                                                                  \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_CONVOLUTION_KERNEL(Conv2DForward)
REGISTER_CUDA_CONVOLUTION_KERNEL(Conv2DBackward)

#undef REGISTER_CUDA_CONVOLUTION_KERNEL