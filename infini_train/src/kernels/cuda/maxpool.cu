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

// CUDA kernel for MaxPool2D forward pass with shared memory
template <int TILE_SIZE>
__global__ void MaxPool2DForwardKernel(const float *input, float *output, float *mask, int batch, int channels,
                                       int input_h, int input_w, int kernel_size, int stride, int padding,
                                       int dilation) {
    // Calculate effective kernel size with dilation
    const int effective_kernel_h = kernel_size + (kernel_size - 1) * (dilation - 1);
    const int effective_kernel_w = effective_kernel_h;

    // Get batch, channel, and spatial indices
    int b = blockIdx.z;
    int c = blockIdx.y;
    int oh = blockIdx.x / TILE_SIZE;
    int ow_tile = blockIdx.x % TILE_SIZE;

    // Thread indices within the tile
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int ow = ow_tile * TILE_SIZE + tx;

    if (b >= batch || c >= channels || oh >= (input_h + 2 * padding - effective_kernel_h + stride - 1) / stride + 1
        || ow >= (input_w + 2 * padding - effective_kernel_w + stride - 1) / stride + 1) {
        return;
    }

    // Shared memory for tile data
    __shared__ float tile_data[TILE_SIZE + 2 * TILE_SIZE][TILE_SIZE + 2 * TILE_SIZE];
    __shared__ int max_indices[TILE_SIZE + 2 * TILE_SIZE][TILE_SIZE + 2 * TILE_SIZE];

    // Calculate input window boundaries
    int ih_start = oh * stride - padding;
    int iw_start = ow * stride - padding;
    int ih_end = ih_start + effective_kernel_h;
    int iw_end = iw_start + effective_kernel_w;

    // Clamp to valid input region
    int start_h = max(ih_start, 0);
    int start_w = max(iw_start, 0);
    int end_h = min(ih_end, input_h);
    int end_w = min(iw_end, input_w);

    // Skip if window is outside input
    if (start_h >= end_h || start_w >= end_w) {
        return;
    }

    // Initialize max values
    float max_val = -FLT_MAX;
    int max_ih = -1, max_iw = -1;

    // Load input data into shared memory
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int ih = start_h + kh * dilation;
            int iw = start_w + kw * dilation;

            if (ih < end_h && iw < end_w) {
                int input_offset = b * channels * input_h * input_w + c * input_h * input_w + ih * input_w + iw;
                tile_data[kh][kw] = input[input_offset];

                if (tile_data[kh][kw] > max_val) {
                    max_val = tile_data[kh][kw];
                    max_ih = ih;
                    max_iw = iw;
                }
            } else {
                tile_data[kh][kw] = -FLT_MAX;
            }
        }
    }
    __syncthreads();

    // Update max indices in shared memory
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            if (tile_data[kh][kw] == max_val) {
                max_indices[kh][kw] = 1;
            } else {
                max_indices[kh][kw] = 0;
            }
        }
    }
    __syncthreads();

    // Write output and mask
    if (ty < TILE_SIZE && tx < TILE_SIZE) {
        int out_h = oh * TILE_SIZE + ty;
        int out_w = ow;

        if (out_h < (input_h + 2 * padding - effective_kernel_h + stride - 1) / stride + 1
            && out_w < (input_w + 2 * padding - effective_kernel_w + stride - 1) / stride + 1) {

            int output_offset = b * channels * ((input_h + 2 * padding - effective_kernel_h + stride - 1) / stride + 1)
                                  * ((input_w + 2 * padding - effective_kernel_w + stride - 1) / stride + 1)
                              + c * ((input_h + 2 * padding - effective_kernel_h + stride - 1) / stride + 1)
                                    * ((input_w + 2 * padding - effective_kernel_w + stride - 1) / stride + 1)
                              + out_h * ((input_w + 2 * padding - effective_kernel_w + stride - 1) / stride + 1)
                              + out_w;

            output[output_offset] = max_val;

            if (max_ih != -1 && max_iw != -1) {
                int mask_offset = b * channels * input_h * input_w + c * input_h * input_w + max_ih * input_w + max_iw;
                mask[mask_offset] = 1.0f;
            }
        }
    }
}

// CUDA kernel for MaxPool2D backward pass
template <int TILE_SIZE>
__global__ void MaxPool2DBackwardKernel(const float *grad_output, const float *mask, float *grad_input, int batch,
                                        int channels, int input_h, int input_w, int output_h, int output_w,
                                        int kernel_size, int stride, int padding, int dilation) {
    // Get batch, channel, and spatial indices
    int b = blockIdx.z;
    int c = blockIdx.y;
    int ih = blockIdx.x / TILE_SIZE;
    int iw_tile = blockIdx.x % TILE_SIZE;

    // Thread indices within the tile
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int iw = iw_tile * TILE_SIZE + tx;

    if (b >= batch || c >= channels || ih >= input_h || iw >= input_w) {
        return;
    }

    // Calculate effective kernel size with dilation
    const int effective_kernel_h = kernel_size + (kernel_size - 1) * (dilation - 1);
    const int effective_kernel_w = effective_kernel_h;

    // Shared memory for gradient and mask data
    __shared__ float grad_tile[TILE_SIZE + 2 * TILE_SIZE][TILE_SIZE + 2 * TILE_SIZE];
    __shared__ float mask_tile[TILE_SIZE + 2 * TILE_SIZE][TILE_SIZE + 2 * TILE_SIZE];

    // Calculate output window that includes current input position
    int oh_start = max((ih + padding - 0) / stride, 0);
    int ow_start = max((iw + padding - 0) / stride, 0);
    int oh_end = min((ih + padding + effective_kernel_h - 1) / stride + 1, output_h);
    int ow_end = min((iw + padding + effective_kernel_w - 1) / stride + 1, output_w);

    // Initialize gradient sum
    float grad_sum = 0.0f;

    // Load gradient and mask data into shared memory
    for (int oh = oh_start; oh < oh_end; ++oh) {
        for (int ow = ow_start; ow < ow_end; ++ow) {
            int go_offset = b * channels * output_h * output_w + c * output_h * output_w + oh * output_w + ow;
            grad_tile[oh - oh_start][ow - ow_start] = grad_output[go_offset];

            int mask_offset = b * channels * input_h * input_w + c * input_h * input_w + ih * input_w + iw;
            mask_tile[oh - oh_start][ow - ow_start] = mask[mask_offset];
        }
    }
    __syncthreads();

    // Accumulate gradient
    for (int oh = 0; oh < (oh_end - oh_start); ++oh) {
        for (int ow = 0; ow < (ow_end - ow_start); ++ow) {
            if (mask_tile[oh][ow] > 0.5f) {
                grad_sum += grad_tile[oh][ow];
            }
        }
    }
    __syncthreads();

    // Write gradient to input
    if (ty < TILE_SIZE && tx < TILE_SIZE) {
        int final_ih = ih * TILE_SIZE + ty;
        int final_iw = iw;

        if (final_ih < input_h && final_iw < input_w) {
            int gi_offset = b * channels * input_h * input_w + c * input_h * input_w + final_ih * input_w + final_iw;
            grad_input[gi_offset] = grad_sum;
        }
    }
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> MaxPool2DForward(const std::shared_ptr<Tensor> &input,
                                                                              int64_t kernel_size, int64_t stride,
                                                                              size_t padding, size_t dilation) {
    // Validate input dimensions
    const auto &input_dims = input->Dims();
    CHECK_EQ(input_dims.size(), 4) << "Input must be 4D tensor (NCHW)";

    // Extract dimensions
    const int64_t batch = input_dims[0];
    const int64_t channels = input_dims[1];
    const int64_t input_h = input_dims[2];
    const int64_t input_w = input_dims[3];

    // Validate parameters
    CHECK_GT(kernel_size, 0) << "Kernel size must be positive";
    CHECK_GT(stride, 0) << "Stride must be positive";
    CHECK_GE(dilation, 1) << "Dilation must be at least 1";

    // Calculate effective kernel size
    const int64_t effective_kernel_h = kernel_size + (kernel_size - 1) * (dilation - 1);
    const int64_t effective_kernel_w = effective_kernel_h;

    // Calculate output dimensions (ceil mode)
    const int64_t output_h = (input_h + 2 * padding - effective_kernel_h + stride - 1) / stride + 1;
    const int64_t output_w = (input_w + 2 * padding - effective_kernel_w + stride - 1) / stride + 1;

    // Ensure valid output dimensions
    CHECK_GT(output_h, 0) << "Invalid output height";
    CHECK_GT(output_w, 0) << "Invalid output width";

    // Create output and mask tensors
    std::vector<int64_t> output_dims = {batch, channels, output_h, output_w};
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32, input->GetDevice());
    auto mask = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32, input->GetDevice());

    // Initialize tensors
    output->Fill<float>(0.0f);
    mask->Fill<float>(0.0f);

    // Get data pointers
    const float *input_data = static_cast<const float *>(input->DataPtr());
    float *output_data = static_cast<float *>(output->DataPtr());
    float *mask_data = static_cast<float *>(mask->DataPtr());

    // CUDA configuration
    constexpr int TILE_SIZE = 16;
    dim3 block(TILE_SIZE, TILE_SIZE, 1);
    dim3 grid((output_h * output_w + TILE_SIZE - 1) / TILE_SIZE, channels, batch);

    // Launch forward kernel
    MaxPool2DForwardKernel<TILE_SIZE><<<grid, block>>>(input_data, output_data, mask_data, batch, channels, input_h,
                                                       input_w, kernel_size, stride, padding, dilation);
    CUDA_CHECK(cudaDeviceSynchronize());

    return {output, mask};
}

std::shared_ptr<Tensor> MaxPool2DBackward(const std::shared_ptr<Tensor> &grad_output,
                                          const std::shared_ptr<Tensor> &mask, size_t kernel_size, size_t stride,
                                          size_t padding, size_t dilation) {
    // Validate input dimensions
    const auto &go_dims = grad_output->Dims();
    const auto &mask_dims = mask->Dims();
    CHECK_EQ(go_dims.size(), 4) << "Gradient output must be 4D tensor";
    CHECK_EQ(mask_dims.size(), 4) << "Mask must be 4D tensor";

    // Ensure batch and channel dimensions match
    CHECK_EQ(go_dims[0], mask_dims[0]) << "Batch size mismatch";
    CHECK_EQ(go_dims[1], mask_dims[1]) << "Channel count mismatch";

    // Extract dimensions
    const int64_t batch = go_dims[0];
    const int64_t channels = go_dims[1];
    const int64_t output_h = go_dims[2];
    const int64_t output_w = go_dims[3];
    const int64_t input_h = mask_dims[2];
    const int64_t input_w = mask_dims[3];

    // Validate parameters
    CHECK_GT(kernel_size, 0) << "Kernel size must be positive";
    CHECK_GT(stride, 0) << "Stride must be positive";
    CHECK_GE(dilation, 1) << "Dilation must be at least 1";
    CHECK_GE(padding, 0) << "Padding must be non-negative";

    // Create gradient input tensor
    std::vector<int64_t> grad_input_dims = {batch, channels, input_h, input_w};
    auto grad_input = std::make_shared<Tensor>(grad_input_dims, DataType::kFLOAT32, grad_output->GetDevice());
    grad_input->Fill<float>(0.0f);

    // Get data pointers
    const float *go_data = static_cast<const float *>(grad_output->DataPtr());
    const float *mask_data = static_cast<const float *>(mask->DataPtr());
    float *gi_data = static_cast<float *>(grad_input->DataPtr());

    // CUDA configuration
    constexpr int TILE_SIZE = 16;
    dim3 block(TILE_SIZE, TILE_SIZE, 1);
    dim3 grid((input_h * input_w + TILE_SIZE - 1) / TILE_SIZE, channels, batch);

    // Launch backward kernel
    MaxPool2DBackwardKernel<TILE_SIZE><<<grid, block>>>(go_data, mask_data, gi_data, batch, channels, input_h, input_w,
                                                        output_h, output_w, kernel_size, stride, padding, dilation);
    CUDA_CHECK(cudaDeviceSynchronize());

    return grad_input;
}

} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_MAXPOOL_KERNEL(kernel_name)                                                                      \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_MAXPOOL_KERNEL(MaxPool2DForward);
REGISTER_CUDA_MAXPOOL_KERNEL(MaxPool2DBackward);

#undef REGISTER_CUDA_MAXPOOL_KERNEL