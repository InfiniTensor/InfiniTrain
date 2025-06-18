#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
// MaxPool2D forward pass with ceil mode
std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> MaxPool2DForward(const std::shared_ptr<Tensor> &input,
                                                                              int64_t kernel_size, int64_t stride,
                                                                              size_t padding, size_t dilation) {

    // Validate input tensor dimensions (must be 4D: NCHW)
    const auto &input_dims = input->Dims();
    CHECK_EQ(input_dims.size(), 4) << "Input must be 4D tensor (NCHW)";

    // Extract input dimensions
    const int64_t batch = input_dims[0];    // Batch size
    const int64_t channels = input_dims[1]; // Number of channels
    const int64_t input_h = input_dims[2];  // Input height
    const int64_t input_w = input_dims[3];  // Input width

    // Validate parameters
    CHECK_GT(kernel_size, 0) << "Kernel size must be positive";
    CHECK_GT(stride, 0) << "Stride must be positive";
    CHECK_GE(dilation, 1) << "Dilation must be at least 1";

    // Compute effective kernel size with dilation
    const int64_t effective_kernel_h = kernel_size + (kernel_size - 1) * (dilation - 1);
    const int64_t effective_kernel_w = effective_kernel_h; // assuming square kernel

    // Compute output spatial dimensions with CEILING MODE
    const int64_t numerator_h = input_h + 2 * padding - effective_kernel_h;
    const int64_t numerator_w = input_w + 2 * padding - effective_kernel_w;

    // Ceiling division: (numerator + stride - 1) / stride
    const int64_t output_h = (numerator_h + stride - 1) / stride + 1;
    const int64_t output_w = (numerator_w + stride - 1) / stride + 1;

    // Ensure valid output dimensions
    CHECK_GT(output_h, 0) << "Invalid output height (non-positive)";
    CHECK_GT(output_w, 0) << "Invalid output width (non-positive)";

    // Initialize output tensor (pooled values) and mask tensor
    std::vector<int64_t> output_dims = {batch, channels, output_h, output_w};
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32);
    auto mask = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);

    // Initialize tensors to 0
    output->Fill<float>(0.0f);
    mask->Fill<float>(0.0f);

    // Get raw pointers to tensor data
    const float *input_data = static_cast<const float *>(input->DataPtr());
    float *output_data = static_cast<float *>(output->DataPtr());
    float *mask_data = static_cast<float *>(mask->DataPtr());

    // Precompute sizes for indexing
    const int64_t channel_size = input_h * input_w;          // Size of each input channel (H*W)
    const int64_t output_channel_size = output_h * output_w; // Size of each output channel

    // Iterate over each sample in the batch
    for (int64_t n = 0; n < batch; ++n) {
        // Iterate over each channel
        for (int64_t c = 0; c < channels; ++c) {
            // Pointers to current input channel, output channel, and mask
            const float *channel_in = input_data + n * channels * channel_size + c * channel_size;
            float *channel_out = output_data + n * channels * output_channel_size + c * output_channel_size;
            float *channel_mask = mask_data + n * channels * channel_size + c * channel_size;

            // Slide pooling window over the input
            for (int64_t oh = 0; oh < output_h; ++oh) {
                for (int64_t ow = 0; ow < output_w; ++ow) {
                    // Calculate input window boundaries (with padding)
                    const int64_t ih_start = oh * stride - padding;
                    const int64_t iw_start = ow * stride - padding;
                    const int64_t ih_end = ih_start + effective_kernel_h;
                    const int64_t iw_end = iw_start + effective_kernel_w;

                    // Clamp to valid input region (handles padding)
                    const int64_t start_h = std::max(ih_start, (int64_t)0);
                    const int64_t start_w = std::max(iw_start, (int64_t)0);
                    const int64_t end_h = std::min(ih_end, input_h);
                    const int64_t end_w = std::min(iw_end, input_w);

                    // Skip if window is entirely outside input (due to padding)
                    if (start_h >= end_h || start_w >= end_w) {
                        continue;
                    }

                    // Find max value in current window (considering dilation)
                    float max_val = -std::numeric_limits<float>::infinity();
                    int64_t max_ih = -1, max_iw = -1;

                    for (int64_t kh = 0; kh < kernel_size; ++kh) {
                        for (int64_t kw = 0; kw < kernel_size; ++kw) {
                            const int64_t ih = start_h + kh * dilation;
                            const int64_t iw = start_w + kw * dilation;

                            // Check if within actual input bounds
                            if (ih < end_h && iw < end_w) {
                                const int64_t offset = ih * input_w + iw;
                                const float val = channel_in[offset];

                                if (val > max_val) {
                                    max_val = val;
                                    max_ih = ih;
                                    max_iw = iw;
                                }
                            }
                        }
                    }

                    // Store max value and update mask
                    if (max_ih != -1 && max_iw != -1) {
                        const int64_t out_offset = oh * output_w + ow;
                        channel_out[out_offset] = max_val;

                        const int64_t mask_offset = max_ih * input_w + max_iw;
                        channel_mask[mask_offset] = 1.0f; // Mark max location
                    }
                }
            }
        }
    }

    return {output, mask};
}

// MaxPool2D backward pass
std::shared_ptr<Tensor> MaxPool2DBackward(const std::shared_ptr<Tensor> &grad_output,
                                          const std::shared_ptr<Tensor> &mask, size_t kernel_size, size_t stride,
                                          size_t padding, size_t dilation) {

    // Validate input tensor dimensions (must be 4D: NCHW)
    const auto &go_dims = grad_output->Dims();
    const auto &mask_dims = mask->Dims();
    CHECK_EQ(go_dims.size(), 4) << "Gradient output must be 4D tensor (NCHW)";
    CHECK_EQ(mask_dims.size(), 4) << "Mask must be 4D tensor (NCHW)";

    // Ensure batch and channel dimensions match
    CHECK_EQ(go_dims[0], mask_dims[0]) << "Batch size mismatch";
    CHECK_EQ(go_dims[1], mask_dims[1]) << "Channel count mismatch";

    // Validate parameters
    CHECK_GT(kernel_size, 0) << "Kernel size must be positive";
    CHECK_GT(stride, 0) << "Stride must be positive";
    CHECK_GE(dilation, 1) << "Dilation must be at least 1";
    CHECK_GE(padding, 0) << "Padding must be non-negative";

    // Extract dimensions
    const int64_t batch = go_dims[0];     // Batch size
    const int64_t channels = go_dims[1];  // Number of channels
    const int64_t output_h = go_dims[2];  // Output height
    const int64_t output_w = go_dims[3];  // Output width
    const int64_t input_h = mask_dims[2]; // Input height (from mask)
    const int64_t input_w = mask_dims[3]; // Input width (from mask)

    // Initialize gradient input tensor with zeros
    std::vector<int64_t> grad_input_dims = {batch, channels, input_h, input_w};
    auto grad_input = std::make_shared<Tensor>(grad_input_dims, DataType::kFLOAT32);
    grad_input->Fill<float>(0.0f);

    // Get raw pointers to tensor data
    const float *go_data = static_cast<const float *>(grad_output->DataPtr());
    const float *mask_data = static_cast<const float *>(mask->DataPtr());
    float *gi_data = static_cast<float *>(grad_input->DataPtr());

    // Precompute sizes for indexing
    const int64_t go_channel_size = output_h * output_w; // Size of each output channel
    const int64_t mask_channel_size = input_h * input_w; // Size of each input channel

    // Iterate over each sample in the batch
    for (int64_t n = 0; n < batch; ++n) {
        // Iterate over each channel
        for (int64_t c = 0; c < channels; ++c) {
            // Get pointers to current channels
            const float *go_channel = go_data + n * channels * go_channel_size + c * go_channel_size;
            const float *mask_channel = mask_data + n * channels * mask_channel_size + c * mask_channel_size;
            float *gi_channel = gi_data + n * channels * mask_channel_size + c * mask_channel_size;

            // Process each output position
            for (int64_t oh = 0; oh < output_h; ++oh) {
                for (int64_t ow = 0; ow < output_w; ++ow) {
                    // Get gradient value at current output position
                    const int64_t go_offset = oh * output_w + ow;
                    const float grad_val = go_channel[go_offset];

                    // Skip if gradient is zero (optimization)
                    if (grad_val == 0.0f) {
                        continue;
                    }

                    // Calculate input window boundaries (with padding and dilation)
                    const int64_t ih_center = oh * stride;
                    const int64_t iw_center = ow * stride;

                    // Iterate over kernel positions with dilation
                    for (int64_t kh = 0; kh < kernel_size; ++kh) {
                        for (int64_t kw = 0; kw < kernel_size; ++kw) {
                            const int64_t ih = ih_center - padding + kh * dilation;
                            const int64_t iw = iw_center - padding + kw * dilation;

                            // Check if position is within input bounds
                            if (ih >= 0 && ih < input_h && iw >= 0 && iw < input_w) {
                                const int64_t mask_offset = ih * input_w + iw;

                                // If this position was the max in forward pass, accumulate gradient
                                if (mask_channel[mask_offset] > 0.5f) {
                                    gi_channel[mask_offset] += grad_val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return grad_input;
}
} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_MAXPOOL_KERNEL(kernel_name)                                                                       \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_MAXPOOL_KERNEL(MaxPool2DForward);
REGISTER_CPU_MAXPOOL_KERNEL(MaxPool2DBackward);

#undef REGISTER_CPU_MAXPOOL_KERNEL