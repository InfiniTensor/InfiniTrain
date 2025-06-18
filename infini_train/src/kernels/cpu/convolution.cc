#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
// Forward pass for 2D convolution with optional bias
std::shared_ptr<Tensor> Conv2DForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &kernel,
                                      const std::shared_ptr<Tensor> &bias, // Bias tensor
                                      const size_t kernel_size, const int64_t in_channels, const int64_t out_channels,
                                      const int64_t stride, const int64_t padding, const int64_t dilation,
                                      const int64_t groups,
                                      const bool use_bias) { // Flag to enable/disable bias
    // Verify input tensor dimensions: [batch_size, in_channels, height, width]
    const auto &input_dims = input->Dims();
    const auto &kernel_dims = kernel->Dims();

    CHECK_EQ(input_dims.size(), 4) << "Input must be 4D tensor";
    CHECK_EQ(kernel_dims.size(), 4) << "Kernel must be 4D tensor";

    // Extract dimensions from input tensor
    const auto batch_size = input_dims[0];
    const auto input_channels = input_dims[1];
    const auto input_rows = input_dims[2];
    const auto input_cols = input_dims[3];
    CHECK_EQ(input_channels, in_channels) << "Input channel count mismatch";

    // Verify kernel dimensions: [out_channels, in_channels/groups, kernel_size, kernel_size]
    CHECK_EQ(kernel_dims[0], out_channels) << "Kernel output channel mismatch";
    CHECK_EQ(kernel_dims[1], in_channels / groups) << "Kernel input channel mismatch";
    CHECK_EQ(kernel_dims[2], kernel_size) << "Kernel height mismatch";
    CHECK_EQ(kernel_dims[3], kernel_size) << "Kernel width mismatch";

    // Validate bias tensor if used
    if (use_bias) {
        CHECK(bias != nullptr) << "Bias tensor is null when use_bias is true";
        const auto &bias_dims = bias->Dims();
        CHECK_EQ(bias_dims.size(), 1) << "Bias must be 1D tensor";
        CHECK_EQ(bias_dims[0], out_channels) << "Bias size must match output channels";
    }

    // Calculate output dimensions using convolution formula
    const int64_t output_rows = (input_rows + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int64_t output_cols = (input_cols + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    CHECK_GT(output_rows, 0) << "Invalid output height";
    CHECK_GT(output_cols, 0) << "Invalid output width";

    // Create output tensor: [batch_size, out_channels, output_rows, output_cols]
    auto output = std::make_shared<Tensor>(std::vector<int64_t>{batch_size, out_channels, output_rows, output_cols},
                                           DataType::kFLOAT32);
    output->Fill(0.0f); // Initialize with zeros

    // Get raw data pointers
    auto output_data = static_cast<float *>(output->DataPtr());
    auto input_data = static_cast<const float *>(input->DataPtr());
    auto kernel_data = static_cast<const float *>(kernel->DataPtr());
    const float *bias_data = use_bias ? static_cast<const float *>(bias->DataPtr()) : nullptr;

    // Precompute strides for efficient memory access (Channel First format)
    const int64_t input_batch_stride = in_channels * input_rows * input_cols; // Stride between batches
    const int64_t input_channel_stride = input_rows * input_cols;             // Stride between channels
    const int64_t input_row_stride = input_cols;                              // Stride between rows

    const int64_t kernel_spatial_stride = kernel_size * kernel_size; // Total kernel elements
    const int64_t kernel_channel_stride
        = (in_channels / groups) * kernel_spatial_stride;            // Stride per input channel group
    const int64_t kernel_out_channel_stride = kernel_channel_stride; // Stride per output channel

    const int64_t output_batch_stride = out_channels * output_rows * output_cols; // Stride between batches
    const int64_t output_channel_stride = output_rows * output_cols;              // Stride between channels
    const int64_t output_row_stride = output_cols;                                // Stride between rows

    // Process each sample in the batch
    for (int64_t b = 0; b < batch_size; ++b) {
        const float *current_input = input_data + b * input_batch_stride;
        float *current_output = output_data + b * output_batch_stride;

        // Process each output spatial position
        for (int64_t oh = 0; oh < output_rows; ++oh) {
            for (int64_t ow = 0; ow < output_cols; ++ow) {

                // Process each output channel
                for (int64_t oc = 0; oc < out_channels; ++oc) {
                    float conv_sum = 0.0f; // Convolution accumulator

                    // Determine group parameters for grouped convolution
                    const int64_t group = oc / (out_channels / groups);  // Current group index
                    const int64_t group_channels = in_channels / groups; // Channels per group
                    const int64_t group_start = group * group_channels;  // Start channel index for group

                    // Process each kernel position
                    for (int64_t kh = 0; kh < kernel_size; ++kh) {
                        const int64_t ih = oh * stride + kh * dilation - padding; // Mapped input row
                        if (ih < 0 || ih >= input_rows) {
                            continue; // Skip padding positions
                        }

                        for (int64_t kw = 0; kw < kernel_size; ++kw) {
                            const int64_t iw = ow * stride + kw * dilation - padding; // Mapped input column
                            if (iw < 0 || iw >= input_cols) {
                                continue; // Skip padding positions
                            }

                            // Process each channel in the current group
                            for (int64_t ic = 0; ic < group_channels; ++ic) {
                                const int64_t input_channel = group_start + ic; // Actual input channel index

                                // Calculate memory offsets
                                const int64_t input_offset
                                    = input_channel * input_channel_stride + ih * input_row_stride + iw;
                                const int64_t kernel_offset = oc * kernel_out_channel_stride
                                                            + ic * kernel_spatial_stride + kh * kernel_size + kw;

                                // Multiply and accumulate
                                conv_sum += current_input[input_offset] * kernel_data[kernel_offset];
                            }
                        }
                    }

                    // Add bias if enabled
                    if (use_bias) {
                        conv_sum += bias_data[oc];
                    }

                    // Store result at [batch, channel, oh, ow]
                    current_output[oc * output_channel_stride + oh * output_row_stride + ow] = conv_sum;
                }
            }
        }
    }

    return output;
}

// Backward pass for computing kernel gradient
std::shared_ptr<Tensor> Conv2DKernelGrad(const std::shared_ptr<Tensor> &grad_output,
                                         const std::shared_ptr<Tensor> &input, const int64_t in_channels,
                                         const int64_t out_channels, const int64_t kernel_size, const int64_t stride,
                                         const int64_t padding, const int64_t dilation, const int64_t groups) {

    // Verify input dimensions (Channel First format)
    const auto &input_dims = input->Dims();
    const auto &grad_output_dims = grad_output->Dims();
    CHECK_EQ(input_dims.size(), 4) << "Input must be 4D tensor";
    CHECK_EQ(grad_output_dims.size(), 4) << "Gradient output must be 4D tensor";

    // Extract dimensions
    const auto batch_size = input_dims[0];
    const auto input_channels = input_dims[1];
    const auto input_rows = input_dims[2];
    const auto input_cols = input_dims[3];
    CHECK_EQ(input_channels, in_channels) << "Input channel mismatch";

    const auto grad_output_channels = grad_output_dims[1];
    const auto grad_output_rows = grad_output_dims[2];
    const auto grad_output_cols = grad_output_dims[3];
    CHECK_EQ(grad_output_channels, out_channels) << "Gradient output channel mismatch";

    // Initialize kernel gradient tensor: [out_channels, in_channels/groups, kernel_size, kernel_size]
    auto grad_kernel = std::make_shared<Tensor>(
        std::vector<int64_t>{out_channels, in_channels / groups, kernel_size, kernel_size}, DataType::kFLOAT32);
    grad_kernel->Fill(0.0f); // Initialize with zeros

    // Get raw data pointers
    auto grad_output_data = static_cast<const float *>(grad_output->DataPtr());
    auto input_data = static_cast<const float *>(input->DataPtr());
    auto grad_kernel_data = static_cast<float *>(grad_kernel->DataPtr());

    // Precompute strides for efficient memory access
    const int64_t input_batch_stride = in_channels * input_rows * input_cols;
    const int64_t input_channel_stride = input_rows * input_cols;
    const int64_t input_row_stride = input_cols;

    const int64_t grad_kernel_out_channel_stride = (in_channels / groups) * kernel_size * kernel_size;
    const int64_t grad_kernel_group_stride = kernel_size * kernel_size;

    const int64_t grad_output_batch_stride = out_channels * grad_output_rows * grad_output_cols;
    const int64_t grad_output_channel_stride = grad_output_rows * grad_output_cols;
    const int64_t grad_output_row_stride = grad_output_cols;

    // Backpropagate to compute kernel gradient
    for (int64_t b = 0; b < batch_size; ++b) {
        const float *current_input = input_data + b * input_batch_stride;
        const float *current_grad_output = grad_output_data + b * grad_output_batch_stride;

        // Process each output spatial position
        for (int64_t oh = 0; oh < grad_output_rows; ++oh) {
            for (int64_t ow = 0; ow < grad_output_cols; ++ow) {

                // Process each output channel
                for (int64_t oc = 0; oc < out_channels; ++oc) {
                    const float grad_val
                        = current_grad_output[oc * grad_output_channel_stride + oh * grad_output_row_stride + ow];

                    // Determine group parameters
                    const int64_t group = oc / (out_channels / groups);
                    const int64_t group_channels = in_channels / groups;
                    const int64_t group_start = group * group_channels;

                    // Process each kernel position
                    for (int64_t kh = 0; kh < kernel_size; ++kh) {
                        const int64_t ih = oh * stride + kh * dilation - padding;
                        if (ih < 0 || ih >= input_rows) {
                            continue;
                        }

                        for (int64_t kw = 0; kw < kernel_size; ++kw) {
                            const int64_t iw = ow * stride + kw * dilation - padding;
                            if (iw < 0 || iw >= input_cols) {
                                continue;
                            }

                            // Process each channel in the current group
                            for (int64_t ic = 0; ic < group_channels; ++ic) {
                                const int64_t input_channel = group_start + ic;

                                // Calculate memory offsets
                                const int64_t input_offset
                                    = input_channel * input_channel_stride + ih * input_row_stride + iw;
                                const int64_t grad_kernel_offset = oc * grad_kernel_out_channel_stride
                                                                 + ic * grad_kernel_group_stride + kh * kernel_size
                                                                 + kw;

                                // Accumulate gradient
                                grad_kernel_data[grad_kernel_offset] += grad_val * current_input[input_offset];
                            }
                        }
                    }
                }
            }
        }
    }

    return grad_kernel;
}

// Backward pass for computing bias gradient
std::shared_ptr<Tensor> Conv2DBiasGrad(const std::shared_ptr<Tensor> &grad_output) {

    // Verify gradient output dimensions: [batch_size, out_channels, height, width]
    const auto &dims = grad_output->Dims();
    CHECK_EQ(dims.size(), 4) << "Gradient output must be 4D tensor";

    // Extract dimensions
    const int64_t batch_size = dims[0];
    const int64_t out_channels = dims[1];
    const int64_t output_rows = dims[2];
    const int64_t output_cols = dims[3];

    // Initialize bias gradient tensor: [out_channels]
    auto grad_bias = std::make_shared<Tensor>(std::vector<int64_t>{out_channels}, DataType::kFLOAT32);
    grad_bias->Fill(0.0f); // Initialize with zeros

    // Get raw data pointers
    const float *grad_output_data = static_cast<const float *>(grad_output->DataPtr());
    float *grad_bias_data = static_cast<float *>(grad_bias->DataPtr());

    // Precompute strides
    const int64_t grad_output_batch_stride = out_channels * output_rows * output_cols;
    const int64_t grad_output_channel_stride = output_rows * output_cols;

    // Compute bias gradient: sum of gradients over spatial dimensions and batch
    for (int64_t oc = 0; oc < out_channels; ++oc) {
        float sum = 0.0f;

        // Sum gradients across all batches and spatial positions
        for (int64_t b = 0; b < batch_size; ++b) {
            const float *current_grad_output = grad_output_data + b * grad_output_batch_stride;
            const float *channel_grad = current_grad_output + oc * grad_output_channel_stride;

            // Sum over spatial positions
            for (int64_t i = 0; i < output_rows * output_cols; ++i) { sum += channel_grad[i]; }
        }

        grad_bias_data[oc] = sum;
    }

    return grad_bias;
}

// Backward pass for computing input gradient
std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
Conv2DBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &input,
               const size_t kernel_size, const int64_t in_channels, const int64_t out_channels, const int64_t stride,
               const int64_t padding, const int64_t dilation, const int64_t groups, const bool bias) {

    // Verify input dimensions (Channel First)
    const auto &input_dims = input->Dims();
    CHECK_EQ(input_dims.size(), 4) << "Input must be 4D tensor";

    // Extract dimensions
    const auto batch_size = input_dims[0];
    const auto input_channels = input_dims[1];
    const auto input_rows = input_dims[2];
    const auto input_cols = input_dims[3];
    CHECK_EQ(input_channels, in_channels) << "Input channel mismatch";

    // Compute kernel gradient
    auto grad_kernel = Conv2DKernelGrad(grad_output, input, in_channels, out_channels, kernel_size, stride, padding,
                                        dilation, groups);

    // Compute bias gradient
    std::shared_ptr<Tensor> grad_bias = nullptr;
    if (bias) {
        grad_bias = Conv2DBiasGrad(grad_output);
    }

    // Verify gradient output dimensions
    const auto &grad_output_dims = grad_output->Dims();
    CHECK_EQ(grad_output_dims.size(), 4) << "Gradient output must be 4D tensor";
    const auto grad_output_channels = grad_output_dims[1];
    const auto grad_output_rows = grad_output_dims[2];
    const auto grad_output_cols = grad_output_dims[3];
    CHECK_EQ(grad_output_channels, out_channels) << "Gradient output channel mismatch";

    // Initialize input gradient tensor: [batch_size, in_channels, height, width]
    auto grad_input = std::make_shared<Tensor>(std::vector<int64_t>{batch_size, in_channels, input_rows, input_cols},
                                               DataType::kFLOAT32);
    grad_input->Fill(0.0f); // Initialize with zeros

    // Get raw data pointers
    auto grad_output_data = static_cast<const float *>(grad_output->DataPtr());
    auto grad_kernel_data = static_cast<const float *>(grad_kernel->DataPtr());
    auto grad_input_data = static_cast<float *>(grad_input->DataPtr());

    // Precompute strides for efficient memory access
    const int64_t input_batch_stride = in_channels * input_rows * input_cols;
    const int64_t input_channel_stride = input_rows * input_cols;
    const int64_t input_row_stride = input_cols;

    const int64_t kernel_out_channel_stride = (in_channels / groups) * kernel_size * kernel_size;
    const int64_t kernel_group_stride = kernel_size * kernel_size;

    const int64_t grad_output_batch_stride = out_channels * grad_output_rows * grad_output_cols;
    const int64_t grad_output_channel_stride = grad_output_rows * grad_output_cols;
    const int64_t grad_output_row_stride = grad_output_cols;

    // Backpropagate to compute input gradient
    for (int64_t b = 0; b < batch_size; ++b) {
        const float *current_grad_output = grad_output_data + b * grad_output_batch_stride;
        float *current_grad_input = grad_input_data + b * input_batch_stride;

        // Process each output spatial position
        for (int64_t oh = 0; oh < grad_output_rows; ++oh) {
            for (int64_t ow = 0; ow < grad_output_cols; ++ow) {

                // Process each output channel
                for (int64_t oc = 0; oc < out_channels; ++oc) {
                    const float grad_val
                        = current_grad_output[oc * grad_output_channel_stride + oh * grad_output_row_stride + ow];

                    // Determine group parameters
                    const int64_t group = oc / (out_channels / groups);
                    const int64_t group_channels = in_channels / groups;
                    const int64_t group_start = group * group_channels;

                    // Process each kernel position
                    for (int64_t kh = 0; kh < kernel_size; ++kh) {
                        const int64_t ih = oh * stride + kh * dilation - padding;
                        if (ih < 0 || ih >= input_rows) {
                            continue;
                        }

                        for (int64_t kw = 0; kw < kernel_size; ++kw) {
                            const int64_t iw = ow * stride + kw * dilation - padding;
                            if (iw < 0 || iw >= input_cols) {
                                continue;
                            }

                            // Process each channel in the current group
                            for (int64_t ic = 0; ic < group_channels; ++ic) {
                                const int64_t input_channel = group_start + ic;

                                // Calculate memory offsets
                                const int64_t input_offset
                                    = input_channel * input_channel_stride + ih * input_row_stride + iw;
                                const int64_t kernel_offset
                                    = oc * kernel_out_channel_stride + ic * kernel_group_stride + kh * kernel_size + kw;

                                // Accumulate gradient
                                current_grad_input[input_offset] += grad_val * grad_kernel_data[kernel_offset];
                            }
                        }
                    }
                }
            }
        }
    }

    return {grad_input, grad_kernel, grad_bias};
}
} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_CONVOLUTION_KERNEL(kernel_name)                                                                   \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_CONVOLUTION_KERNEL(Conv2DForward);
REGISTER_CPU_CONVOLUTION_KERNEL(Conv2DBackward);

#undef REGISTER_CPU_CONVOLUTION_KERNEL
