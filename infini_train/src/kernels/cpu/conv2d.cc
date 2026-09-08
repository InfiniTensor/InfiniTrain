#include <cstdint>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
namespace {

struct Conv2dShape {
    int64_t batch;
    int64_t in_channels;
    int64_t out_channels;
    int64_t input_height;
    int64_t input_width;
    int64_t kernel_height;
    int64_t kernel_width;
    int64_t output_height;
    int64_t output_width;
};

Conv2dShape ValidateShapes(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight, int64_t stride,
                           int64_t padding) {
    CHECK(input->Dtype() == DataType::kFLOAT32);
    CHECK(weight->Dtype() == DataType::kFLOAT32);
    CHECK(input->GetDevice() == weight->GetDevice());
    CHECK_EQ(input->Dims().size(), 4) << "Conv2d expects NCHW input";
    CHECK_EQ(weight->Dims().size(), 4) << "Conv2d expects OIHW weight";
    CHECK_GT(stride, 0);
    CHECK_GE(padding, 0);
    const auto &input_dims = input->Dims();
    const auto &weight_dims = weight->Dims();
    CHECK_EQ(input_dims[1], weight_dims[1]);
    CHECK_GE(input_dims[2] + 2 * padding, weight_dims[2]);
    CHECK_GE(input_dims[3] + 2 * padding, weight_dims[3]);
    const int64_t output_height = (input_dims[2] + 2 * padding - weight_dims[2]) / stride + 1;
    const int64_t output_width = (input_dims[3] + 2 * padding - weight_dims[3]) / stride + 1;
    CHECK_GT(output_height, 0);
    CHECK_GT(output_width, 0);
    return {input_dims[0],  input_dims[1],  weight_dims[0], input_dims[2], input_dims[3],
            weight_dims[2], weight_dims[3], output_height,  output_width};
}

size_t InputOffset(const Conv2dShape &shape, int64_t batch, int64_t channel, int64_t height, int64_t width) {
    return ((batch * shape.in_channels + channel) * shape.input_height + height) * shape.input_width + width;
}

size_t WeightOffset(const Conv2dShape &shape, int64_t out_channel, int64_t in_channel, int64_t kernel_height,
                    int64_t kernel_width) {
    return ((out_channel * shape.in_channels + in_channel) * shape.kernel_height + kernel_height) * shape.kernel_width
         + kernel_width;
}

size_t OutputOffset(const Conv2dShape &shape, int64_t batch, int64_t channel, int64_t height, int64_t width) {
    return ((batch * shape.out_channels + channel) * shape.output_height + height) * shape.output_width + width;
}

} // namespace

std::shared_ptr<Tensor> Conv2dForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                                      const std::shared_ptr<Tensor> &bias, int64_t stride, int64_t padding) {
    const Conv2dShape shape = ValidateShapes(input, weight, stride, padding);
    if (bias) {
        CHECK(bias->Dtype() == DataType::kFLOAT32);
        CHECK(bias->GetDevice() == input->GetDevice());
        CHECK(bias->Dims() == (std::vector<int64_t>{shape.out_channels}));
    }
    auto output = std::make_shared<Tensor>(
        std::vector<int64_t>{shape.batch, shape.out_channels, shape.output_height, shape.output_width},
        DataType::kFLOAT32);
    const auto *input_data = static_cast<const float *>(input->DataPtr());
    const auto *weight_data = static_cast<const float *>(weight->DataPtr());
    const auto *bias_data = bias ? static_cast<const float *>(bias->DataPtr()) : nullptr;
    auto *output_data = static_cast<float *>(output->DataPtr());
    for (int64_t batch = 0; batch < shape.batch; ++batch) {
        for (int64_t out_channel = 0; out_channel < shape.out_channels; ++out_channel) {
            for (int64_t out_height = 0; out_height < shape.output_height; ++out_height) {
                for (int64_t out_width = 0; out_width < shape.output_width; ++out_width) {
                    float value = bias_data ? bias_data[out_channel] : 0.0f;
                    for (int64_t in_channel = 0; in_channel < shape.in_channels; ++in_channel) {
                        for (int64_t kernel_height = 0; kernel_height < shape.kernel_height; ++kernel_height) {
                            const int64_t input_height = out_height * stride + kernel_height - padding;
                            if (input_height < 0 || input_height >= shape.input_height) {
                                continue;
                            }
                            for (int64_t kernel_width = 0; kernel_width < shape.kernel_width; ++kernel_width) {
                                const int64_t input_width = out_width * stride + kernel_width - padding;
                                if (input_width >= 0 && input_width < shape.input_width) {
                                    value
                                        += input_data[InputOffset(shape, batch, in_channel, input_height, input_width)]
                                         * weight_data[WeightOffset(shape, out_channel, in_channel, kernel_height,
                                                                    kernel_width)];
                                }
                            }
                        }
                    }
                    output_data[OutputOffset(shape, batch, out_channel, out_height, out_width)] = value;
                }
            }
        }
    }
    return output;
}

std::shared_ptr<Tensor> Conv2dBackwardInput(const std::shared_ptr<Tensor> &weight,
                                            const std::shared_ptr<Tensor> &grad_output,
                                            const std::vector<int64_t> &input_dims, int64_t stride, int64_t padding) {
    CHECK(weight->GetDevice() == grad_output->GetDevice());
    CHECK(grad_output->Dtype() == DataType::kFLOAT32);
    auto input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);
    const Conv2dShape shape = ValidateShapes(input, weight, stride, padding);
    CHECK(grad_output->Dims()
          == (std::vector<int64_t>{shape.batch, shape.out_channels, shape.output_height, shape.output_width}));
    input->Fill(0.0f);
    auto *grad_input_data = static_cast<float *>(input->DataPtr());
    const auto *weight_data = static_cast<const float *>(weight->DataPtr());
    const auto *grad_output_data = static_cast<const float *>(grad_output->DataPtr());
    for (int64_t batch = 0; batch < shape.batch; ++batch) {
        for (int64_t out_channel = 0; out_channel < shape.out_channels; ++out_channel) {
            for (int64_t out_height = 0; out_height < shape.output_height; ++out_height) {
                for (int64_t out_width = 0; out_width < shape.output_width; ++out_width) {
                    const float grad = grad_output_data[OutputOffset(shape, batch, out_channel, out_height, out_width)];
                    for (int64_t in_channel = 0; in_channel < shape.in_channels; ++in_channel) {
                        for (int64_t kernel_height = 0; kernel_height < shape.kernel_height; ++kernel_height) {
                            const int64_t input_height = out_height * stride + kernel_height - padding;
                            if (input_height < 0 || input_height >= shape.input_height) {
                                continue;
                            }
                            for (int64_t kernel_width = 0; kernel_width < shape.kernel_width; ++kernel_width) {
                                const int64_t input_width = out_width * stride + kernel_width - padding;
                                if (input_width >= 0 && input_width < shape.input_width) {
                                    grad_input_data[InputOffset(shape, batch, in_channel, input_height, input_width)]
                                        += grad
                                         * weight_data[WeightOffset(shape, out_channel, in_channel, kernel_height,
                                                                    kernel_width)];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return input;
}

std::shared_ptr<Tensor> Conv2dBackwardWeight(const std::shared_ptr<Tensor> &input,
                                             const std::shared_ptr<Tensor> &grad_output,
                                             const std::vector<int64_t> &weight_dims, int64_t stride, int64_t padding) {
    CHECK(input->GetDevice() == grad_output->GetDevice());
    CHECK(grad_output->Dtype() == DataType::kFLOAT32);
    auto weight = std::make_shared<Tensor>(weight_dims, DataType::kFLOAT32);
    const Conv2dShape shape = ValidateShapes(input, weight, stride, padding);
    CHECK(grad_output->Dims()
          == (std::vector<int64_t>{shape.batch, shape.out_channels, shape.output_height, shape.output_width}));
    auto *grad_weight_data = static_cast<float *>(weight->DataPtr());
    const auto *input_data = static_cast<const float *>(input->DataPtr());
    const auto *grad_output_data = static_cast<const float *>(grad_output->DataPtr());
    for (int64_t out_channel = 0; out_channel < shape.out_channels; ++out_channel) {
        for (int64_t in_channel = 0; in_channel < shape.in_channels; ++in_channel) {
            for (int64_t kernel_height = 0; kernel_height < shape.kernel_height; ++kernel_height) {
                for (int64_t kernel_width = 0; kernel_width < shape.kernel_width; ++kernel_width) {
                    float value = 0.0f;
                    for (int64_t batch = 0; batch < shape.batch; ++batch) {
                        for (int64_t out_height = 0; out_height < shape.output_height; ++out_height) {
                            const int64_t input_height = out_height * stride + kernel_height - padding;
                            if (input_height < 0 || input_height >= shape.input_height) {
                                continue;
                            }
                            for (int64_t out_width = 0; out_width < shape.output_width; ++out_width) {
                                const int64_t input_width = out_width * stride + kernel_width - padding;
                                if (input_width >= 0 && input_width < shape.input_width) {
                                    value
                                        += input_data[InputOffset(shape, batch, in_channel, input_height, input_width)]
                                         * grad_output_data[OutputOffset(shape, batch, out_channel, out_height,
                                                                         out_width)];
                                }
                            }
                        }
                    }
                    grad_weight_data[WeightOffset(shape, out_channel, in_channel, kernel_height, kernel_width)] = value;
                }
            }
        }
    }
    return weight;
}

std::shared_ptr<Tensor> Conv2dBackwardBias(const std::shared_ptr<Tensor> &grad_output) {
    CHECK(grad_output->Dtype() == DataType::kFLOAT32);
    CHECK_EQ(grad_output->Dims().size(), 4);
    const auto &dims = grad_output->Dims();
    auto grad_bias = std::make_shared<Tensor>(std::vector<int64_t>{dims[1]}, DataType::kFLOAT32);
    const auto *grad_output_data = static_cast<const float *>(grad_output->DataPtr());
    auto *grad_bias_data = static_cast<float *>(grad_bias->DataPtr());
    const size_t spatial_size = dims[2] * dims[3];
    for (int64_t channel = 0; channel < dims[1]; ++channel) {
        float value = 0.0f;
        for (int64_t batch = 0; batch < dims[0]; ++batch) {
            const size_t offset = (batch * dims[1] + channel) * spatial_size;
            for (size_t index = 0; index < spatial_size; ++index) { value += grad_output_data[offset + index]; }
        }
        grad_bias_data[channel] = value;
    }
    return grad_bias;
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_CONV2D_KERNEL(kernel_name)                                                                        \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_CONV2D_KERNEL(Conv2dForward)
REGISTER_CPU_CONV2D_KERNEL(Conv2dBackwardInput)
REGISTER_CPU_CONV2D_KERNEL(Conv2dBackwardWeight)
REGISTER_CPU_CONV2D_KERNEL(Conv2dBackwardBias)

#undef REGISTER_CPU_CONV2D_KERNEL
