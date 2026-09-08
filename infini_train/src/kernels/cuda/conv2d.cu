#include <memory>
#include <vector>

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

namespace infini_train::kernels::cuda {
namespace {

struct Conv2dShape {
    int batch;
    int in_channels;
    int out_channels;
    int input_height;
    int input_width;
    int kernel_height;
    int kernel_width;
    int output_height;
    int output_width;
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
    return {static_cast<int>(input_dims[0]),  static_cast<int>(input_dims[1]), static_cast<int>(weight_dims[0]),
            static_cast<int>(input_dims[2]),  static_cast<int>(input_dims[3]), static_cast<int>(weight_dims[2]),
            static_cast<int>(weight_dims[3]), static_cast<int>(output_height), static_cast<int>(output_width)};
}

cudaStream_t GetCudaStream(const Device &device) {
    return dynamic_cast<core::cuda::CudaStream *>(core::GetDeviceGuardImpl(device.type())->GetStream(device))
        ->cuda_stream();
}

__device__ size_t InputOffset(const Conv2dShape &shape, int batch, int channel, int height, int width) {
    return ((batch * shape.in_channels + channel) * shape.input_height + height) * shape.input_width + width;
}

__device__ size_t WeightOffset(const Conv2dShape &shape, int out_channel, int in_channel, int kernel_height,
                               int kernel_width) {
    return ((out_channel * shape.in_channels + in_channel) * shape.kernel_height + kernel_height) * shape.kernel_width
         + kernel_width;
}

__device__ size_t OutputOffset(const Conv2dShape &shape, int batch, int channel, int height, int width) {
    return ((batch * shape.out_channels + channel) * shape.output_height + height) * shape.output_width + width;
}

__global__ void Conv2dForwardKernel(const float *input, const float *weight, const float *bias, float *output,
                                    Conv2dShape shape, int stride, int padding) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t num_elements
        = static_cast<size_t>(shape.batch) * shape.out_channels * shape.output_height * shape.output_width;
    if (index >= num_elements) {
        return;
    }
    int temporary = static_cast<int>(index);
    const int out_width = temporary % shape.output_width;
    temporary /= shape.output_width;
    const int out_height = temporary % shape.output_height;
    temporary /= shape.output_height;
    const int out_channel = temporary % shape.out_channels;
    const int batch = temporary / shape.out_channels;
    float value = bias ? bias[out_channel] : 0.0f;
    for (int in_channel = 0; in_channel < shape.in_channels; ++in_channel) {
        for (int kernel_height = 0; kernel_height < shape.kernel_height; ++kernel_height) {
            const int input_height = out_height * stride + kernel_height - padding;
            if (input_height < 0 || input_height >= shape.input_height) {
                continue;
            }
            for (int kernel_width = 0; kernel_width < shape.kernel_width; ++kernel_width) {
                const int input_width = out_width * stride + kernel_width - padding;
                if (input_width >= 0 && input_width < shape.input_width) {
                    value += input[InputOffset(shape, batch, in_channel, input_height, input_width)]
                           * weight[WeightOffset(shape, out_channel, in_channel, kernel_height, kernel_width)];
                }
            }
        }
    }
    output[index] = value;
}

__global__ void Conv2dBackwardInputKernel(const float *weight, const float *grad_output, float *grad_input,
                                          Conv2dShape shape, int stride, int padding) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t num_elements
        = static_cast<size_t>(shape.batch) * shape.in_channels * shape.input_height * shape.input_width;
    if (index >= num_elements) {
        return;
    }
    int temporary = static_cast<int>(index);
    const int input_width = temporary % shape.input_width;
    temporary /= shape.input_width;
    const int input_height = temporary % shape.input_height;
    temporary /= shape.input_height;
    const int in_channel = temporary % shape.in_channels;
    const int batch = temporary / shape.in_channels;
    float value = 0.0f;
    for (int out_channel = 0; out_channel < shape.out_channels; ++out_channel) {
        for (int kernel_height = 0; kernel_height < shape.kernel_height; ++kernel_height) {
            const int numerator_height = input_height + padding - kernel_height;
            if (numerator_height < 0 || numerator_height % stride != 0) {
                continue;
            }
            const int out_height = numerator_height / stride;
            if (out_height >= shape.output_height) {
                continue;
            }
            for (int kernel_width = 0; kernel_width < shape.kernel_width; ++kernel_width) {
                const int numerator_width = input_width + padding - kernel_width;
                if (numerator_width < 0 || numerator_width % stride != 0) {
                    continue;
                }
                const int out_width = numerator_width / stride;
                if (out_width < shape.output_width) {
                    value += grad_output[OutputOffset(shape, batch, out_channel, out_height, out_width)]
                           * weight[WeightOffset(shape, out_channel, in_channel, kernel_height, kernel_width)];
                }
            }
        }
    }
    grad_input[index] = value;
}

__global__ void Conv2dBackwardWeightKernel(const float *input, const float *grad_output, float *grad_weight,
                                           Conv2dShape shape, int stride, int padding) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t num_elements
        = static_cast<size_t>(shape.out_channels) * shape.in_channels * shape.kernel_height * shape.kernel_width;
    if (index >= num_elements) {
        return;
    }
    int temporary = static_cast<int>(index);
    const int kernel_width = temporary % shape.kernel_width;
    temporary /= shape.kernel_width;
    const int kernel_height = temporary % shape.kernel_height;
    temporary /= shape.kernel_height;
    const int in_channel = temporary % shape.in_channels;
    const int out_channel = temporary / shape.in_channels;
    float value = 0.0f;
    for (int batch = 0; batch < shape.batch; ++batch) {
        for (int out_height = 0; out_height < shape.output_height; ++out_height) {
            const int input_height = out_height * stride + kernel_height - padding;
            if (input_height < 0 || input_height >= shape.input_height) {
                continue;
            }
            for (int out_width = 0; out_width < shape.output_width; ++out_width) {
                const int input_width = out_width * stride + kernel_width - padding;
                if (input_width >= 0 && input_width < shape.input_width) {
                    value += input[InputOffset(shape, batch, in_channel, input_height, input_width)]
                           * grad_output[OutputOffset(shape, batch, out_channel, out_height, out_width)];
                }
            }
        }
    }
    grad_weight[index] = value;
}

__global__ void Conv2dBackwardBiasKernel(const float *grad_output, float *grad_bias, Conv2dShape shape) {
    const int out_channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_channel >= shape.out_channels) {
        return;
    }
    float value = 0.0f;
    for (int batch = 0; batch < shape.batch; ++batch) {
        for (int out_height = 0; out_height < shape.output_height; ++out_height) {
            for (int out_width = 0; out_width < shape.output_width; ++out_width) {
                value += grad_output[OutputOffset(shape, batch, out_channel, out_height, out_width)];
            }
        }
    }
    grad_bias[out_channel] = value;
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
        DataType::kFLOAT32, input->GetDevice());
    constexpr int kThreads = 256;
    const size_t num_elements = output->NumElements();
    const int blocks = (num_elements + kThreads - 1) / kThreads;
    Conv2dForwardKernel<<<blocks, kThreads, 0, GetCudaStream(input->GetDevice())>>>(
        static_cast<const float *>(input->DataPtr()), static_cast<const float *>(weight->DataPtr()),
        bias ? static_cast<const float *>(bias->DataPtr()) : nullptr, static_cast<float *>(output->DataPtr()), shape,
        static_cast<int>(stride), static_cast<int>(padding));
    CUDA_CHECK(cudaGetLastError());
    return output;
}

std::shared_ptr<Tensor> Conv2dBackwardInput(const std::shared_ptr<Tensor> &weight,
                                            const std::shared_ptr<Tensor> &grad_output,
                                            const std::vector<int64_t> &input_dims, int64_t stride, int64_t padding) {
    CHECK(weight->GetDevice() == grad_output->GetDevice());
    CHECK(grad_output->Dtype() == DataType::kFLOAT32);
    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32, grad_output->GetDevice());
    auto input_for_shape = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32, grad_output->GetDevice());
    const Conv2dShape shape = ValidateShapes(input_for_shape, weight, stride, padding);
    CHECK(grad_output->Dims()
          == (std::vector<int64_t>{shape.batch, shape.out_channels, shape.output_height, shape.output_width}));
    constexpr int kThreads = 256;
    const int blocks = (grad_input->NumElements() + kThreads - 1) / kThreads;
    Conv2dBackwardInputKernel<<<blocks, kThreads, 0, GetCudaStream(grad_output->GetDevice())>>>(
        static_cast<const float *>(weight->DataPtr()), static_cast<const float *>(grad_output->DataPtr()),
        static_cast<float *>(grad_input->DataPtr()), shape, static_cast<int>(stride), static_cast<int>(padding));
    CUDA_CHECK(cudaGetLastError());
    return grad_input;
}

std::shared_ptr<Tensor> Conv2dBackwardWeight(const std::shared_ptr<Tensor> &input,
                                             const std::shared_ptr<Tensor> &grad_output,
                                             const std::vector<int64_t> &weight_dims, int64_t stride, int64_t padding) {
    CHECK(input->GetDevice() == grad_output->GetDevice());
    CHECK(grad_output->Dtype() == DataType::kFLOAT32);
    auto grad_weight = std::make_shared<Tensor>(weight_dims, DataType::kFLOAT32, input->GetDevice());
    auto weight_for_shape = std::make_shared<Tensor>(weight_dims, DataType::kFLOAT32, input->GetDevice());
    const Conv2dShape shape = ValidateShapes(input, weight_for_shape, stride, padding);
    CHECK(grad_output->Dims()
          == (std::vector<int64_t>{shape.batch, shape.out_channels, shape.output_height, shape.output_width}));
    constexpr int kThreads = 256;
    const int blocks = (grad_weight->NumElements() + kThreads - 1) / kThreads;
    Conv2dBackwardWeightKernel<<<blocks, kThreads, 0, GetCudaStream(input->GetDevice())>>>(
        static_cast<const float *>(input->DataPtr()), static_cast<const float *>(grad_output->DataPtr()),
        static_cast<float *>(grad_weight->DataPtr()), shape, static_cast<int>(stride), static_cast<int>(padding));
    CUDA_CHECK(cudaGetLastError());
    return grad_weight;
}

std::shared_ptr<Tensor> Conv2dBackwardBias(const std::shared_ptr<Tensor> &grad_output) {
    CHECK(grad_output->Dtype() == DataType::kFLOAT32);
    CHECK_EQ(grad_output->Dims().size(), 4);
    const auto &dims = grad_output->Dims();
    Conv2dShape shape{static_cast<int>(dims[0]), 0, static_cast<int>(dims[1]), 0, 0, 0, 0, static_cast<int>(dims[2]),
                      static_cast<int>(dims[3])};
    auto grad_bias
        = std::make_shared<Tensor>(std::vector<int64_t>{dims[1]}, DataType::kFLOAT32, grad_output->GetDevice());
    constexpr int kThreads = 256;
    const int blocks = (shape.out_channels + kThreads - 1) / kThreads;
    Conv2dBackwardBiasKernel<<<blocks, kThreads, 0, GetCudaStream(grad_output->GetDevice())>>>(
        static_cast<const float *>(grad_output->DataPtr()), static_cast<float *>(grad_bias->DataPtr()), shape);
    CUDA_CHECK(cudaGetLastError());
    return grad_bias;
}

} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_CONV2D_KERNEL(kernel_name)                                                                       \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_CONV2D_KERNEL(Conv2dForward)
REGISTER_CUDA_CONV2D_KERNEL(Conv2dBackwardInput)
REGISTER_CUDA_CONV2D_KERNEL(Conv2dBackwardWeight)
REGISTER_CUDA_CONV2D_KERNEL(Conv2dBackwardBias)

#undef REGISTER_CUDA_CONV2D_KERNEL
