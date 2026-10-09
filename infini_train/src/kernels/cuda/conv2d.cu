#include <limits>
#include <memory>
#include <vector>

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"
#include "infini_train/src/kernels/common/gemm.h"

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

Conv2dShape ValidateShapes(const std::vector<int64_t> &input_dims, const std::vector<int64_t> &weight_dims,
                           int64_t stride, int64_t padding) {
    CHECK_EQ(input_dims.size(), 4) << "Conv2d expects NCHW input";
    CHECK_EQ(weight_dims.size(), 4) << "Conv2d expects OIHW weight";
    constexpr int64_t kMaxDim = std::numeric_limits<int>::max();
    for (const auto &dims : {input_dims, weight_dims}) {
        for (int64_t dim : dims) {
            CHECK_GT(dim, 0);
            CHECK_LE(dim, kMaxDim);
        }
    }
    CHECK_GT(stride, 0);
    CHECK_LE(stride, kMaxDim);
    CHECK_GE(padding, 0);
    CHECK_LE(padding, kMaxDim);
    CHECK_EQ(input_dims[1], weight_dims[1]);
    CHECK_GE(input_dims[2] + 2 * padding, weight_dims[2]);
    CHECK_GE(input_dims[3] + 2 * padding, weight_dims[3]);
    const int64_t output_height = (input_dims[2] + 2 * padding - weight_dims[2]) / stride + 1;
    const int64_t output_width = (input_dims[3] + 2 * padding - weight_dims[3]) / stride + 1;
    CHECK_GT(output_height, 0);
    CHECK_GT(output_width, 0);
    CHECK_LE(output_height, kMaxDim / output_width) << "Conv2d GEMM spatial dimension exceeds int range";
    CHECK_LE(weight_dims[1], kMaxDim / weight_dims[2]);
    CHECK_LE(weight_dims[1] * weight_dims[2], kMaxDim / weight_dims[3])
        << "Conv2d GEMM reduction dimension exceeds int range";
    return {static_cast<int>(input_dims[0]),  static_cast<int>(input_dims[1]), static_cast<int>(weight_dims[0]),
            static_cast<int>(input_dims[2]),  static_cast<int>(input_dims[3]), static_cast<int>(weight_dims[2]),
            static_cast<int>(weight_dims[3]), static_cast<int>(output_height), static_cast<int>(output_width)};
}

cudaStream_t GetCudaStream(const Device &device) {
    return dynamic_cast<core::cuda::CudaStream *>(core::GetDeviceGuardImpl(device.type())->GetStream(device))
        ->cuda_stream();
}

__device__ size_t InputOffset(const Conv2dShape &shape, int batch, int channel, int height, int width) {
    return ((static_cast<size_t>(batch) * shape.in_channels + channel) * shape.input_height + height)
             * shape.input_width
         + width;
}

__device__ size_t OutputOffset(const Conv2dShape &shape, int batch, int channel, int height, int width) {
    return ((static_cast<size_t>(batch) * shape.out_channels + channel) * shape.output_height + height)
             * shape.output_width
         + width;
}

__global__ void Im2colKernel(const float *input, float *columns, Conv2dShape shape, int stride, int padding,
                             size_t num_elements) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= num_elements) {
        return;
    }
    size_t temporary = index;
    const int out_width = temporary % shape.output_width;
    temporary /= shape.output_width;
    const int out_height = temporary % shape.output_height;
    temporary /= shape.output_height;
    const int kernel_width = temporary % shape.kernel_width;
    temporary /= shape.kernel_width;
    const int kernel_height = temporary % shape.kernel_height;
    temporary /= shape.kernel_height;
    const int channel = temporary % shape.in_channels;
    const int batch = temporary / shape.in_channels;
    const int64_t height = static_cast<int64_t>(out_height) * stride + kernel_height - padding;
    const int64_t width = static_cast<int64_t>(out_width) * stride + kernel_width - padding;
    columns[index] = height >= 0 && height < shape.input_height && width >= 0 && width < shape.input_width
                       ? input[InputOffset(shape, batch, channel, height, width)]
                       : 0.0f;
}

__global__ void Col2imKernel(const float *columns, float *grad_input, Conv2dShape shape, int stride, int padding,
                             size_t num_elements) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= num_elements) {
        return;
    }
    size_t temporary = index;
    const int input_width = temporary % shape.input_width;
    temporary /= shape.input_width;
    const int input_height = temporary % shape.input_height;
    temporary /= shape.input_height;
    const int channel = temporary % shape.in_channels;
    const int batch = temporary / shape.in_channels;
    float value = 0.0f;
    for (int kh = 0; kh < shape.kernel_height; ++kh) {
        const int64_t height = static_cast<int64_t>(input_height) + padding - kh;
        if (height < 0 || height % stride != 0 || height / stride >= shape.output_height) {
            continue;
        }
        for (int kw = 0; kw < shape.kernel_width; ++kw) {
            const int64_t width = static_cast<int64_t>(input_width) + padding - kw;
            if (width >= 0 && width % stride == 0 && width / stride < shape.output_width) {
                const size_t row
                    = ((static_cast<size_t>(batch) * shape.in_channels + channel) * shape.kernel_height + kh)
                        * shape.kernel_width
                    + kw;
                value += columns[(row * shape.output_height + height / stride) * shape.output_width + width / stride];
            }
        }
    }
    grad_input[index] = value;
}

__global__ void AddBiasKernel(float *output, const float *bias, int spatial_size, int channels, size_t num_elements) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < num_elements) {
        output[index] += bias[(index / spatial_size) % channels];
    }
}

std::shared_ptr<Tensor> Im2col(const std::shared_ptr<Tensor> &input, const Conv2dShape &shape, int64_t stride,
                               int64_t padding) {
    const int k = shape.in_channels * shape.kernel_height * shape.kernel_width;
    const int l = shape.output_height * shape.output_width;
    auto columns
        = std::make_shared<Tensor>(std::vector<int64_t>{shape.batch, k, l}, DataType::kFLOAT32, input->GetDevice());
    constexpr int kThreads = 256;
    const size_t blocks = (columns->NumElements() + kThreads - 1) / kThreads;
    Im2colKernel<<<blocks, kThreads, 0, GetCudaStream(input->GetDevice())>>>(
        static_cast<const float *>(input->DataPtr()), static_cast<float *>(columns->DataPtr()), shape, stride, padding,
        columns->NumElements());
    CUDA_CHECK(cudaGetLastError());
    return columns;
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
    CHECK(input->Dtype() == DataType::kFLOAT32);
    CHECK(weight->Dtype() == DataType::kFLOAT32);
    CHECK(input->GetDevice() == weight->GetDevice());
    const Conv2dShape shape = ValidateShapes(input->Dims(), weight->Dims(), stride, padding);
    if (bias) {
        CHECK(bias->Dtype() == DataType::kFLOAT32);
        CHECK(bias->GetDevice() == input->GetDevice());
        CHECK(bias->Dims() == (std::vector<int64_t>{shape.out_channels}));
    }
    auto output = std::make_shared<Tensor>(
        std::vector<int64_t>{shape.batch, shape.out_channels, shape.output_height, shape.output_width},
        DataType::kFLOAT32, input->GetDevice());
    auto columns = Im2col(input, shape, stride, padding);
    const int k = shape.in_channels * shape.kernel_height * shape.kernel_width;
    const int l = shape.output_height * shape.output_width;
    // Row-major [O, L] = W[O, K] * columns[K, L], viewed as column-major matrices.
    GemmParams params;
    params.m = l;
    params.n = shape.out_channels;
    params.k = k;
    params.A = columns->DataPtr();
    params.lda = l;
    params.B = weight->DataPtr();
    params.ldb = k;
    params.C = output->DataPtr();
    params.ldc = l;
    params.batch_count = shape.batch;
    if (shape.batch > 1) {
        params.stride_a = static_cast<long long>(k) * l;
        params.stride_c = static_cast<long long>(shape.out_channels) * l;
    }
    params.input_dtype = params.output_dtype = DataType::kFLOAT32;
    Dispatcher::Instance().Call<void>({input->GetDevice().type(), "Gemm"}, input->GetDevice(), params);
    if (bias) {
        constexpr int kThreads = 256;
        const size_t blocks = (output->NumElements() + kThreads - 1) / kThreads;
        AddBiasKernel<<<blocks, kThreads, 0, GetCudaStream(input->GetDevice())>>>(
            static_cast<float *>(output->DataPtr()), static_cast<const float *>(bias->DataPtr()), l, shape.out_channels,
            output->NumElements());
        CUDA_CHECK(cudaGetLastError());
    }
    return output;
}

std::shared_ptr<Tensor> Conv2dBackwardInput(const std::shared_ptr<Tensor> &weight,
                                            const std::shared_ptr<Tensor> &grad_output,
                                            const std::vector<int64_t> &input_dims, int64_t stride, int64_t padding) {
    CHECK(weight->GetDevice() == grad_output->GetDevice());
    CHECK(grad_output->Dtype() == DataType::kFLOAT32);
    CHECK(weight->Dtype() == DataType::kFLOAT32);
    const Conv2dShape shape = ValidateShapes(input_dims, weight->Dims(), stride, padding);
    CHECK(grad_output->Dims()
          == (std::vector<int64_t>{shape.batch, shape.out_channels, shape.output_height, shape.output_width}));
    const int k = shape.in_channels * shape.kernel_height * shape.kernel_width;
    const int l = shape.output_height * shape.output_width;
    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32, grad_output->GetDevice());
    auto columns = std::make_shared<Tensor>(std::vector<int64_t>{shape.batch, k, l}, DataType::kFLOAT32,
                                            grad_output->GetDevice());
    // dColumns[K, L] = W^T[K, O] * dOutput[O, L].
    GemmParams params;
    params.trans_b = GemmTranspose::kTranspose;
    params.m = l;
    params.n = k;
    params.k = shape.out_channels;
    params.A = grad_output->DataPtr();
    params.lda = l;
    params.B = weight->DataPtr();
    params.ldb = k;
    params.C = columns->DataPtr();
    params.ldc = l;
    params.batch_count = shape.batch;
    if (shape.batch > 1) {
        params.stride_a = static_cast<long long>(shape.out_channels) * l;
        params.stride_c = static_cast<long long>(k) * l;
    }
    params.input_dtype = params.output_dtype = DataType::kFLOAT32;
    Dispatcher::Instance().Call<void>({grad_output->GetDevice().type(), "Gemm"}, grad_output->GetDevice(), params);
    constexpr int kThreads = 256;
    const size_t blocks = (grad_input->NumElements() + kThreads - 1) / kThreads;
    Col2imKernel<<<blocks, kThreads, 0, GetCudaStream(grad_output->GetDevice())>>>(
        static_cast<const float *>(columns->DataPtr()), static_cast<float *>(grad_input->DataPtr()), shape, stride,
        padding, grad_input->NumElements());
    CUDA_CHECK(cudaGetLastError());
    return grad_input;
}

std::shared_ptr<Tensor> Conv2dBackwardWeight(const std::shared_ptr<Tensor> &input,
                                             const std::shared_ptr<Tensor> &grad_output,
                                             const std::vector<int64_t> &weight_dims, int64_t stride, int64_t padding) {
    CHECK(input->GetDevice() == grad_output->GetDevice());
    CHECK(grad_output->Dtype() == DataType::kFLOAT32);
    CHECK(input->Dtype() == DataType::kFLOAT32);
    const Conv2dShape shape = ValidateShapes(input->Dims(), weight_dims, stride, padding);
    CHECK(grad_output->Dims()
          == (std::vector<int64_t>{shape.batch, shape.out_channels, shape.output_height, shape.output_width}));
    auto grad_weight = std::make_shared<Tensor>(weight_dims, DataType::kFLOAT32, input->GetDevice());
    auto columns = Im2col(input, shape, stride, padding);
    const int k = shape.in_channels * shape.kernel_height * shape.kernel_width;
    const int l = shape.output_height * shape.output_width;
    // Accumulate dWeight[O, K] = dOutput[O, L] * columns^T[L, K] over the batch.
    GemmParams params;
    params.trans_a = GemmTranspose::kTranspose;
    params.m = k;
    params.n = shape.out_channels;
    params.k = l;
    params.lda = l;
    params.ldb = l;
    params.C = grad_weight->DataPtr();
    params.ldc = k;
    params.input_dtype = params.output_dtype = DataType::kFLOAT32;
    for (int batch = 0; batch < shape.batch; ++batch) {
        params.A = static_cast<const float *>(columns->DataPtr()) + static_cast<size_t>(batch) * k * l;
        params.B
            = static_cast<const float *>(grad_output->DataPtr()) + static_cast<size_t>(batch) * shape.out_channels * l;
        params.beta = batch == 0 ? 0.0f : 1.0f;
        Dispatcher::Instance().Call<void>({input->GetDevice().type(), "Gemm"}, input->GetDevice(), params);
    }
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
