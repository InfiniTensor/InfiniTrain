#include <cstdint>
#include <memory>
#include <vector>

#include <cub/block/block_reduce.cuh>

#include "glog/logging.h"

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"
#include "infini_train/src/kernels/common/gemm.h"
#include "infini_train/src/kernels/cuda/common/gemm.cuh"

namespace infini_train::kernels::cuda {
namespace {

// im2col for stride=1 / padding=0 cross-correlation. Each thread writes one
// element of the (flat_kernel, patches) column-major scratch buffer, so a single
// image's buffer is addressable as col[kk * patches + p]. This column-major
// order is what the cuBLAS GEMM geometry below expects. The (c, u, v) tap index
// is the same row-major flattening as an (O, C, kH, kW) weight.
__global__ void Im2colKernel(const float *__restrict__ input, float *__restrict__ col, int64_t total, int64_t channels,
                             int64_t height, int64_t width, int64_t kernel_h, int64_t kernel_w, int64_t out_width,
                             int64_t patches, int64_t flat_kernel) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        const int64_t p = idx % patches;
        const int64_t kk = (idx / patches) % flat_kernel;
        const int64_t n = idx / (flat_kernel * patches);

        const int64_t i = p / out_width;
        const int64_t j = p % out_width;
        const int64_t c = kk / (kernel_h * kernel_w);
        const int64_t rem = kk % (kernel_h * kernel_w);
        const int64_t u = rem / kernel_w;
        const int64_t v = rem % kernel_w;

        const int64_t in_idx = n * channels * height * width + c * height * width + (i + u) * width + (j + v);
        col[idx] = input[in_idx];
    }
}

// col2im: inverse of Im2col for the input gradient. An input pixel is covered by
// multiple output windows, so all patches that touch it are gathered. Each thread
// owns one (n, c, x, y) output cell and writes it once, so no atomics are needed.
__global__ void Col2imKernel(const float *__restrict__ col, float *__restrict__ grad_input, int64_t total,
                             int64_t channels, int64_t height, int64_t width, int64_t kernel_h, int64_t kernel_w,
                             int64_t out_height, int64_t out_width, int64_t patches, int64_t flat_kernel) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        const int64_t y = idx % width;
        const int64_t x = (idx / width) % height;
        const int64_t c = (idx / (width * height)) % channels;
        const int64_t n = idx / (width * height * channels);

        float sum = 0.0f;
        for (int64_t u = 0; u < kernel_h; ++u) {
            const int64_t i = x - u;
            if (i < 0 || i >= out_height) {
                continue;
            }
            for (int64_t v = 0; v < kernel_w; ++v) {
                const int64_t j = y - v;
                if (j < 0 || j >= out_width) {
                    continue;
                }
                const int64_t kk = c * kernel_h * kernel_w + u * kernel_w + v;
                const int64_t col_idx = n * (flat_kernel * patches) + kk * patches + i * out_width + j;
                sum += col[col_idx];
            }
        }
        grad_input[idx] = sum;
    }
}

// Broadcast bias over every spatial position of the (N, O, H, W) output, so a
// subsequent GEMM with beta=1 accumulates the convolution onto it.
__global__ void BroadcastBiasKernel(float *__restrict__ output, const float *__restrict__ bias, int64_t total,
                                    int64_t patches, int64_t out_channels) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        const int64_t o = (idx / patches) % out_channels;
        output[idx] = bias[o];
    }
}

// grad_bias(o) = sum_{n, i, j} grad_output(n, o, i, j). A parallel block reduction keeps the
// accumulation error at the same order as the cuBLAS reference rather than a single-thread sum.
template <int BLOCK_SIZE>
__global__ void GradBiasKernel(const float *__restrict__ grad_output, float *__restrict__ grad_bias, int64_t batch,
                               int64_t patches, int64_t out_channels) {
    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    const int64_t o = blockIdx.x;
    if (o >= out_channels) {
        return;
    }

    const int64_t total = batch * patches;
    float sum = 0.0f;
    for (int64_t idx = threadIdx.x; idx < total; idx += blockDim.x) {
        const int64_t n = idx / patches;
        const int64_t p = idx % patches;
        sum += grad_output[n * out_channels * patches + o * patches + p];
    }

    const float reduced = BlockReduce(temp_storage).Sum(sum);
    if (threadIdx.x == 0) {
        grad_bias[o] = reduced;
    }
}

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 1024;

int64_t NumBlocks(int64_t total) {
    int64_t blocks = (total + kThreads - 1) / kThreads;
    return blocks > kMaxBlocks ? kMaxBlocks : blocks;
}

cudaStream_t CurrentCudaStream(const Device &device) {
    return dynamic_cast<infini_train::core::cuda::CudaStream *>(
               infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
        ->cuda_stream();
}

} // namespace

std::shared_ptr<Tensor> Conv2dForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                                      const std::shared_ptr<Tensor> &bias) {
    const auto &input_dims = input->Dims();
    CHECK_EQ(input_dims.size(), 4);
    const int64_t batch = input_dims[0];
    const int64_t channels = input_dims[1];
    const int64_t height = input_dims[2];
    const int64_t width = input_dims[3];

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 4);
    const int64_t out_channels = weight_dims[0];
    const int64_t kernel_h = weight_dims[2];
    const int64_t kernel_w = weight_dims[3];
    CHECK_EQ(weight_dims[1], channels);
    CHECK_GE(height, kernel_h);
    CHECK_GE(width, kernel_w);

    CHECK(input->Dtype() == DataType::kFLOAT32) << "Conv2d requires FP32 tensors";
    CHECK(weight->Dtype() == DataType::kFLOAT32) << "Conv2d requires FP32 tensors";
    CHECK_EQ(input->GetDevice(), weight->GetDevice());

    if (bias) {
        CHECK_EQ(bias->Dims().size(), 1);
        CHECK_EQ(bias->Dims()[0], out_channels);
        CHECK(bias->Dtype() == DataType::kFLOAT32) << "Conv2d requires FP32 tensors";
        CHECK_EQ(input->GetDevice(), bias->GetDevice());
    }

    const int64_t out_height = height - kernel_h + 1;
    const int64_t out_width = width - kernel_w + 1;
    const int64_t patches = out_height * out_width;
    const int64_t flat_kernel = channels * kernel_h * kernel_w;

    auto output = std::make_shared<Tensor>(std::vector<int64_t>{batch, out_channels, out_height, out_width},
                                           DataType::kFLOAT32, input->GetDevice());
    if (batch == 0) {
        return output;
    }

    auto device = input->GetDevice();
    const cudaStream_t stream = CurrentCudaStream(device);

    // Scratch im2col buffer, indexed column-major (patches, flat_kernel) per image.
    auto col = std::make_shared<Tensor>(std::vector<int64_t>{batch, flat_kernel, patches}, DataType::kFLOAT32, device);

    Im2colKernel<<<NumBlocks(batch * patches * flat_kernel), kThreads, 0, stream>>>(
        static_cast<const float *>(input->DataPtr()), static_cast<float *>(col->DataPtr()),
        batch * patches * flat_kernel, channels, height, width, kernel_h, kernel_w, out_width, patches, flat_kernel);
    CUDA_CHECK(cudaGetLastError());

    if (bias) {
        // Prefill output with the bias, then the GEMM below accumulates onto it.
        BroadcastBiasKernel<<<NumBlocks(batch * out_channels * patches), kThreads, 0, stream>>>(
            static_cast<float *>(output->DataPtr()), static_cast<const float *>(bias->DataPtr()),
            batch * out_channels * patches, patches, out_channels);
        CUDA_CHECK(cudaGetLastError());
    }

    const float *col_data = static_cast<const float *>(col->DataPtr());
    const float *weight_data = static_cast<const float *>(weight->DataPtr());
    float *output_data = static_cast<float *>(output->DataPtr());
    const float beta = bias ? 1.0f : 0.0f;
    // One strided-batched GEMM over the batch: A steps through the per-image col blocks,
    // B (the weight) is shared by every image via stride 0, C steps through the output.
    // batch == 1 falls back to the non-batched Gemm path, which expects zero strides.
    Dispatcher::Instance().Call<void>({device.type(), "Gemm"}, device,
                                      GemmParams{
                                          .trans_a = GemmTranspose::kNoTranspose,
                                          .trans_b = GemmTranspose::kNoTranspose,
                                          .m = static_cast<int>(patches),
                                          .n = static_cast<int>(out_channels),
                                          .k = static_cast<int>(flat_kernel),
                                          .A = col_data,
                                          .lda = static_cast<int>(patches),
                                          .B = weight_data,
                                          .ldb = static_cast<int>(flat_kernel),
                                          .C = output_data,
                                          .ldc = static_cast<int>(patches),
                                          .alpha = 1.0f,
                                          .beta = beta,
                                          .batch_count = static_cast<int>(batch),
                                          .stride_a = batch > 1 ? patches * flat_kernel : 0,
                                          .stride_b = 0,
                                          .stride_c = batch > 1 ? out_channels * patches : 0,
                                          .input_dtype = DataType::kFLOAT32,
                                          .output_dtype = DataType::kFLOAT32,
                                      });

    return output;
}

std::shared_ptr<Tensor> Conv2dBackwardInput(const std::shared_ptr<Tensor> &weight,
                                            const std::shared_ptr<Tensor> &grad_output,
                                            const std::vector<int64_t> &input_dims) {
    CHECK_EQ(input_dims.size(), 4);
    const int64_t batch = input_dims[0];
    const int64_t channels = input_dims[1];
    const int64_t height = input_dims[2];
    const int64_t width = input_dims[3];

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 4);
    const int64_t out_channels = weight_dims[0];
    const int64_t kernel_h = weight_dims[2];
    const int64_t kernel_w = weight_dims[3];
    CHECK_EQ(weight_dims[1], channels);
    CHECK_GE(height, kernel_h);
    CHECK_GE(width, kernel_w);

    CHECK(weight->Dtype() == DataType::kFLOAT32) << "Conv2d requires FP32 tensors";
    CHECK(grad_output->Dtype() == DataType::kFLOAT32) << "Conv2d requires FP32 tensors";
    CHECK(weight->GetDevice() == grad_output->GetDevice());

    const int64_t out_height = height - kernel_h + 1;
    const int64_t out_width = width - kernel_w + 1;
    const int64_t patches = out_height * out_width;
    const int64_t flat_kernel = channels * kernel_h * kernel_w;

    const auto &grad_dims = grad_output->Dims();
    CHECK_EQ(grad_dims.size(), 4);
    CHECK_EQ(grad_dims[0], batch);
    CHECK_EQ(grad_dims[1], out_channels);
    CHECK_EQ(grad_dims[2], out_height);
    CHECK_EQ(grad_dims[3], out_width);

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32, grad_output->GetDevice());
    if (batch == 0) {
        return grad_input;
    }

    auto device = grad_output->GetDevice();
    const cudaStream_t stream = CurrentCudaStream(device);

    // Whole-batch scratch: col(n) = grad_output(n)^T * weight per image, stored column-major
    // (patches, flat_kernel), so the col2im below can gather every image in a single launch.
    auto col = std::make_shared<Tensor>(std::vector<int64_t>{batch, flat_kernel, patches}, DataType::kFLOAT32, device);

    const float *weight_data = static_cast<const float *>(weight->DataPtr());
    const float *grad_output_data = static_cast<const float *>(grad_output->DataPtr());
    float *grad_input_data = static_cast<float *>(grad_input->DataPtr());
    float *col_data = static_cast<float *>(col->DataPtr());

    // The transpose of the forward GEMM for every image at once: A steps through grad_output,
    // B (the weight) is shared by every image via stride 0, C steps through the col buffer.
    // batch == 1 falls back to the non-batched Gemm path, which expects zero strides.
    Dispatcher::Instance().Call<void>({device.type(), "Gemm"}, device,
                                      GemmParams{
                                          .trans_a = GemmTranspose::kNoTranspose,
                                          .trans_b = GemmTranspose::kTranspose,
                                          .m = static_cast<int>(patches),
                                          .n = static_cast<int>(flat_kernel),
                                          .k = static_cast<int>(out_channels),
                                          .A = grad_output_data,
                                          .lda = static_cast<int>(patches),
                                          .B = weight_data,
                                          .ldb = static_cast<int>(flat_kernel),
                                          .C = col_data,
                                          .ldc = static_cast<int>(patches),
                                          .alpha = 1.0f,
                                          .beta = 0.0f,
                                          .batch_count = static_cast<int>(batch),
                                          .stride_a = batch > 1 ? out_channels * patches : 0,
                                          .stride_b = 0,
                                          .stride_c = batch > 1 ? flat_kernel * patches : 0,
                                          .input_dtype = DataType::kFLOAT32,
                                          .output_dtype = DataType::kFLOAT32,
                                      });

    Col2imKernel<<<NumBlocks(batch * channels * height * width), kThreads, 0, stream>>>(
        col_data, grad_input_data, batch * channels * height * width, channels, height, width, kernel_h, kernel_w,
        out_height, out_width, patches, flat_kernel);
    CUDA_CHECK(cudaGetLastError());

    return grad_input;
}

std::shared_ptr<Tensor> Conv2dBackwardWeight(const std::shared_ptr<Tensor> &input,
                                             const std::shared_ptr<Tensor> &grad_output) {
    const auto &input_dims = input->Dims();
    CHECK_EQ(input_dims.size(), 4);
    const int64_t batch = input_dims[0];
    const int64_t channels = input_dims[1];
    const int64_t height = input_dims[2];
    const int64_t width = input_dims[3];

    const auto &grad_dims = grad_output->Dims();
    CHECK_EQ(grad_dims.size(), 4);
    CHECK_EQ(grad_dims[0], batch);

    CHECK(input->Dtype() == DataType::kFLOAT32) << "Conv2d requires FP32 tensors";
    CHECK(grad_output->Dtype() == DataType::kFLOAT32) << "Conv2d requires FP32 tensors";
    CHECK(input->GetDevice() == grad_output->GetDevice());

    const int64_t kernel_h = height - grad_dims[2] + 1;
    const int64_t kernel_w = width - grad_dims[3] + 1;
    CHECK_GT(kernel_h, 0);
    CHECK_GT(kernel_w, 0);

    const int64_t out_channels = grad_dims[1];
    const int64_t out_height = grad_dims[2];
    const int64_t out_width = grad_dims[3];
    const int64_t patches = out_height * out_width;
    const int64_t flat_kernel = channels * kernel_h * kernel_w;

    auto grad_weight = std::make_shared<Tensor>(std::vector<int64_t>{out_channels, channels, kernel_h, kernel_w},
                                                DataType::kFLOAT32, input->GetDevice());
    grad_weight->Fill(0.0f);
    if (batch == 0) {
        return grad_weight;
    }

    auto device = input->GetDevice();
    const cudaStream_t stream = CurrentCudaStream(device);

    auto col = std::make_shared<Tensor>(std::vector<int64_t>{batch, flat_kernel, patches}, DataType::kFLOAT32, device);

    Im2colKernel<<<NumBlocks(batch * patches * flat_kernel), kThreads, 0, stream>>>(
        static_cast<const float *>(input->DataPtr()), static_cast<float *>(col->DataPtr()),
        batch * patches * flat_kernel, channels, height, width, kernel_h, kernel_w, out_width, patches, flat_kernel);
    CUDA_CHECK(cudaGetLastError());

    const float *col_data = static_cast<const float *>(col->DataPtr());
    const float *grad_output_data = static_cast<const float *>(grad_output->DataPtr());
    float *grad_weight_data = static_cast<float *>(grad_weight->DataPtr());

    // grad_weight(o, kk) = sum_n grad_output(n, o, p) * col(n, kk, p). Computed as
    // the transpose of the forward GEMM and accumulated over the batch.
    for (int64_t n = 0; n < batch; ++n) {
        Dispatcher::Instance().Call<void>({device.type(), "Gemm"}, device,
                                          GemmParams{
                                              .trans_a = GemmTranspose::kTranspose,
                                              .trans_b = GemmTranspose::kNoTranspose,
                                              .m = static_cast<int>(flat_kernel),
                                              .n = static_cast<int>(out_channels),
                                              .k = static_cast<int>(patches),
                                              .A = col_data + n * patches * flat_kernel,
                                              .lda = static_cast<int>(patches),
                                              .B = grad_output_data + n * out_channels * patches,
                                              .ldb = static_cast<int>(patches),
                                              .C = grad_weight_data,
                                              .ldc = static_cast<int>(flat_kernel),
                                              .alpha = 1.0f,
                                              .beta = 1.0f,
                                              .batch_count = 1,
                                              .input_dtype = DataType::kFLOAT32,
                                              .output_dtype = DataType::kFLOAT32,
                                          });
    }

    return grad_weight;
}

std::shared_ptr<Tensor> Conv2dBackwardBias(const std::shared_ptr<Tensor> &grad_output, int64_t out_channels) {
    const auto &grad_dims = grad_output->Dims();
    CHECK_EQ(grad_dims.size(), 4);
    CHECK_EQ(grad_dims[1], out_channels);
    CHECK(grad_output->Dtype() == DataType::kFLOAT32) << "Conv2d requires FP32 tensors";

    const int64_t batch = grad_dims[0];
    const int64_t patches = grad_dims[2] * grad_dims[3];

    auto grad_bias
        = std::make_shared<Tensor>(std::vector<int64_t>{out_channels}, DataType::kFLOAT32, grad_output->GetDevice());

    auto device = grad_output->GetDevice();
    const cudaStream_t stream = CurrentCudaStream(device);

    constexpr int kGradBiasBlock = 256;
    GradBiasKernel<kGradBiasBlock><<<out_channels, kGradBiasBlock, 0, stream>>>(
        static_cast<const float *>(grad_output->DataPtr()), static_cast<float *>(grad_bias->DataPtr()), batch, patches,
        out_channels);
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
