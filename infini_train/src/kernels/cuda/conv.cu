#include <cstdint>
#include <memory>
#include <vector>

#include <cub/block/block_reduce.cuh>

#include "glog/logging.h"

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/cuda/cuda_dispatch.h"
#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"
#include "infini_train/src/kernels/common/conv.h"
#include "infini_train/src/kernels/cuda/common/gemm.cuh"

namespace infini_train::kernels::cuda {
namespace {

// Column layout: per image [K, P] row-major with K = Cin * Kh * Kw (kernel row
// k = (ic * Kh + kh) * Kw + kw) and P = Hout * Wout (patch column
// p = oh * Wout + ow). Out-of-bounds taps (padding) read as zero.
// Batched output is [N, K, P]. All shape arithmetic comes from Conv2dMeta.
__global__ void Im2colKernel(const float *input, float *columns, Conv2dMeta meta) {
    const int64_t total = meta.batch * meta.kernel_elems * meta.patches;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const int64_t p = idx % meta.patches;
    const int64_t k = (idx / meta.patches) % meta.kernel_elems;
    const int64_t n = idx / (meta.patches * meta.kernel_elems);

    const int64_t kw = k % meta.kernel_w;
    const int64_t kh = (k / meta.kernel_w) % meta.kernel_h;
    const int64_t ic = k / (meta.kernel_w * meta.kernel_h);

    const int64_t oh = p / meta.output_w;
    const int64_t ow = p % meta.output_w;
    const int64_t ih = oh * meta.stride + kh - meta.padding;
    const int64_t iw = ow * meta.stride + kw - meta.padding;

    float value = 0.0f;
    if (ih >= 0 && ih < meta.input_h && iw >= 0 && iw < meta.input_w) {
        value = input[((n * meta.in_channels + ic) * meta.input_h + ih) * meta.input_w + iw];
    }
    columns[idx] = value;
}

// Broadcast per-channel bias over an [N, Cout, P] row-major buffer:
// out[(n * Cout + oc) * P + p] = bias[oc].
__global__ void ConvBiasCopyKernel(float *output, const float *bias, int64_t out_channels, int64_t patches,
                                   int64_t total) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    output[idx] = bias[(idx / patches) % out_channels];
}

// dB tiles are reduced over N*P on-device: grad_output holds row-major
// [N, Cout, P] (P = Hout * Wout); channel oc's taps are strided as
// [(n * Cout + oc) * P + p]. One block per channel (same pattern as
// cuda::LinearBackwardBias ReduceRowsKernel), each thread striding over
// rows = N * P with a cub BlockReduce sum.
template <int BLOCK_SIZE>
__global__ void ConvBackwardBiasKernel(const float *__restrict__ grad_output, float *__restrict__ grad_bias,
                                       int64_t rows, int64_t out_channels, int64_t patches) {
    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    const int64_t oc = blockIdx.x;
    float sum = 0.0f;
    for (int64_t r = threadIdx.x; r < rows; r += blockDim.x) {
        const int64_t n = r / patches;
        const int64_t p = r % patches;
        sum += grad_output[(n * out_channels + oc) * patches + p];
    }

    const float reduced = BlockReduce(temp_storage).Sum(sum);
    if (threadIdx.x == 0) {
        grad_bias[oc] = reduced;
    }
}

// dW tiles are reduced over the batch on-device: per_image holds one
// row-major [Cout, K] tile per image (tile = oc * K + k); grad_weight is
// the flat [Cout * K] sum (= OIHW element count, same memory order).
__global__ void ConvBackwardWeightSumKernel(const float *per_image, float *grad_weight, int64_t batch, int64_t elems) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= elems) {
        return;
    }
    float sum = 0.0f;
    for (int64_t n = 0; n < batch; ++n) { sum += per_image[n * elems + idx]; }
    grad_weight[idx] = sum;
}

} // namespace

// dXcol tiles hold one row-major [K, P] tile per image (tile = k * P + p,
// K = Cin * Kh * Kw matches Im2colKernel's row order, P = Hout * Wout);
// Col2imKernel gathers each input element's covering patches into grad_input.
__global__ void Col2imKernel(const float *columns, float *grad_input, Conv2dMeta meta) {
    const int64_t hw = meta.input_h * meta.input_w;
    const int64_t chw = meta.in_channels * hw;
    const int64_t total = meta.batch * chw;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const int64_t w = idx % meta.input_w;
    const int64_t h = (idx / meta.input_w) % meta.input_h;
    const int64_t c = (idx / hw) % meta.in_channels;
    const int64_t n = idx / chw;

    // Gather: input tap (h, w) is covered by patches whose receptive field
    // contains it, i.e. h == oh * stride + kh - padding (same for w).
    // Padding taps contribute 0 (skipped via the divisibility/bounds checks).
    float sum = 0.0f;
    for (int64_t kh = 0; kh < meta.kernel_h; ++kh) {
        const int64_t t = h + meta.padding - kh;
        if (t < 0 || t % meta.stride != 0) {
            continue;
        }
        const int64_t oh = t / meta.stride;
        if (oh >= meta.output_h) {
            continue;
        }
        for (int64_t kw = 0; kw < meta.kernel_w; ++kw) {
            const int64_t u = w + meta.padding - kw;
            if (u < 0 || u % meta.stride != 0) {
                continue;
            }
            const int64_t ow = u / meta.stride;
            if (ow >= meta.output_w) {
                continue;
            }
            const int64_t k = (c * meta.kernel_h + kh) * meta.kernel_w + kw;
            const int64_t p = oh * meta.output_w + ow;
            sum += columns[(n * meta.kernel_elems + k) * meta.patches + p];
        }
    }
    grad_input[idx] = sum;
}

// NOTE: meta is taken by value (not const-ref) on purpose. Dispatcher::Call
// deduces argument types by value, and a trivially-copyable struct of this
// size is passed on the stack by value but as a pointer when bound to const&,
// so const-ref here would read garbage through the type-erased call.
std::shared_ptr<Tensor> Im2colForward(const std::shared_ptr<Tensor> &input, Conv2dMeta meta) {
    CHECK(input->Dtype() == DataType::kFLOAT32) << "CUDA Im2colForward currently supports float32 only";
    const auto &dims = input->Dims();
    CHECK_EQ(dims.size(), 4) << "Im2colForward input must be NCHW";
    CHECK_EQ(dims[0], meta.batch);
    CHECK_EQ(dims[1], meta.in_channels);
    CHECK_EQ(dims[2], meta.input_h);
    CHECK_EQ(dims[3], meta.input_w);

    auto columns = std::make_shared<Tensor>(std::vector<int64_t>{meta.batch, meta.kernel_elems, meta.patches},
                                            DataType::kFLOAT32, input->GetDevice());
    const int64_t total = meta.batch * meta.kernel_elems * meta.patches;
    if (total == 0) {
        return columns;
    }

    const int threads_per_block = 256;
    const int num_blocks = static_cast<int>((total + threads_per_block - 1) / threads_per_block);
    auto device = input->GetDevice();
    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();
    Im2colKernel<<<num_blocks, threads_per_block, 0, cuda_stream>>>(static_cast<const float *>(input->DataPtr()),
                                                                    static_cast<float *>(columns->DataPtr()), meta);
    return columns;
}

// Forward via im2col + framework GEMM (FP32 only). Weight OIHW is viewed as a
// row-major [Cout, K] matrix (K = Cin * Kh * Kw matches Im2colKernel's row
// order); per image Y[Cout, P] = W * Xcol[K, P].
// NOTE: this entry point mirrors the CPU signature (shared_ptr const&, int64
// by value) so the dispatcher resolves the identical "Conv2dForward" overload
// on CUDA.
std::shared_ptr<Tensor> Conv2dForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                                      const std::shared_ptr<Tensor> &bias, int64_t stride, int64_t padding) {
    CHECK(input->Dtype() == DataType::kFLOAT32) << "CUDA Conv2dForward currently supports float32 only";
    CHECK(weight->Dtype() == DataType::kFLOAT32) << "CUDA Conv2dForward currently supports float32 only";
    if (bias) {
        CHECK(bias->Dtype() == DataType::kFLOAT32) << "CUDA Conv2dForward currently supports float32 only";
    }

    const Conv2dMeta meta = MakeConv2dMeta(input, weight, stride, padding);
    if (bias) {
        CHECK_EQ(bias->Dims().size(), 1);
        CHECK_EQ(bias->Dims()[0], meta.out_channels);
    }

    auto device = input->GetDevice();
    auto output = std::make_shared<Tensor>(Conv2dOutputDims(meta), DataType::kFLOAT32, device);
    if (meta.batch == 0) {
        return output;
    }

    auto columns = Im2colForward(input, meta);

    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();

    // Bias is pre-staged into the output and accumulated with beta=1 (same
    // pattern as cuda::LinearForward); without bias beta=0 overwrites output.
    float beta = 0.0f;
    if (bias) {
        const int64_t total = meta.batch * meta.out_channels * meta.patches;
        const int threads_per_block = 256;
        const int num_blocks = static_cast<int>((total + threads_per_block - 1) / threads_per_block);
        ConvBiasCopyKernel<<<num_blocks, threads_per_block, 0, cuda_stream>>>(
            static_cast<float *>(output->DataPtr()), static_cast<const float *>(bias->DataPtr()), meta.out_channels,
            meta.patches, total);
        beta = 1.0f;
    }

    // cuBLAS is column-major: row-major Y[Cout, P] = W[Cout, K] * X[K, P] is
    // computed as Y^T[P, Cout] = X^T[P, K] * W^T[K, Cout].
    // C = Y^T[P, Cout], A = X^T[P, K], B = W^T[K, Cout].
    // One strided-batched GEMM over the batch: weight is shared across images
    // (stride_b = 0); batch_count == 1 uses zero strides per Gemm convention.
    const int batch_count = static_cast<int>(meta.batch);
    const bool batched = meta.batch > 1;
    Dispatcher::Instance().Call<void>({device.type(), "Gemm"}, device,
                                      GemmParams{
                                          .trans_a = GemmTranspose::kNoTranspose,
                                          .trans_b = GemmTranspose::kNoTranspose,
                                          .m = static_cast<int>(meta.patches),
                                          .n = static_cast<int>(meta.out_channels),
                                          .k = static_cast<int>(meta.kernel_elems),
                                          .A = columns->DataPtr(),
                                          .lda = static_cast<int>(meta.patches),
                                          .B = weight->DataPtr(),
                                          .ldb = static_cast<int>(meta.kernel_elems),
                                          .C = output->DataPtr(),
                                          .ldc = static_cast<int>(meta.patches),
                                          .alpha = 1.0f,
                                          .beta = beta,
                                          .batch_count = batch_count,
                                          .stride_a = batched ? meta.kernel_elems * meta.patches : 0,
                                          .stride_b = 0,
                                          .stride_c = batched ? meta.out_channels * meta.patches : 0,
                                          .input_dtype = DataType::kFLOAT32,
                                          .output_dtype = DataType::kFLOAT32,
                                      });
    return output;
}

// Weight grad via im2col + framework GEMM (FP32 only). Per image,
// row-major dW_n[Cout, K] = dY_n[Cout, P] * Xcol_n[K, P]^T (K = Cin*Kh*Kw,
// P = Hout*Wout matches Im2colKernel's row order), summed over N.
// NOTE: this entry point mirrors the CPU signature (shared_ptr const&,
// int64 by value, vector<int64_t> const&) so the dispatcher resolves the
// identical "Conv2dBackwardWeight" overload on CUDA.
std::shared_ptr<Tensor> Conv2dBackwardWeight(const std::shared_ptr<Tensor> &input,
                                             const std::shared_ptr<Tensor> &grad_output, int64_t stride,
                                             int64_t padding, const std::vector<int64_t> &weight_dims) {
    CHECK(input->Dtype() == DataType::kFLOAT32) << "CUDA Conv2dBackwardWeight currently supports float32 only";
    CHECK(grad_output->Dtype() == DataType::kFLOAT32) << "CUDA Conv2dBackwardWeight currently supports float32 only";
    CHECK_EQ(weight_dims.size(), 4);

    const auto &in_dims = input->Dims();
    CHECK_EQ(in_dims.size(), 4);
    const auto &go_dims = grad_output->Dims();
    CHECK_EQ(go_dims.size(), 4);
    CHECK_EQ(go_dims[0], in_dims[0]);
    CHECK_EQ(weight_dims[0], go_dims[1]);
    CHECK_EQ(weight_dims[1], in_dims[1]);
    CHECK_EQ(go_dims[2], (in_dims[2] + 2 * padding - weight_dims[2]) / stride + 1);
    CHECK_EQ(go_dims[3], (in_dims[3] + 2 * padding - weight_dims[3]) / stride + 1);

    Conv2dMeta meta;
    meta.batch = in_dims[0];
    meta.in_channels = in_dims[1];
    meta.input_h = in_dims[2];
    meta.input_w = in_dims[3];
    meta.out_channels = go_dims[1];
    meta.kernel_h = weight_dims[2];
    meta.kernel_w = weight_dims[3];
    meta.stride = stride;
    meta.padding = padding;
    meta.output_h = go_dims[2];
    meta.output_w = go_dims[3];
    meta.patches = meta.output_h * meta.output_w;
    meta.kernel_elems = meta.in_channels * meta.kernel_h * meta.kernel_w;

    auto device = input->GetDevice();
    auto grad_weight = std::make_shared<Tensor>(weight_dims, DataType::kFLOAT32, device);
    if (meta.batch == 0) {
        return grad_weight;
    }

    auto columns = Im2colForward(input, meta);

    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();

    // cuBLAS is column-major: row-major dW_n[Cout, K] = dY_n[Cout, P] *
    // Xcol_n[K, P]^T is computed as dW_n^T[K, Cout] = op(Xcol)[K, P] *
    // dY_n^T[P, Cout], with Xcol viewed as the column-major [P, K]
    // transpose (trans_a=T) and dY viewed as column-major [P, Cout]
    // (trans_b=N).
    // C = dW^T[K, Cout], A = Xcol[P, K]^T, B = dY^T[P, Cout].
    // Strides are per-image tiles: Xcol K*P, dY Cout*P, dW Cout*K;
    // batch_count == 1 uses zero strides per Gemm convention.
    const int batch_count = static_cast<int>(meta.batch);
    const bool batched = meta.batch > 1;
    void *gemm_c = grad_weight->DataPtr();
    long long stride_c = 0;
    std::shared_ptr<Tensor> per_image;
    if (batched) {
        per_image = std::make_shared<Tensor>(std::vector<int64_t>{meta.batch, meta.out_channels, meta.kernel_elems},
                                             DataType::kFLOAT32, device);
        gemm_c = per_image->DataPtr();
        stride_c = meta.out_channels * meta.kernel_elems;
    }
    Dispatcher::Instance().Call<void>({device.type(), "Gemm"}, device,
                                      GemmParams{
                                          .trans_a = GemmTranspose::kTranspose,
                                          .trans_b = GemmTranspose::kNoTranspose,
                                          .m = static_cast<int>(meta.kernel_elems),
                                          .n = static_cast<int>(meta.out_channels),
                                          .k = static_cast<int>(meta.patches),
                                          .A = columns->DataPtr(),
                                          .lda = static_cast<int>(meta.patches),
                                          .B = grad_output->DataPtr(),
                                          .ldb = static_cast<int>(meta.patches),
                                          .C = gemm_c,
                                          .ldc = static_cast<int>(meta.kernel_elems),
                                          .alpha = 1.0f,
                                          .beta = 0.0f,
                                          .batch_count = batch_count,
                                          .stride_a = batched ? meta.kernel_elems * meta.patches : 0,
                                          .stride_b = batched ? meta.out_channels * meta.patches : 0,
                                          .stride_c = stride_c,
                                          .input_dtype = DataType::kFLOAT32,
                                          .output_dtype = DataType::kFLOAT32,
                                      });
    if (batched) {
        // N == 1 fast path above overwrote grad_weight directly (beta=0).
        const int64_t elems = meta.out_channels * meta.kernel_elems;
        const int threads_per_block = 256;
        const int num_blocks = static_cast<int>((elems + threads_per_block - 1) / threads_per_block);
        ConvBackwardWeightSumKernel<<<num_blocks, threads_per_block, 0, cuda_stream>>>(
            static_cast<const float *>(per_image->DataPtr()), static_cast<float *>(grad_weight->DataPtr()), meta.batch,
            elems);
    }
    return grad_weight;
}

// Input grad via GEMM + gather col2im (FP32 only). Per image,
// row-major dXcol[K, P] = W^T[K, Cout] * dY[Cout, P] (K = Cin*Kh*Kw,
// P = Hout*Wout matches Im2colKernel's row order), then Col2imKernel
// gathers each input element's covering patches (no atomics, deterministic).
// NOTE: this entry point mirrors the CPU signature (shared_ptr const&,
// int64 by value, vector<int64_t> const&) so the dispatcher resolves the
// identical "Conv2dBackwardInput" overload on CUDA.
std::shared_ptr<Tensor> Conv2dBackwardInput(const std::shared_ptr<Tensor> &weight,
                                            const std::shared_ptr<Tensor> &grad_output, int64_t stride, int64_t padding,
                                            const std::vector<int64_t> &input_dims) {
    CHECK(weight->Dtype() == DataType::kFLOAT32) << "CUDA Conv2dBackwardInput currently supports float32 only";
    CHECK(grad_output->Dtype() == DataType::kFLOAT32) << "CUDA Conv2dBackwardInput currently supports float32 only";
    CHECK_EQ(input_dims.size(), 4);

    const auto &go_dims = grad_output->Dims();
    CHECK_EQ(go_dims.size(), 4);
    const auto &w_dims = weight->Dims();
    CHECK_EQ(w_dims.size(), 4);
    CHECK_EQ(w_dims[0], go_dims[1]);
    CHECK_EQ(w_dims[1], input_dims[1]);
    CHECK_EQ(go_dims[2], (input_dims[2] + 2 * padding - w_dims[2]) / stride + 1);
    CHECK_EQ(go_dims[3], (input_dims[3] + 2 * padding - w_dims[3]) / stride + 1);

    Conv2dMeta meta;
    meta.batch = input_dims[0];
    meta.in_channels = input_dims[1];
    meta.input_h = input_dims[2];
    meta.input_w = input_dims[3];
    meta.out_channels = go_dims[1];
    meta.kernel_h = w_dims[2];
    meta.kernel_w = w_dims[3];
    meta.stride = stride;
    meta.padding = padding;
    meta.output_h = go_dims[2];
    meta.output_w = go_dims[3];
    meta.patches = meta.output_h * meta.output_w;
    meta.kernel_elems = meta.in_channels * meta.kernel_h * meta.kernel_w;

    auto device = grad_output->GetDevice();
    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32, device);
    if (meta.batch == 0) {
        return grad_input;
    }

    auto columns = std::make_shared<Tensor>(std::vector<int64_t>{meta.batch, meta.kernel_elems, meta.patches},
                                            DataType::kFLOAT32, device);

    // cuBLAS is column-major: row-major dXcol[K, P] = W^T[K, Cout] *
    // dY[Cout, P] is computed as dXcol^T[P, K] = dY^T[P, Cout] * W[Cout, K],
    // with dY viewed as the column-major [P, Cout] transpose (trans_a=N)
    // and W viewed as column-major [K, Cout] transposed (trans_b=T).
    // C = dXcol^T[P, K], A = dY^T[P, Cout], B = W[K, Cout]^T.
    // Weight is shared across images (stride_b = 0); batch_count == 1 uses
    // zero strides per Gemm convention.
    const int batch_count = static_cast<int>(meta.batch);
    const bool batched = meta.batch > 1;
    Dispatcher::Instance().Call<void>({device.type(), "Gemm"}, device,
                                      GemmParams{
                                          .trans_a = GemmTranspose::kNoTranspose,
                                          .trans_b = GemmTranspose::kTranspose,
                                          .m = static_cast<int>(meta.patches),
                                          .n = static_cast<int>(meta.kernel_elems),
                                          .k = static_cast<int>(meta.out_channels),
                                          .A = grad_output->DataPtr(),
                                          .lda = static_cast<int>(meta.patches),
                                          .B = weight->DataPtr(),
                                          .ldb = static_cast<int>(meta.kernel_elems),
                                          .C = columns->DataPtr(),
                                          .ldc = static_cast<int>(meta.patches),
                                          .alpha = 1.0f,
                                          .beta = 0.0f,
                                          .batch_count = batch_count,
                                          .stride_a = batched ? meta.out_channels * meta.patches : 0,
                                          .stride_b = 0,
                                          .stride_c = batched ? meta.kernel_elems * meta.patches : 0,
                                          .input_dtype = DataType::kFLOAT32,
                                          .output_dtype = DataType::kFLOAT32,
                                      });

    const int64_t total = meta.batch * meta.in_channels * meta.input_h * meta.input_w;
    if (total == 0) {
        return grad_input;
    }
    const int threads_per_block = 256;
    const int num_blocks = static_cast<int>((total + threads_per_block - 1) / threads_per_block);
    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();
    Col2imKernel<<<num_blocks, threads_per_block, 0, cuda_stream>>>(static_cast<const float *>(columns->DataPtr()),
                                                                    static_cast<float *>(grad_input->DataPtr()), meta);
    return grad_input;
}

// Bias grad via N*P reduction (FP32 only). db[oc] = sum over N*Hout*Wout
// of dY (grad_output is row-major [N, Cout, P], P = Hout * Wout).
// NOTE: this entry point mirrors the CPU signature (shared_ptr const&) so
// the dispatcher resolves the identical "Conv2dBackwardBias" overload on
// CUDA.
std::shared_ptr<Tensor> Conv2dBackwardBias(const std::shared_ptr<Tensor> &grad_output) {
    CHECK(grad_output->Dtype() == DataType::kFLOAT32) << "CUDA Conv2dBackwardBias currently supports float32 only";
    const auto &go_dims = grad_output->Dims();
    CHECK_EQ(go_dims.size(), 4);

    const int64_t out_channels = go_dims[1];
    const int64_t patches = go_dims[2] * go_dims[3];
    const int64_t rows = go_dims[0] * patches;

    auto device = grad_output->GetDevice();
    auto grad_bias = std::make_shared<Tensor>(std::vector<int64_t>{out_channels}, DataType::kFLOAT32, device);
    if (out_channels == 0) {
        return grad_bias;
    }

    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();
    // One block per channel (same pattern as cuda::LinearBackwardBias);
    // rows == 0 writes zeros via the empty BlockReduce sum.
    constexpr int kBlockSize = 256;
    ConvBackwardBiasKernel<kBlockSize><<<out_channels, kBlockSize, 0, cuda_stream>>>(
        static_cast<const float *>(grad_output->DataPtr()), static_cast<float *>(grad_bias->DataPtr()), rows,
        out_channels, patches);
    return grad_bias;
}

} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_CONV_KERNEL(kernel_name)                                                                         \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_CONV_KERNEL(Im2colForward)
REGISTER_CUDA_CONV_KERNEL(Conv2dForward)
REGISTER_CUDA_CONV_KERNEL(Conv2dBackwardInput)
REGISTER_CUDA_CONV_KERNEL(Conv2dBackwardWeight)
REGISTER_CUDA_CONV_KERNEL(Conv2dBackwardBias)

#undef REGISTER_CUDA_CONV_KERNEL
