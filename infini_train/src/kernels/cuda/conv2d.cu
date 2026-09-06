#include <cstdint>
#include <memory>
#include <vector>

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"
#include "infini_train/src/kernels/common/gemm.h"
#include "infini_train/src/kernels/cuda/common/gemm.cuh"

namespace infini_train::kernels::cuda {
namespace {

constexpr int kThreadsPerBlock = 256;

// Conv kernels only support fp32, which covers the MNIST CNN training use case.
void CheckFloat32(const std::shared_ptr<Tensor> &tensor, const char *name) {
    CHECK_EQ(static_cast<int>(tensor->Dtype()), static_cast<int>(DataType::kFLOAT32))
        << "Conv2d kernel expects fp32 " << name;
    CHECK_EQ(static_cast<int>(tensor->GetDevice().type()), static_cast<int>(Device::DeviceType::kCUDA))
        << "Conv2d cuda kernel expects a cuda tensor: " << name;
}

int64_t ConvOutSize(int64_t in_size, int64_t kernel_size, int64_t stride, int64_t padding) {
    CHECK_GT(stride, 0);
    CHECK_GE(padding, 0);
    const int64_t out_size = (in_size + 2 * padding - kernel_size) / stride + 1;
    CHECK_GT(out_size, 0) << "Non-positive convolution output size";
    return out_size;
}

struct Conv2dDims {
    int64_t N;
    int64_t C_in;
    int64_t H;
    int64_t W;
    int64_t C_out;
    int64_t kH;
    int64_t kW;
    int64_t H_out;
    int64_t W_out;
};

Conv2dDims ResolveConv2dDims(const std::vector<int64_t> &input_dims, const std::vector<int64_t> &weight_dims,
                             int64_t stride, int64_t padding) {
    CHECK_EQ(input_dims.size(), 4);
    CHECK_EQ(weight_dims.size(), 4);
    Conv2dDims d{};
    d.N = input_dims[0];
    d.C_in = input_dims[1];
    d.H = input_dims[2];
    d.W = input_dims[3];
    d.C_out = weight_dims[0];
    CHECK_EQ(d.C_in, weight_dims[1]) << "Conv2d input channel mismatch with weight";
    d.kH = weight_dims[2];
    d.kW = weight_dims[3];
    d.H_out = ConvOutSize(d.H, d.kH, stride, padding);
    d.W_out = ConvOutSize(d.W, d.kW, stride, padding);
    return d;
}

cudaStream_t GetCudaStream(const Device &device) {
    return dynamic_cast<infini_train::core::cuda::CudaStream *>(
               infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
        ->cuda_stream();
}

int NumBlocks(int64_t total) { return static_cast<int>((total + kThreadsPerBlock - 1) / kThreadsPerBlock); }

// cuBLAS ignores strides for batchCount==1, but the Gemm wrapper expects them zeroed in that case.
void CallConvGemm(const Device &device, GemmParams params) {
    if (params.batch_count == 1) {
        params.stride_a = 0;
        params.stride_b = 0;
        params.stride_c = 0;
    }
    Gemm(device, params);
}

// One thread per (n, k, p): gathers input[n] (C_in, H, W) into col[n] (K, P); padded regions contribute zeros.
// col row k = (c*kH+kh)*kW+kw holds the input values under the kernel tap (kh, kw) for every output position.
__global__ void Im2ColKernel(const float *input, float *col, int64_t total, int64_t K, int64_t P, int64_t C_in,
                             int64_t H, int64_t W, int64_t kH, int64_t kW, int64_t stride, int64_t padding,
                             int64_t H_out, int64_t W_out) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const int64_t p = idx % P;
    const int64_t k = (idx / P) % K;
    const int64_t n = idx / (P * K);
    const int64_t c = k / (kH * kW);
    const int64_t kh = (k / kW) % kH;
    const int64_t kw = k % kW;
    const int64_t oh = p / W_out;
    const int64_t ow = p % W_out;
    const int64_t ih = oh * stride + kh - padding;
    const int64_t iw = ow * stride + kw - padding;
    const bool in_bounds = ih >= 0 && ih < H && iw >= 0 && iw < W;
    col[idx] = in_bounds ? input[(n * C_in + c) * H * W + ih * W + iw] : 0.0f;
}

// Inverse of Im2ColKernel: scatter-adds grad_col[n] (K, P) into grad_input[n] (C_in, H, W).
__global__ void Col2ImKernel(const float *grad_col, float *grad_input, int64_t total, int64_t K, int64_t P,
                             int64_t C_in, int64_t H, int64_t W, int64_t kH, int64_t kW, int64_t stride,
                             int64_t padding, int64_t H_out, int64_t W_out) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const int64_t p = idx % P;
    const int64_t k = (idx / P) % K;
    const int64_t n = idx / (P * K);
    const int64_t c = k / (kH * kW);
    const int64_t kh = (k / kW) % kH;
    const int64_t kw = k % kW;
    const int64_t oh = p / W_out;
    const int64_t ow = p % W_out;
    const int64_t ih = oh * stride + kh - padding;
    const int64_t iw = ow * stride + kw - padding;
    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
        atomicAdd(&grad_input[(n * C_in + c) * H * W + ih * W + iw], grad_col[idx]);
    }
}

// output (N, C_out, P): adds bias[c] to every spatial position of channel c.
__global__ void BiasAddKernel(float *output, const float *bias, int64_t total, int64_t C_out, int64_t P) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const int64_t c = (idx / P) % C_out;
    output[idx] += bias[c];
}

// grad_bias (C_out): one block per output channel, sums over batch and spatial positions of
// grad_output (N, C_out, P).
__global__ void BiasBackwardKernel(const float *grad_output, float *grad_bias, int64_t N, int64_t C_out, int64_t P) {
    const int64_t c = blockIdx.x;
    if (c >= C_out) {
        return;
    }
    float sum = 0.0f;
    for (int64_t i = threadIdx.x; i < N * P; i += blockDim.x) {
        sum += grad_output[(i / P) * C_out * P + c * P + (i % P)];
    }
    atomicAdd(&grad_bias[c], sum);
}

// out(idx) = sum over the leading batch dimension of batched(b, idx).
__global__ void BatchSumKernel(const float *batched, float *out, int64_t batch, int64_t num_elements) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_elements) {
        return;
    }
    float sum = 0.0f;
    for (int64_t b = 0; b < batch; ++b) { sum += batched[b * num_elements + idx]; }
    out[idx] = sum;
}
} // namespace

std::shared_ptr<Tensor> Conv2dForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                                      const std::shared_ptr<Tensor> &bias, int64_t stride, int64_t padding) {
    /*
    input: (N, C_in, H, W), weight: (C_out, C_in, kH, kW), bias: (C_out)
    output: (N, C_out, H_out, W_out), where H_out = (H + 2*padding - kH) / stride + 1
    */
    CheckFloat32(input, "input");
    CheckFloat32(weight, "weight");
    if (bias) {
        CheckFloat32(bias, "bias");
        CHECK_EQ(bias->Dims().size(), 1);
        CHECK_EQ(bias->Dims()[0], weight->Dims()[0]);
    }

    const Conv2dDims d = ResolveConv2dDims(input->Dims(), weight->Dims(), stride, padding);
    const int64_t K = d.C_in * d.kH * d.kW;
    const int64_t P = d.H_out * d.W_out;

    auto device = input->GetDevice();
    const auto cuda_stream = GetCudaStream(device);

    auto output
        = std::make_shared<Tensor>(std::vector<int64_t>{d.N, d.C_out, d.H_out, d.W_out}, DataType::kFLOAT32, device);
    auto col = std::make_shared<Tensor>(std::vector<int64_t>{d.N * K * P}, DataType::kFLOAT32, device);

    const int64_t im2col_total = d.N * K * P;
    Im2ColKernel<<<NumBlocks(im2col_total), kThreadsPerBlock, 0, cuda_stream>>>(
        static_cast<const float *>(input->DataPtr()), static_cast<float *>(col->DataPtr()), im2col_total, K, P, d.C_in,
        d.H, d.W, d.kH, d.kW, stride, padding, d.H_out, d.W_out);

    // out_n (C_out, P) = weight (C_out, K) * col_n (K, P) per sample; cuBLAS is column-major, so a row-major GEMM
    // C(M, N) = L(M, K) * R(K, N) maps to op(A)=R, op(B)=L with m=N, n=M, k=K (see linear.cu).
    CallConvGemm(device, GemmParams{
                             .trans_a = GemmTranspose::kNoTranspose,
                             .trans_b = GemmTranspose::kNoTranspose,
                             .m = static_cast<int>(P),
                             .n = static_cast<int>(d.C_out),
                             .k = static_cast<int>(K),
                             .A = col->DataPtr(),
                             .lda = static_cast<int>(P),
                             .B = weight->DataPtr(),
                             .ldb = static_cast<int>(K),
                             .C = output->DataPtr(),
                             .ldc = static_cast<int>(P),
                             .alpha = 1.0f,
                             .beta = 0.0f,
                             .batch_count = static_cast<int>(d.N),
                             .stride_a = K * P,
                             .stride_b = 0,
                             .stride_c = d.C_out * P,
                             .input_dtype = DataType::kFLOAT32,
                             .output_dtype = DataType::kFLOAT32,
                         });

    if (bias) {
        const int64_t total = d.N * d.C_out * P;
        BiasAddKernel<<<NumBlocks(total), kThreadsPerBlock, 0, cuda_stream>>>(
            static_cast<float *>(output->DataPtr()), static_cast<const float *>(bias->DataPtr()), total, d.C_out, P);
    }
    return output;
}

std::shared_ptr<Tensor> Conv2dBackwardInput(const std::shared_ptr<Tensor> &weight,
                                            const std::shared_ptr<Tensor> &grad_output, int64_t stride, int64_t padding,
                                            const std::vector<int64_t> &input_dims) {
    /*
    grad_input = col2im(weight^T * grad_output)
    weight: (C_out, C_in, kH, kW), grad_output: (N, C_out, H_out, W_out), grad_input: (N, C_in, H, W)
    */
    CheckFloat32(weight, "weight");
    CheckFloat32(grad_output, "grad_output");

    const Conv2dDims d = ResolveConv2dDims(input_dims, weight->Dims(), stride, padding);
    const auto &grad_output_dims = grad_output->Dims();
    CHECK_EQ(grad_output_dims.size(), 4);
    CHECK_EQ(grad_output_dims[0], d.N);
    CHECK_EQ(grad_output_dims[1], d.C_out);
    CHECK_EQ(grad_output_dims[2], d.H_out);
    CHECK_EQ(grad_output_dims[3], d.W_out);

    const int64_t K = d.C_in * d.kH * d.kW;
    const int64_t P = d.H_out * d.W_out;

    auto device = grad_output->GetDevice();
    const auto cuda_stream = GetCudaStream(device);

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32, device);
    CUDA_CHECK(cudaMemsetAsync(grad_input->DataPtr(), 0, grad_input->SizeInBytes(), cuda_stream));
    auto grad_col = std::make_shared<Tensor>(std::vector<int64_t>{d.N * K * P}, DataType::kFLOAT32, device);

    // grad_col_n (K, P) = weight^T (K, C_out) * grad_out_n (C_out, P): op(B) transposes the weight.
    CallConvGemm(device, GemmParams{
                             .trans_a = GemmTranspose::kNoTranspose,
                             .trans_b = GemmTranspose::kTranspose,
                             .m = static_cast<int>(P),
                             .n = static_cast<int>(K),
                             .k = static_cast<int>(d.C_out),
                             .A = grad_output->DataPtr(),
                             .lda = static_cast<int>(P),
                             .B = weight->DataPtr(),
                             .ldb = static_cast<int>(K),
                             .C = grad_col->DataPtr(),
                             .ldc = static_cast<int>(P),
                             .alpha = 1.0f,
                             .beta = 0.0f,
                             .batch_count = static_cast<int>(d.N),
                             .stride_a = d.C_out * P,
                             .stride_b = 0,
                             .stride_c = K * P,
                             .input_dtype = DataType::kFLOAT32,
                             .output_dtype = DataType::kFLOAT32,
                         });

    const int64_t col2im_total = d.N * K * P;
    Col2ImKernel<<<NumBlocks(col2im_total), kThreadsPerBlock, 0, cuda_stream>>>(
        static_cast<const float *>(grad_col->DataPtr()), static_cast<float *>(grad_input->DataPtr()), col2im_total, K,
        P, d.C_in, d.H, d.W, d.kH, d.kW, stride, padding, d.H_out, d.W_out);
    return grad_input;
}

std::shared_ptr<Tensor> Conv2dBackwardWeight(const std::shared_ptr<Tensor> &input,
                                             const std::shared_ptr<Tensor> &grad_output, int64_t stride,
                                             int64_t padding, const std::vector<int64_t> &weight_dims) {
    /*
    grad_weight = sum_n grad_out_n (C_out, P) * col(input[n])^T (P, K)
    input: (N, C_in, H, W), grad_output: (N, C_out, H_out, W_out), grad_weight: (C_out, C_in, kH, kW)
    */
    CheckFloat32(input, "input");
    CheckFloat32(grad_output, "grad_output");
    CHECK_EQ(weight_dims.size(), 4);

    const Conv2dDims d = ResolveConv2dDims(input->Dims(), weight_dims, stride, padding);
    const auto &grad_output_dims = grad_output->Dims();
    CHECK_EQ(grad_output_dims.size(), 4);
    CHECK_EQ(grad_output_dims[0], d.N);
    CHECK_EQ(grad_output_dims[1], d.C_out);
    CHECK_EQ(grad_output_dims[2], d.H_out);
    CHECK_EQ(grad_output_dims[3], d.W_out);

    const int64_t K = d.C_in * d.kH * d.kW;
    const int64_t P = d.H_out * d.W_out;

    auto device = grad_output->GetDevice();
    const auto cuda_stream = GetCudaStream(device);

    auto grad_weight = std::make_shared<Tensor>(weight_dims, DataType::kFLOAT32, device);
    auto grad_weight_batched
        = std::make_shared<Tensor>(std::vector<int64_t>{d.N * d.C_out * K}, DataType::kFLOAT32, device);
    auto col = std::make_shared<Tensor>(std::vector<int64_t>{d.N * K * P}, DataType::kFLOAT32, device);

    const int64_t im2col_total = d.N * K * P;
    Im2ColKernel<<<NumBlocks(im2col_total), kThreadsPerBlock, 0, cuda_stream>>>(
        static_cast<const float *>(input->DataPtr()), static_cast<float *>(col->DataPtr()), im2col_total, K, P, d.C_in,
        d.H, d.W, d.kH, d.kW, stride, padding, d.H_out, d.W_out);

    // Per-sample grad_w_n (C_out, K) = grad_out_n (C_out, P) * col_n^T (P, K): op(A) transposes col.
    CallConvGemm(device, GemmParams{
                             .trans_a = GemmTranspose::kTranspose,
                             .trans_b = GemmTranspose::kNoTranspose,
                             .m = static_cast<int>(K),
                             .n = static_cast<int>(d.C_out),
                             .k = static_cast<int>(P),
                             .A = col->DataPtr(),
                             .lda = static_cast<int>(P),
                             .B = grad_output->DataPtr(),
                             .ldb = static_cast<int>(P),
                             .C = grad_weight_batched->DataPtr(),
                             .ldc = static_cast<int>(K),
                             .alpha = 1.0f,
                             .beta = 0.0f,
                             .batch_count = static_cast<int>(d.N),
                             .stride_a = K * P,
                             .stride_b = d.C_out * P,
                             .stride_c = d.C_out * K,
                             .input_dtype = DataType::kFLOAT32,
                             .output_dtype = DataType::kFLOAT32,
                         });

    BatchSumKernel<<<NumBlocks(d.C_out * K), kThreadsPerBlock, 0, cuda_stream>>>(
        static_cast<const float *>(grad_weight_batched->DataPtr()), static_cast<float *>(grad_weight->DataPtr()), d.N,
        d.C_out * K);
    return grad_weight;
}

std::shared_ptr<Tensor> Conv2dBackwardBias(const std::shared_ptr<Tensor> &grad_output) {
    /*
    grad_bias = sum over batch and spatial positions of grad_output
    grad_output: (N, C_out, H_out, W_out), grad_bias: (C_out)
    */
    CheckFloat32(grad_output, "grad_output");
    const auto &grad_output_dims = grad_output->Dims();
    CHECK_EQ(grad_output_dims.size(), 4);
    const int64_t N = grad_output_dims[0];
    const int64_t C_out = grad_output_dims[1];
    const int64_t P = grad_output_dims[2] * grad_output_dims[3];

    auto device = grad_output->GetDevice();
    const auto cuda_stream = GetCudaStream(device);

    auto grad_bias = std::make_shared<Tensor>(std::vector<int64_t>{C_out}, DataType::kFLOAT32, device);
    CUDA_CHECK(cudaMemsetAsync(grad_bias->DataPtr(), 0, grad_bias->SizeInBytes(), cuda_stream));
    BiasBackwardKernel<<<static_cast<int>(C_out), kThreadsPerBlock, 0, cuda_stream>>>(
        static_cast<const float *>(grad_output->DataPtr()), static_cast<float *>(grad_bias->DataPtr()), N, C_out, P);
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
