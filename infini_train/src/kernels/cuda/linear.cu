#include <functional>
#include <memory>
#include <numeric>
#include <vector>

#include <cub/block/block_reduce.cuh>
#include <cublas_v2.h>

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/common/cuda/gemm.cuh"
#include "infini_train/include/common/cuda/kernel_helper.cuh"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {

template <typename T> __global__ void BiasCopyKernel(T *output, const T *bias, int bs, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= bs * out_features) {
        return;
    }
    int j = idx % out_features;
    output[idx] = bias[j];
}

std::shared_ptr<Tensor> LinearForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                                      bool transpose, const std::shared_ptr<Tensor> &bias) {

    /*
        !transpose: output = input * weight + bias
        output[*, out_features] = input[*, in_features] * weight[in_features, out_features] + bias[out_features]

        transpose:  output = input * weight^T + bias
        output[*, out_features] = input[*, in_features] * weight[out_features, in_features]^T + bias[out_features]
    */

    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);

    // As for cublas:
    // C = alpha * op(B) * op(A) + beta * C
    // Dimensions:
    //   input:  (bs, in_features)
    //   weight: (in_features, out_features) or (out_features, in_features) if transposed
    //   output: (bs, out_features)
    const int64_t out_features = weight_dims[transpose ? 0 : 1];

    auto dtype = input->Dtype();
    auto output_dims = input_dims;
    *output_dims.rbegin() = out_features;
    auto output = std::make_shared<Tensor>(output_dims, dtype, input->GetDevice());

    auto device = input->GetDevice();
    const auto cuda_stream = GetCudaStream(device);

    if (bias) {
        CHECK_EQ(bias->Dims().size(), 1);
        CHECK_EQ(bias->Dims()[0], out_features);
        int threads_per_block = 256;
        int num_blocks = (bs * out_features + threads_per_block - 1) / threads_per_block;

        DispatchFunc<DataType::kFLOAT32, DataType::kBFLOAT16>(
            dtype,
            [=]<typename T>() {
                BiasCopyKernel<<<num_blocks, threads_per_block, 0, cuda_stream>>>(
                    static_cast<T *>(output->DataPtr()), static_cast<const T *>(bias->DataPtr()), bs, out_features);
            },
            "CUDA LinearForward");
    } else {
        DispatchFunc<DataType::kFLOAT32, DataType::kBFLOAT16>(
            input->Dtype(), [=]<typename T>() { output->Fill<T>(0); }, "CUDA LinearForward");
    }

    // TODO(zbl): use cublasSgemv if possible for convenience and simplicity
    //
    // - if a is transposed:
    // weight is [out_features, in_features] here
    // output = input * weight.T --> output.T = weight * input.T
    // C = output.T[out_features, bs]
    // A = weight.T[in_features, out_features]
    // B = input.T[in_features, bs]
    //
    // - if a is not transposed:
    // output = input * weight --> output.T =  weight.T * input.T
    // C = output.T[out_features, bs]
    // A = weight.T[out_features, in_features]
    // B = input.T[in_features, bs]
    GemmParams p;
    p.trans_a = transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
    p.trans_b = CUBLAS_OP_N;
    p.m = static_cast<int>(out_features);
    p.n = static_cast<int>(bs);
    p.k = static_cast<int>(in_features);
    p.A = weight->DataPtr();
    p.lda = static_cast<int>(transpose ? in_features : out_features);
    p.B = input->DataPtr();
    p.ldb = static_cast<int>(in_features);
    p.C = output->DataPtr();
    p.ldc = static_cast<int>(out_features);
    p.alpha = 1.0f;
    p.beta = 1.0f; // bias already written into output; beta=1 accumulates
    p.batch_count = 1;
    p.input_dtype = dtype;
    p.output_dtype = dtype;
    p.blas_handle = GetCublasHandle(device);

    GemmCuda(p);

    return output;
}

template <int BLOCK_SIZE, typename TIn, typename TOut>
__global__ void ReduceColumnsKernel(const TIn *__restrict__ input, TOut *__restrict__ output, int num_rows,
                                    int num_cols) {
    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int row = blockIdx.x;
    float sum = 0.0f;

    for (int col = threadIdx.x; col < num_cols; col += blockDim.x) {
        sum += common::cuda::Cast<float>(input[row * num_cols + col]);
    }

    float reduced = BlockReduce(temp_storage).Sum(sum);

    if (threadIdx.x == 0) {
        output[row] = reduced;
    }
}

std::shared_ptr<Tensor> LinearBackwardInput(const std::shared_ptr<Tensor> &weight,
                                            const std::shared_ptr<Tensor> &grad_output, bool transpose,
                                            int64_t in_features, int64_t out_features,
                                            const std::vector<int64_t> &input_dims) {
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});

    auto compute_dtype = weight->Dtype();
    auto grad_output_dtype = grad_output->Dtype();
    auto grad_output_promoted
        = grad_output_dtype == compute_dtype ? grad_output : std::make_shared<Tensor>(grad_output->To(compute_dtype));

    // For bf16 compute, accumulate in fp32 to preserve precision.
    auto output_dtype = (compute_dtype == DataType::kBFLOAT16) ? DataType::kFLOAT32 : compute_dtype;
    // No Fill(0) needed: cuBLAS beta=0.0f fully overwrites output.
    auto grad_input = std::make_shared<Tensor>(input_dims, output_dtype, grad_output->GetDevice());

    // TODO(zbl): use cublasSgemv if possible
    // - if transpose:
    // weight is [out_features, in_features] here
    // d_input = d_output * weight --> d_input.T = weight.T * d_output.T
    // C = d_input.T[in_features, bs]
    // A = weight.T[in_features, out_features]
    // B = d_output.T[out_features, bs]
    //
    // - if not transpose:
    // weight is [in_features, out_features] here
    // d_input = d_output * weight.T --> d_input.T = weight * d_output.T
    // C = d_input.T[in_features, bs]
    // A = weight.T[out_features, in_features]
    // B = d_output.T[out_features, bs]
    GemmParams p;
    p.trans_a = transpose ? CUBLAS_OP_N : CUBLAS_OP_T;
    p.trans_b = CUBLAS_OP_N;
    p.m = static_cast<int>(in_features);
    p.n = static_cast<int>(bs);
    p.k = static_cast<int>(out_features);
    p.A = weight->DataPtr();
    p.lda = static_cast<int>(transpose ? in_features : out_features);
    p.B = grad_output_promoted->DataPtr();
    p.ldb = static_cast<int>(out_features);
    p.C = grad_input->DataPtr();
    p.ldc = static_cast<int>(in_features);
    p.alpha = 1.0f;
    p.beta = 0.0f;
    p.batch_count = 1;
    p.input_dtype = compute_dtype;
    p.output_dtype = output_dtype;
    p.blas_handle = GetCublasHandle(grad_output->GetDevice());

    GemmCuda(p);

    return grad_input;
}

std::shared_ptr<Tensor> LinearBackwardWeight(const std::shared_ptr<Tensor> &input,
                                             const std::shared_ptr<Tensor> &grad_output, bool transpose,
                                             int64_t in_features, int64_t out_features) {
    const auto &grad_output_dims = grad_output->Dims();
    CHECK_GE(grad_output_dims.size(), 2);
    const int64_t bs
        = std::accumulate(grad_output_dims.rbegin() + 1, grad_output_dims.rend(), 1, std::multiplies<int64_t>{});

    auto compute_dtype = input->Dtype();
    auto grad_output_dtype = grad_output->Dtype();
    auto grad_output_promoted
        = grad_output_dtype == compute_dtype ? grad_output : std::make_shared<Tensor>(grad_output->To(compute_dtype));

    // For bf16 compute, accumulate in fp32 to preserve precision.
    auto output_dtype = (compute_dtype == DataType::kBFLOAT16) ? DataType::kFLOAT32 : compute_dtype;
    const std::vector<int64_t> weight_dims
        = transpose ? std::vector<int64_t>{out_features, in_features} : std::vector<int64_t>{in_features, out_features};
    // No Fill(0) needed: cuBLAS beta=0.0f fully overwrites output.
    auto grad_weight = std::make_shared<Tensor>(weight_dims, output_dtype, grad_output->GetDevice());

    // - if transpose:
    // d_weight = d_output.T * input --> d_weight.T = input.T * d_output
    // C = d_weight.T[in_features, out_features]
    // A = input.T[in_features, bs]
    // B = d_output.T[out_features, bs]
    //
    // - if not transpose:
    // d_weight = input.T * d_output --> d_weight.T = d_output.T * input
    // C = d_weight.T[out_features, in_features]
    // A = d_output.T[out_features, bs]
    // B = input.T[in_features, bs]
    const void *a = transpose ? input->DataPtr() : grad_output_promoted->DataPtr();
    const void *b = transpose ? grad_output_promoted->DataPtr() : input->DataPtr();
    const int lda = static_cast<int>(transpose ? in_features : out_features);
    const int ldb = static_cast<int>(transpose ? out_features : in_features);

    GemmParams p;
    p.trans_a = CUBLAS_OP_N;
    p.trans_b = CUBLAS_OP_T;
    p.m = static_cast<int>(transpose ? in_features : out_features);
    p.n = static_cast<int>(transpose ? out_features : in_features);
    p.k = static_cast<int>(bs);
    p.A = a;
    p.lda = lda;
    p.B = b;
    p.ldb = ldb;
    p.C = grad_weight->DataPtr();
    p.ldc = static_cast<int>(transpose ? in_features : out_features);
    p.alpha = 1.0f;
    p.beta = 0.0f;
    p.batch_count = 1;
    p.input_dtype = compute_dtype;
    p.output_dtype = output_dtype;
    p.blas_handle = GetCublasHandle(grad_output->GetDevice());

    GemmCuda(p);

    return grad_weight;
}

std::shared_ptr<Tensor> LinearBackwardBias(const std::shared_ptr<Tensor> &grad_output, int64_t out_features) {
    const auto &dims = grad_output->Dims();
    CHECK_GE(dims.size(), 2);
    const int64_t bs = std::accumulate(dims.rbegin() + 1, dims.rend(), 1, std::multiplies<int64_t>{});

    auto compute_dtype = grad_output->Dtype();
    // For bf16 compute, accumulate in fp32 to preserve precision.
    auto output_dtype = (compute_dtype == DataType::kBFLOAT16) ? DataType::kFLOAT32 : compute_dtype;
    auto grad_bias
        = std::make_shared<Tensor>(std::vector<int64_t>{out_features}, output_dtype, grad_output->GetDevice());

    auto device = grad_output->GetDevice();
    const auto cuda_stream = GetCudaStream(device);

    // d_bias = \sum_i(i=0, bs-1) d_output[i]
    // TODO(dcj): use thrust::fill or reduce kernel do this
    constexpr int BLOCK_SIZE = 256;
    switch (compute_dtype) {
        DISPATCH_CASE(WRAP({
                          ReduceColumnsKernel<BLOCK_SIZE><<<out_features, BLOCK_SIZE, 0, cuda_stream>>>(
                              static_cast<const float *>(grad_output->DataPtr()),
                              static_cast<float *>(grad_bias->DataPtr()), out_features, bs);
                      }),
                      DataType::kFLOAT32)
        DISPATCH_CASE(WRAP({
                          ReduceColumnsKernel<BLOCK_SIZE><<<out_features, BLOCK_SIZE, 0, cuda_stream>>>(
                              static_cast<const nv_bfloat16 *>(grad_output->DataPtr()),
                              static_cast<float *>(grad_bias->DataPtr()), out_features, bs);
                      }),
                      DataType::kBFLOAT16)
    }

    return grad_bias;
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_LINEAR_KERNEL(kernel_name)                                                                       \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_LINEAR_KERNEL(LinearForward)
REGISTER_CUDA_LINEAR_KERNEL(LinearBackwardInput)
REGISTER_CUDA_LINEAR_KERNEL(LinearBackwardWeight)
REGISTER_CUDA_LINEAR_KERNEL(LinearBackwardBias)

#undef REGISTER_CUDA_LINEAR_KERNEL
