#include <functional>
#include <memory>
#include <numeric>
#include <vector>

#include <cub/block/block_reduce.cuh>
#include <cublas_v2.h>

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/common/cuda/kernel_helper.cuh"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

namespace infini_train::kernels::cuda {

std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
    /*
     output[*, m, n] = input[*, m, k] * other[*, k, n]
     */
    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();

    CHECK_GE(input_dims.size(), 2);
    CHECK_GE(other_dims.size(), 2);
    CHECK_EQ(input_dims.size(), other_dims.size());

    const int64_t m = input_dims[input_dims.size() - 2];
    const int64_t k = input_dims[input_dims.size() - 1];
    CHECK_EQ(k, other_dims[other_dims.size() - 2]);
    const int64_t n = other_dims[other_dims.size() - 1];

    const int64_t bs = std::accumulate(input_dims.rbegin() + 2, input_dims.rend(), 1, std::multiplies<int64_t>{});
    for (int64_t i = 0; i < input_dims.size() - 2; ++i) {
        CHECK_EQ(input_dims[i], other_dims[i]) << "Batch dims must match";
    }

    auto dtype = input->Dtype();
    std::vector<int64_t> output_dims = input_dims;
    output_dims[output_dims.size() - 1] = n;
    auto output = std::make_shared<Tensor>(output_dims, dtype, input->GetDevice());

    auto device = input->GetDevice();
    const float alpha = 1.0f, beta = 0.0f;
    cublasHandle_t handle = dynamic_cast<infini_train::core::cuda::CudaBlasHandle *>(
                                infini_train::core::GetDeviceGuardImpl(device.type())->GetBlasHandle(device))
                                ->cublas_handle();

    // cuBLAS is colmun-major
    // output = input * other --> output.T = other.T * input.T
    // C = A * B ==> output.T[*, n, m] = other.T[*, n, k] * input.T[*, k, m]
    // C = output.T[*, n, m]
    // A = other.T[*, n, k]
    // B = input.T[*, k, m]
    int lda = n;
    int ldb = k;
    int ldc = n;
    int64_t stride_a = n * k;
    int64_t stride_b = k * m;
    int64_t stride_c = m * n;
    // NOTE(zbl): the last cublasGemmAlgo_t param has no effect on GPU arch >= sm_80(Ampere)

    switch (dtype) {
        DISPATCH_CASE(WRAP(CUBLAS_CHECK(cublasGemmStridedBatchedEx(
                          handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, other->DataPtr(), CUDA_R_32F, lda,
                          stride_a, input->DataPtr(), CUDA_R_32F, ldb, stride_b, &beta, output->DataPtr(), CUDA_R_32F,
                          ldc, stride_c, bs, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));),
                      DataType::kFLOAT32)
        DISPATCH_CASE(WRAP(CUBLAS_CHECK(cublasGemmStridedBatchedEx(
                          handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, other->DataPtr(), CUDA_R_16BF, lda,
                          stride_a, input->DataPtr(), CUDA_R_16BF, ldb, stride_b, &beta, output->DataPtr(), CUDA_R_16BF,
                          ldc, stride_c, bs, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));),
                      DataType::kBFLOAT16)
    default:
        LOG_UNSUPPORTED_DTYPE(dtype, "CUDA MatmulForward");
    }

    return output;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
MatmulBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other,
               const std::shared_ptr<Tensor> &grad_output) {
    /*
       grad_input[*, m, k] = grad_output[*, m, n] * other[*, k, n]^T
       grad_other[*, k, n] = input[*, m, k]^T * grad_output[*, m, n]
    */

    auto input_dtype = input->Dtype();
    auto other_dtype = other->Dtype();
    auto grad_output_dtype = grad_output->Dtype();
    // Compute dtype determined by saved tensors (forward compute dtype), not grad_output
    DataType compute_dtype = DispatchFunc<DataTypeList<INFINI_ALL_TYPES>, DataTypeList<INFINI_ALL_TYPES>>(
        {input_dtype, other_dtype}, [=]<typename Tin, typename To>() { return DataTypeMap_v<WidestType_t<Tin, To>>; },
        "CUDA MatmulBackward");

    auto input_promoted = input_dtype == compute_dtype ? input : std::make_shared<Tensor>(input->To(compute_dtype));
    auto other_promoted = other_dtype == compute_dtype ? other : std::make_shared<Tensor>(other->To(compute_dtype));
    auto grad_output_promoted
        = grad_output_dtype == compute_dtype ? grad_output : std::make_shared<Tensor>(grad_output->To(compute_dtype));

    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();
    const auto &grad_output_dims = grad_output->Dims();

    CHECK_GE(input_dims.size(), 2);
    CHECK_EQ(input_dims.size(), other_dims.size());
    CHECK_EQ(input_dims.size(), grad_output_dims.size());

    const int64_t m = input_dims[input_dims.size() - 2];
    const int64_t k = input_dims[input_dims.size() - 1];
    const int64_t n = other_dims[other_dims.size() - 1];
    CHECK_EQ(k, other_dims[other_dims.size() - 2]);
    CHECK_EQ(m, grad_output_dims[grad_output_dims.size() - 2]);
    CHECK_EQ(n, grad_output_dims[grad_output_dims.size() - 1]);

    const int64_t bs = std::accumulate(input_dims.rbegin() + 2, input_dims.rend(), 1, std::multiplies<int64_t>{});
    for (int64_t i = 0; i < input_dims.size() - 2; ++i) {
        CHECK_EQ(input_dims[i], other_dims[i]) << "Batch dims must match";
        CHECK_EQ(input_dims[i], grad_output_dims[i]) << "Batch dims must match";
    }

    // For bf16 compute, output in fp32 to preserve accumulation precision (matches PyTorch behavior)
    auto output_dtype = (compute_dtype == DataType::kBFLOAT16) ? DataType::kFLOAT32 : compute_dtype;
    auto grad_input = std::make_shared<Tensor>(input_dims, output_dtype, grad_output->GetDevice());
    auto grad_other = std::make_shared<Tensor>(other_dims, output_dtype, grad_output->GetDevice());

    // No Fill(0) needed: cuBLAS beta=0.0f means C is fully overwritten, never read.

    auto device = input_promoted->GetDevice();
    const float alpha = 1.0f, beta = 0.0f;
    cublasHandle_t handle = dynamic_cast<infini_train::core::cuda::CudaBlasHandle *>(
                                infini_train::core::GetDeviceGuardImpl(device.type())->GetBlasHandle(device))
                                ->cublas_handle();

    {
        // cuBLAS is colmun-major
        // grad_input = grad_output * other.T --> grad_input.T = other * grad_output.T
        // C = A.T * B ==> grad_input.T[*, k, m] = other[*, k, n] * grad_output.T[*, n, m]
        // C = grad_input.T[*, k, m]
        // A = other.T[*, n, k]
        // B = grad_output.T[*, n, m]
        const int lda = n, ldb = n, ldc = k;
        const int64_t stride_a = k * n;
        const int64_t stride_b = n * m;
        const int64_t stride_c = m * k;
        switch (compute_dtype) {
            DISPATCH_CASE(WRAP(CUBLAS_CHECK(cublasGemmStridedBatchedEx(
                              handle, CUBLAS_OP_T, CUBLAS_OP_N, k, m, n, &alpha, other_promoted->DataPtr(), CUDA_R_32F,
                              lda, stride_a, grad_output_promoted->DataPtr(), CUDA_R_32F, ldb, stride_b, &beta,
                              grad_input->DataPtr(), CUDA_R_32F, ldc, stride_c, bs, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));),
                          DataType::kFLOAT32)
            DISPATCH_CASE(WRAP(CUBLAS_CHECK(cublasGemmStridedBatchedEx(
                              handle, CUBLAS_OP_T, CUBLAS_OP_N, k, m, n, &alpha, other_promoted->DataPtr(), CUDA_R_16BF,
                              lda, stride_a, grad_output_promoted->DataPtr(), CUDA_R_16BF, ldb, stride_b, &beta,
                              grad_input->DataPtr(), CUDA_R_32F, ldc, stride_c, bs, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));),
                          DataType::kBFLOAT16)
        }
    }

    {
        // cuBLAS is colmun-major
        // grad_other = input.T * grad_output --> grad_other.T =  grad_output.T * input
        // C = A * B.T ==> grad_other.T[*, n, k] = grad_output.T[*, n, m] * input[*, m, k]
        // C = grad_other.T[*, n, k]
        // A = grad_output.T[*, n, m]
        // B = input.T[*, k, m]
        const int lda = n, ldb = k, ldc = n;
        const int64_t stride_a = n * m;
        const int64_t stride_b = k * m;
        const int64_t stride_c = n * k;
        switch (compute_dtype) {
            DISPATCH_CASE(WRAP(CUBLAS_CHECK(cublasGemmStridedBatchedEx(
                              handle, CUBLAS_OP_N, CUBLAS_OP_T, n, k, m, &alpha, grad_output_promoted->DataPtr(),
                              CUDA_R_32F, lda, stride_a, input_promoted->DataPtr(), CUDA_R_32F, ldb, stride_b, &beta,
                              grad_other->DataPtr(), CUDA_R_32F, ldc, stride_c, bs, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));),
                          DataType::kFLOAT32)
            DISPATCH_CASE(WRAP(CUBLAS_CHECK(cublasGemmStridedBatchedEx(
                              handle, CUBLAS_OP_N, CUBLAS_OP_T, n, k, m, &alpha, grad_output_promoted->DataPtr(),
                              CUDA_R_16BF, lda, stride_a, input_promoted->DataPtr(), CUDA_R_16BF, ldb, stride_b, &beta,
                              grad_other->DataPtr(), CUDA_R_32F, ldc, stride_c, bs, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));),
                          DataType::kBFLOAT16)
        }
    }

    return {grad_input, grad_other};
}

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
    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();

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

    const float alpha = 1.0f;
    const float beta = 1.0f;
    auto trans_a = transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
    auto trans_b = CUBLAS_OP_N;
    auto lda = transpose ? in_features : out_features;
    cublasHandle_t handle = dynamic_cast<infini_train::core::cuda::CudaBlasHandle *>(
                                infini_train::core::GetDeviceGuardImpl(device.type())->GetBlasHandle(device))
                                ->cublas_handle();

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
    switch (input->Dtype()) {
        DISPATCH_CASE(WRAP({
                          CUBLAS_CHECK(cublasSgemm(handle, trans_a, trans_b, out_features, bs, in_features, &alpha,
                                                   static_cast<const float *>(weight->DataPtr()), lda,
                                                   static_cast<const float *>(input->DataPtr()), in_features, &beta,
                                                   static_cast<float *>(output->DataPtr()), out_features));
                      }),
                      DataType::kFLOAT32)
        DISPATCH_CASE(WRAP({
                          CUBLAS_CHECK(cublasGemmEx(handle, trans_a, trans_b, out_features, bs, in_features, &alpha,
                                                    weight->DataPtr(), CUDA_R_16BF, lda, input->DataPtr(), CUDA_R_16BF,
                                                    in_features, &beta, output->DataPtr(), CUDA_R_16BF, out_features,
                                                    CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
                      }),
                      DataType::kBFLOAT16)
    }

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

    // For bf16 compute, accumulate in fp32 to preserve precision (matches PyTorch behavior).
    auto output_dtype = (compute_dtype == DataType::kBFLOAT16) ? DataType::kFLOAT32 : compute_dtype;
    // No Fill(0) needed: cuBLAS beta=0.0f fully overwrites output.
    auto grad_input = std::make_shared<Tensor>(input_dims, output_dtype, grad_output->GetDevice());

    auto device = grad_output->GetDevice();
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasHandle_t handle = dynamic_cast<infini_train::core::cuda::CudaBlasHandle *>(
                                infini_train::core::GetDeviceGuardImpl(device.type())->GetBlasHandle(device))
                                ->cublas_handle();

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
    auto trans_a = transpose ? CUBLAS_OP_N : CUBLAS_OP_T;
    auto lda = transpose ? in_features : out_features;

    switch (compute_dtype) {
        DISPATCH_CASE(WRAP({
                          CUBLAS_CHECK(cublasSgemm(handle, trans_a, CUBLAS_OP_N, in_features, bs, out_features, &alpha,
                                                   static_cast<const float *>(weight->DataPtr()), lda,
                                                   static_cast<const float *>(grad_output_promoted->DataPtr()),
                                                   out_features, &beta, static_cast<float *>(grad_input->DataPtr()),
                                                   in_features));
                      }),
                      DataType::kFLOAT32)
        DISPATCH_CASE(WRAP({
                          CUBLAS_CHECK(cublasGemmEx(
                              handle, trans_a, CUBLAS_OP_N, in_features, bs, out_features, &alpha, weight->DataPtr(),
                              CUDA_R_16BF, lda, grad_output_promoted->DataPtr(), CUDA_R_16BF, out_features, &beta,
                              grad_input->DataPtr(), CUDA_R_32F, in_features, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
                      }),
                      DataType::kBFLOAT16)
    }

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

    // For bf16 compute, accumulate in fp32 to preserve precision (matches PyTorch behavior).
    auto output_dtype = (compute_dtype == DataType::kBFLOAT16) ? DataType::kFLOAT32 : compute_dtype;
    const std::vector<int64_t> weight_dims
        = transpose ? std::vector<int64_t>{out_features, in_features} : std::vector<int64_t>{in_features, out_features};
    // No Fill(0) needed: cuBLAS beta=0.0f fully overwrites output.
    auto grad_weight = std::make_shared<Tensor>(weight_dims, output_dtype, grad_output->GetDevice());

    auto device = grad_output->GetDevice();
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasHandle_t handle = dynamic_cast<infini_train::core::cuda::CudaBlasHandle *>(
                                infini_train::core::GetDeviceGuardImpl(device.type())->GetBlasHandle(device))
                                ->cublas_handle();

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
    int m = transpose ? in_features : out_features;
    int n = transpose ? out_features : in_features;
    auto ldc = transpose ? in_features : out_features;

    switch (compute_dtype) {
        DISPATCH_CASE(WRAP({
                          const void *a = transpose ? input->DataPtr() : grad_output_promoted->DataPtr();
                          const void *b = transpose ? grad_output_promoted->DataPtr() : input->DataPtr();
                          auto lda = transpose ? in_features : out_features;
                          auto ldb = transpose ? out_features : in_features;
                          CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, bs, &alpha,
                                                   static_cast<const float *>(a), lda, static_cast<const float *>(b),
                                                   ldb, &beta, static_cast<float *>(grad_weight->DataPtr()), ldc));
                      }),
                      DataType::kFLOAT32)
        DISPATCH_CASE(WRAP({
                          const void *a = transpose ? input->DataPtr() : grad_output_promoted->DataPtr();
                          const void *b = transpose ? grad_output_promoted->DataPtr() : input->DataPtr();
                          auto lda = transpose ? in_features : out_features;
                          auto ldb = transpose ? out_features : in_features;
                          CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, bs, &alpha, a, CUDA_R_16BF,
                                                    lda, b, CUDA_R_16BF, ldb, &beta, grad_weight->DataPtr(), CUDA_R_32F,
                                                    ldc, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
                      }),
                      DataType::kBFLOAT16)
    }

    return grad_weight;
}

std::shared_ptr<Tensor> LinearBackwardBias(const std::shared_ptr<Tensor> &grad_output, int64_t out_features) {
    const auto &dims = grad_output->Dims();
    CHECK_GE(dims.size(), 2);
    const int64_t bs = std::accumulate(dims.rbegin() + 1, dims.rend(), 1, std::multiplies<int64_t>{});

    auto compute_dtype = grad_output->Dtype();
    // For bf16 compute, accumulate in fp32 to preserve precision (matches PyTorch behavior).
    auto output_dtype = (compute_dtype == DataType::kBFLOAT16) ? DataType::kFLOAT32 : compute_dtype;
    auto grad_bias
        = std::make_shared<Tensor>(std::vector<int64_t>{out_features}, output_dtype, grad_output->GetDevice());

    auto device = grad_output->GetDevice();
    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();

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

REGISTER_CUDA_LINEAR_KERNEL(MatmulForward)
REGISTER_CUDA_LINEAR_KERNEL(MatmulBackward)
REGISTER_CUDA_LINEAR_KERNEL(LinearForward)
REGISTER_CUDA_LINEAR_KERNEL(LinearBackwardInput)
REGISTER_CUDA_LINEAR_KERNEL(LinearBackwardWeight)
REGISTER_CUDA_LINEAR_KERNEL(LinearBackwardBias)

#undef REGISTER_CUDA_LINEAR_KERNEL
