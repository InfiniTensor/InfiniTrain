#include "cublas_v2.h"
#include "glog/logging.h"
#include <cuda_bf16.h>

#include "infini_train/include/common/cuda/common_cuda.cuh"

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

    std::vector<int64_t> output_dims = input_dims;
    output_dims[output_dims.size() - 1] = n;
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32, input->GetDevice());

    const float alpha = 1.0f, beta = 0.0f;
    cublasHandle_t handle;
    // TODO(zbl): create handle only once
    CUBLAS_CHECK(cublasCreate(&handle));

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

    switch (input->Dtype()) {
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
        LOG(FATAL) << "CUDA Matmul forward: 'Unsupported data type' at " << __FILE__ << ":" << __LINE__;
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    return output;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
MatmulBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other,
               const std::shared_ptr<Tensor> &grad_output) {
    /*
       grad_input[*, m, k] = grad_output[*, m, n] * other[*, k, n]^T
       grad_other[*, k, n] = input[*, m, k]^T * grad_output[*, m, n]
    */
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

    auto dtype = input->Dtype();
    auto grad_input = std::make_shared<Tensor>(input_dims, dtype, grad_output->GetDevice());
    auto grad_other = std::make_shared<Tensor>(other_dims, dtype, grad_output->GetDevice());
    switch (dtype) {
        DISPATCH_CASE(WRAP({
                          grad_input->Fill<float>(0.0f);
                          grad_other->Fill<float>(0.0f);
                      }),
                      DataType::kFLOAT32)
        DISPATCH_CASE(WRAP({
                          grad_input->Fill<nv_bfloat16>(0);
                          grad_other->Fill<nv_bfloat16>(0);
                      }),
                      DataType::kBFLOAT16)
    default:
        LOG(FATAL) << "CUDA Matmul backward: 'Unsupported data type' at " << __FILE__ << ":" << __LINE__;
    }

    const float alpha = 1.0f, beta = 0.0f;
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

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
        switch (dtype) {
            DISPATCH_CASE(WRAP(CUBLAS_CHECK(cublasGemmStridedBatchedEx(
                              handle, CUBLAS_OP_T, CUBLAS_OP_N, k, m, n, &alpha, other->DataPtr(), CUDA_R_32F, lda,
                              stride_a, grad_output->DataPtr(), CUDA_R_32F, ldb, stride_b, &beta, grad_input->DataPtr(),
                              CUDA_R_32F, ldc, stride_c, bs, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));),
                          DataType::kFLOAT32)
            DISPATCH_CASE(
                WRAP(CUBLAS_CHECK(cublasGemmStridedBatchedEx(
                    handle, CUBLAS_OP_T, CUBLAS_OP_N, k, m, n, &alpha, other->DataPtr(), CUDA_R_16BF, lda, stride_a,
                    grad_output->DataPtr(), CUDA_R_16BF, ldb, stride_b, &beta, grad_input->DataPtr(), CUDA_R_16BF, ldc,
                    stride_c, bs, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));),
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
        switch (dtype) {
            DISPATCH_CASE(WRAP(CUBLAS_CHECK(cublasGemmStridedBatchedEx(
                              handle, CUBLAS_OP_N, CUBLAS_OP_T, n, k, m, &alpha, grad_output->DataPtr(), CUDA_R_32F,
                              lda, stride_a, input->DataPtr(), CUDA_R_32F, ldb, stride_b, &beta, grad_other->DataPtr(),
                              CUDA_R_32F, ldc, stride_c, bs, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));),
                          DataType::kFLOAT32)
            DISPATCH_CASE(WRAP(CUBLAS_CHECK(cublasGemmStridedBatchedEx(
                              handle, CUBLAS_OP_N, CUBLAS_OP_T, n, k, m, &alpha, grad_output->DataPtr(), CUDA_R_16BF,
                              lda, stride_a, input->DataPtr(), CUDA_R_16BF, ldb, stride_b, &beta, grad_other->DataPtr(),
                              CUDA_R_16BF, ldc, stride_c, bs, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));),
                          DataType::kBFLOAT16)
        }
    }

    CUBLAS_CHECK(cublasDestroy(handle));
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

    auto output_dims = input_dims;
    *output_dims.rbegin() = out_features;
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32, input->GetDevice());

    if (bias) {
        CHECK_EQ(bias->Dims().size(), 1);
        CHECK_EQ(bias->Dims()[0], out_features);
        int threads_per_block = 256;
        int num_blocks = (bs * out_features + threads_per_block - 1) / threads_per_block;

        switch (input->Dtype()) {
            DISPATCH_CASE(WRAP(BiasCopyKernel<<<num_blocks, threads_per_block>>>(
                                   static_cast<float *>(output->DataPtr()), static_cast<const float *>(bias->DataPtr()),
                                   bs, out_features);),
                          DataType::kFLOAT32)
            DISPATCH_CASE(WRAP(BiasCopyKernel<<<num_blocks, threads_per_block>>>(
                                   static_cast<nv_bfloat16 *>(output->DataPtr()),
                                   static_cast<const nv_bfloat16 *>(bias->DataPtr()), bs, out_features);),
                          DataType::kBFLOAT16)
        default:
            LOG(FATAL) << "CUDA Linear forward: 'Unsupported data type' at " << __FILE__ << ":" << __LINE__;
        }

    } else {
        switch (input->Dtype()) {
            DISPATCH_CASE(WRAP(output->Fill<float>(0.0f);), DataType::kFLOAT32)
            DISPATCH_CASE(WRAP(output->Fill<nv_bfloat16>(0);), DataType::kBFLOAT16)
        default:
            LOG(FATAL) << "CUDA Linear forward: 'Unsupported data type' at " << __FILE__ << ":" << __LINE__;
        }
    }

    const float alpha = 1.0f;
    const float beta = 1.0f;
    auto trans_a = transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
    auto trans_b = CUBLAS_OP_N;
    auto lda = transpose ? in_features : out_features;
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // TODO(zbl): use cublasSgemv if possible
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
    CUBLAS_CHECK(cublasDestroy(handle));
    return output;
}

template <int BLOCK_SIZE, typename T>
__global__ void ReduceColumnsKernel(const T *__restrict__ input, T *__restrict__ output, int num_rows, int num_cols) {
    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int row = blockIdx.x;
    float sum = 0.0f;

    for (int col = threadIdx.x; col < num_cols; col += blockDim.x) { sum += input[row * num_cols + col]; }

    float reduced = BlockReduce(temp_storage).Sum(sum);

    if (threadIdx.x == 0) {
        output[row] = reduced;
    }
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
LinearBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight, bool transpose,
               int64_t out_features, const std::shared_ptr<Tensor> &grad_output, const bool bias) {
    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);
    CHECK_EQ(out_features, weight_dims[transpose ? 0 : 1]);

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32, grad_output->GetDevice());
    auto grad_weight = std::make_shared<Tensor>(weight_dims, DataType::kFLOAT32, grad_output->GetDevice());
    std::shared_ptr<Tensor> grad_bias = nullptr;

    auto initialize_gradients = [&](auto zero_value, DataType dtype) {
        using T = decltype(zero_value);
        grad_input->Fill<T>(zero_value);
        grad_weight->Fill<T>(zero_value);
        if (bias) {
            grad_bias = std::make_shared<Tensor>(std::vector<int64_t>{out_features}, dtype, grad_output->GetDevice());
            grad_bias->Fill<T>(zero_value);
        }
    };

    switch (input->Dtype()) {
        DISPATCH_CASE(WRAP(initialize_gradients(0.0f, DataType::kFLOAT32);), DataType::kFLOAT32)
        DISPATCH_CASE(WRAP(initialize_gradients(nv_bfloat16(0), DataType::kBFLOAT16);), DataType::kBFLOAT16)
    default:
        LOG(FATAL) << "CUDA Linear backward: 'Unsupported data type' at " << __FILE__ << ":" << __LINE__;
    }

    float alpha = 1.0f;
    float beta = 0.0f;
    auto trans_a1 = transpose ? CUBLAS_OP_N : CUBLAS_OP_T;
    auto trans_b1 = CUBLAS_OP_N;
    auto lda1 = transpose ? in_features : out_features;
    auto trans_a2 = CUBLAS_OP_N;
    auto trans_b2 = CUBLAS_OP_T;
    int m2 = transpose ? in_features : out_features;
    int n2 = transpose ? out_features : in_features;
    const void *a2 = transpose ? input->DataPtr() : grad_output->DataPtr();
    const void *b2 = transpose ? grad_output->DataPtr() : input->DataPtr();
    auto lda2 = transpose ? in_features : out_features;
    auto ldb2 = transpose ? out_features : in_features;
    auto ldc2 = transpose ? in_features : out_features;

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    switch (input->Dtype()) {
        // TODO(zbl): use cublasSgemv if possible
        DISPATCH_CASE(WRAP({
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
                          CUBLAS_CHECK(cublasSgemm(handle, trans_a1, trans_b1, in_features, bs, out_features, &alpha,
                                                   static_cast<const float *>(weight->DataPtr()), lda1,
                                                   static_cast<const float *>(grad_output->DataPtr()), out_features,
                                                   &beta, static_cast<float *>(grad_input->DataPtr()), in_features));
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
                          CUBLAS_CHECK(cublasSgemm(handle, trans_a2, trans_b2, m2, n2, bs, &alpha,
                                                   static_cast<const float *>(a2), lda2, static_cast<const float *>(b2),
                                                   ldb2, &beta, static_cast<float *>(grad_weight->DataPtr()), ldc2));
                          // d_bias = \sum_i(i=0, bs-1) d_output[i]
                          // TODO(dcj): use thrust::fill or reduce kernel do this
                          if (bias) {
                              constexpr int BLOCK_SIZE = 256;
                              int threads_per_block = BLOCK_SIZE;
                              int num_blocks = out_features;
                              ReduceColumnsKernel<BLOCK_SIZE><<<num_blocks, threads_per_block>>>(
                                  static_cast<const float *>(grad_output->DataPtr()),
                                  static_cast<float *>(grad_bias->DataPtr()), out_features, bs);
                          }
                      }),
                      DataType::kFLOAT32)
        DISPATCH_CASE(WRAP({
                          CUBLAS_CHECK(cublasGemmEx(handle, trans_a1, trans_b1, in_features, bs, out_features, &alpha,
                                                    weight->DataPtr(), CUDA_R_16BF, lda1, grad_output->DataPtr(),
                                                    CUDA_R_16BF, out_features, &beta, grad_input->DataPtr(),
                                                    CUDA_R_16BF, in_features, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
                          CUBLAS_CHECK(cublasGemmEx(handle, trans_a2, trans_b2, m2, n2, bs, &alpha, a2, CUDA_R_16BF,
                                                    lda2, b2, CUDA_R_16BF, ldb2, &beta, grad_weight->DataPtr(),
                                                    CUDA_R_16BF, ldc2, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
                          if (bias) {
                              constexpr int BLOCK_SIZE = 256;
                              int threads_per_block = BLOCK_SIZE;
                              int num_blocks = out_features;
                              ReduceColumnsKernel<BLOCK_SIZE><<<num_blocks, threads_per_block>>>(
                                  static_cast<const nv_bfloat16 *>(grad_output->DataPtr()),
                                  static_cast<nv_bfloat16 *>(grad_bias->DataPtr()), out_features, bs);
                          }
                      }),
                      DataType::kBFLOAT16)
    }

    CUBLAS_CHECK(cublasDestroy(handle));

    return {grad_input, grad_weight, grad_bias};
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_LINEAR_KERNEL(kernel_name)                                                                       \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_LINEAR_KERNEL(MatmulForward)
REGISTER_CUDA_LINEAR_KERNEL(MatmulBackward)
REGISTER_CUDA_LINEAR_KERNEL(LinearForward)
REGISTER_CUDA_LINEAR_KERNEL(LinearBackward)

#undef REGISTER_CUDA_LINEAR_KERNEL
