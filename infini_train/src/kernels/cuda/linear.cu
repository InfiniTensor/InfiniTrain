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

// template <typename T> __global__ void BiasCopyKernel(T *output, const T *bias, int bs, int out_features) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= bs * out_features) {
//         return;
//     }
//     int j = idx % out_features;
//     output[idx] = bias[j];
// }

// 向量化写入优化
template <typename T> 
__global__ void BiasCopyKernel(T *output, const T *bias, int bs, int out_features) {
    constexpr int VEC_SIZE = 16 / sizeof(T);
    size_t idx = (size_t)(blockIdx.x * blockDim.x + threadIdx.x) * VEC_SIZE;
    size_t total_elements = (size_t)bs * out_features;

    if (idx + VEC_SIZE <= total_elements) {
        T local_vals[VEC_SIZE];
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            local_vals[i] = bias[(idx + i) % out_features];
        }
        *reinterpret_cast<int4*>(output + idx) = *reinterpret_cast<int4*>(local_vals);
    } else {
        for (size_t i = idx; i < total_elements; ++i) {
            output[i] = bias[i % out_features];
        }
    }
}

// template <int BLOCK_SIZE, typename T>
// __global__ void ReduceColumnsKernel(const T *__restrict__ input, T *__restrict__ output, int num_rows, int num_cols) {
//     using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
//     __shared__ typename BlockReduce::TempStorage temp_storage;
//     int row = blockIdx.x;
//     float sum = 0.0f;
//     for (int col = threadIdx.x; col < num_cols; col += blockDim.x) {
//         sum += common::cuda::Cast<float>(input[row * num_cols + col]);
//     }
//     float reduced = BlockReduce(temp_storage).Sum(sum);
//     if (threadIdx.x == 0) {
//         output[row] = reduced;
//     }
// }

// 向量化加载与规约优化
template <int BLOCK_SIZE, typename T>
__global__ void ReduceColumnsKernel(const T *__restrict__ input, T *__restrict__ output, int num_rows, int num_cols) {
    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int row = blockIdx.x; 
    float sum = 0.0f;
    constexpr int VEC_SIZE = 16 / sizeof(T);
    int col = threadIdx.x * VEC_SIZE;
    
    for (; col + VEC_SIZE <= num_cols; col += blockDim.x * VEC_SIZE) {
        T local_vals[VEC_SIZE];
        *reinterpret_cast<int4*>(local_vals) = *reinterpret_cast<const int4*>(input + (size_t)row * num_cols + col);
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            sum += common::cuda::Cast<float>(local_vals[i]);
        }
    }
    for (int c = col + (threadIdx.x % VEC_SIZE); c < num_cols; c += blockDim.x) {
        sum += common::cuda::Cast<float>(input[(size_t)row * num_cols + c]);
    }

    float reduced = BlockReduce(temp_storage).Sum(sum);
    if (threadIdx.x == 0) {
        output[row] = common::cuda::Cast<T>(reduced);
    }
}

std::shared_ptr<infini_train::Tensor> MatmulForward(const std::shared_ptr<infini_train::Tensor> &input, const std::shared_ptr<infini_train::Tensor> &other) {
    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();
    const int64_t m = input_dims[input_dims.size() - 2];
    const int64_t k = input_dims[input_dims.size() - 1];
    const int64_t n = other_dims[other_dims.size() - 1];
    const int64_t bs = std::accumulate(input_dims.rbegin() + 2, input_dims.rend(), 1, std::multiplies<int64_t>{});

    auto output = std::make_shared<infini_train::Tensor>(input_dims, input->Dtype(), input->GetDevice());
    auto device = input->GetDevice();
    const float alpha = 1.0f, beta = 0.0f;
    cublasHandle_t handle = dynamic_cast<infini_train::core::cuda::CudaBlasHandle *>(
                                infini_train::core::GetDeviceGuardImpl(device.type())->GetBlasHandle(device))
                                ->cublas_handle();

    // 开启TF32加速
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));

    int lda = n, ldb = k, ldc = n;
    int64_t stride_a = n * k, stride_b = k * m, stride_c = m * n;

    switch (input->Dtype()) {
        DISPATCH_CASE(WRAP(CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, other->DataPtr(), CUDA_R_32F, lda, stride_a, input->DataPtr(), CUDA_R_32F, ldb, stride_b, &beta, output->DataPtr(), CUDA_R_32F, ldc, stride_c, bs, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));), infini_train::DataType::kFLOAT32)
        DISPATCH_CASE(WRAP(CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, other->DataPtr(), CUDA_R_16BF, lda, stride_a, input->DataPtr(), CUDA_R_16BF, ldb, stride_b, &beta, output->DataPtr(), CUDA_R_16BF, ldc, stride_c, bs, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));), infini_train::DataType::kBFLOAT16)
    }
    return output;
}

std::tuple<std::shared_ptr<infini_train::Tensor>, std::shared_ptr<infini_train::Tensor>>
MatmulBackward(const std::shared_ptr<infini_train::Tensor> &input, const std::shared_ptr<infini_train::Tensor> &other,
               const std::shared_ptr<infini_train::Tensor> &grad_output) {
    auto input_dtype = input->Dtype();
    auto other_dtype = other->Dtype();
    auto grad_output_dtype = grad_output->Dtype();
    
    DataType promoted_type = DispatchFunc<DataTypeList<INFINI_ALL_TYPES>, DataTypeList<INFINI_ALL_TYPES>, DataTypeList<INFINI_ALL_TYPES>>(
        {input_dtype, other_dtype, grad_output_dtype},
        [=]<typename Tin, typename To, typename Tgrad>() { return DataTypeMap_v<WidestType_t<Tin, To, Tgrad>>; },
        "CUDA MatmulBackward");

    auto input_promoted = input_dtype == promoted_type ? input : std::make_shared<infini_train::Tensor>(input->To(promoted_type));
    auto other_promoted = other_dtype == promoted_type ? other : std::make_shared<infini_train::Tensor>(other->To(promoted_type));
    auto grad_output_promoted = grad_output_dtype == promoted_type ? grad_output : std::make_shared<infini_train::Tensor>(grad_output->To(promoted_type));

    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();
    const int64_t m = input_dims[input_dims.size() - 2];
    const int64_t k = input_dims[input_dims.size() - 1];
    const int64_t n = other_dims[other_dims.size() - 1];
    const int64_t bs = std::accumulate(input_dims.rbegin() + 2, input_dims.rend(), 1, std::multiplies<int64_t>{});

    auto grad_input = std::make_shared<infini_train::Tensor>(input_dims, promoted_type, grad_output->GetDevice());
    auto grad_other = std::make_shared<infini_train::Tensor>(other_dims, promoted_type, grad_output->GetDevice());

    DispatchFunc<DataType::kFLOAT32, DataType::kBFLOAT16>(
        promoted_type, [=]<typename T>() { grad_input->Fill<T>(0); grad_other->Fill<T>(0); }, "CUDA MatmulBackward");

    auto device = input_promoted->GetDevice();
    const float alpha = 1.0f, beta = 0.0f;
    cublasHandle_t handle = dynamic_cast<infini_train::core::cuda::CudaBlasHandle *>(
                                infini_train::core::GetDeviceGuardImpl(device.type())->GetBlasHandle(device))
                                ->cublas_handle();

    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));

    const int lda = n, ldb = n, ldc = k;
    const int64_t stride_a = k * n, stride_b = n * m, stride_c = m * k;
    
    switch (promoted_type) {
        DISPATCH_CASE(WRAP(CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, k, m, n, &alpha, other_promoted->DataPtr(), CUDA_R_32F, lda, stride_a, grad_output_promoted->DataPtr(), CUDA_R_32F, ldb, stride_b, &beta, grad_input->DataPtr(), CUDA_R_32F, ldc, stride_c, bs, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));), DataType::kFLOAT32)
        DISPATCH_CASE(WRAP(CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, k, m, n, &alpha, other_promoted->DataPtr(), CUDA_R_16BF, lda, stride_a, grad_output_promoted->DataPtr(), CUDA_R_16BF, ldb, stride_b, &beta, grad_input->DataPtr(), CUDA_R_16BF, ldc, stride_c, bs, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));), DataType::kBFLOAT16)
    }

    const int lda2 = n, ldb2 = k, ldc2 = n;
    const int64_t stride_a2 = n * m, stride_b2 = k * m, stride_c2 = n * k;
    switch (promoted_type) {
        DISPATCH_CASE(WRAP(CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, k, m, &alpha, grad_output_promoted->DataPtr(), CUDA_R_32F, lda2, stride_a2, input_promoted->DataPtr(), CUDA_R_32F, ldb2, stride_b2, &beta, grad_other->DataPtr(), CUDA_R_32F, ldc2, stride_c2, bs, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));), DataType::kFLOAT32)
        DISPATCH_CASE(WRAP(CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, k, m, &alpha, grad_output_promoted->DataPtr(), CUDA_R_16BF, lda2, stride_a2, input_promoted->DataPtr(), CUDA_R_16BF, ldb2, stride_b2, &beta, grad_other->DataPtr(), CUDA_R_16BF, ldc2, stride_c2, bs, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));), DataType::kBFLOAT16)
    }

    return {grad_input, grad_other};
}

std::shared_ptr<infini_train::Tensor> LinearForward(const std::shared_ptr<infini_train::Tensor> &input, const std::shared_ptr<infini_train::Tensor> &weight,
                                      bool transpose, const std::shared_ptr<infini_train::Tensor> &bias) {
    const auto &input_dims = input->Dims();
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();
    const int64_t out_features = weight->Dims()[transpose ? 0 : 1];

    auto output = std::make_shared<infini_train::Tensor>(input_dims, input->Dtype(), input->GetDevice());
    auto device = input->GetDevice();
    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();

    if (bias) {
        int threads_per_block = 256;
        int num_blocks = (bs * out_features + threads_per_block - 1) / threads_per_block;
        DispatchFunc<INFINI_ALL_FLOATING_TYPES>(input->Dtype(), [=]<typename T>() { BiasCopyKernel<<<num_blocks, threads_per_block, 0, cuda_stream>>>(static_cast<T *>(output->DataPtr()), static_cast<const T *>(bias->DataPtr()), bs, out_features); }, "CUDA LinearForward");
    }

    cublasHandle_t handle = dynamic_cast<infini_train::core::cuda::CudaBlasHandle *>(
                                infini_train::core::GetDeviceGuardImpl(device.type())->GetBlasHandle(device))
                                ->cublas_handle();

    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));

    const float alpha = 1.0f, beta = bias ? 1.0f : 0.0f;
    auto trans_a = transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
    int lda = transpose ? in_features : out_features;

    switch (input->Dtype()) {
        DISPATCH_CASE(WRAP(CUBLAS_CHECK(cublasSgemm(handle, trans_a, CUBLAS_OP_N, out_features, bs, in_features, &alpha, static_cast<const float *>(weight->DataPtr()), lda, static_cast<const float *>(input->DataPtr()), in_features, &beta, static_cast<float *>(output->DataPtr()), out_features));), DataType::kFLOAT32)
        DISPATCH_CASE(WRAP(CUBLAS_CHECK(cublasGemmEx(handle, trans_a, CUBLAS_OP_N, out_features, bs, in_features, &alpha, weight->DataPtr(), CUDA_R_16BF, lda, input->DataPtr(), CUDA_R_16BF, in_features, &beta, output->DataPtr(), CUDA_R_16BF, out_features, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));), DataType::kBFLOAT16)
    }
    return output;
}

std::tuple<std::shared_ptr<infini_train::Tensor>, std::shared_ptr<infini_train::Tensor>, std::shared_ptr<infini_train::Tensor>>
LinearBackward(const std::shared_ptr<infini_train::Tensor> &input, const std::shared_ptr<infini_train::Tensor> &weight, bool transpose,
               int64_t out_features, const std::shared_ptr<infini_train::Tensor> &grad_output, const bool bias) {
    const auto &input_dims = input->Dims();
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();
    DataType promoted_type = DispatchFunc<DataTypeList<INFINI_ALL_TYPES>, DataTypeList<INFINI_ALL_TYPES>, DataTypeList<INFINI_ALL_TYPES>>(
        {input->Dtype(), weight->Dtype(), grad_output->Dtype()},
        [=]<typename Tin, typename Tw, typename Tgrad>() { return DataTypeMap_v<WidestType_t<Tin, Tw, Tgrad>>; },
        "CUDA LinearBackward");

    auto grad_input = std::make_shared<infini_train::Tensor>(input_dims, promoted_type, grad_output->GetDevice());
    auto grad_weight = std::make_shared<infini_train::Tensor>(weight->Dims(), promoted_type, grad_output->GetDevice());
    std::shared_ptr<infini_train::Tensor> grad_bias = nullptr;

    DispatchFunc<DataType::kFLOAT32, DataType::kBFLOAT16>(promoted_type, [=, &grad_bias]<typename T>() { 
        grad_input->Fill<T>(0); grad_weight->Fill<T>(0); 
        if (bias) { grad_bias = std::make_shared<infini_train::Tensor>(std::vector<int64_t>{out_features}, promoted_type, grad_output->GetDevice()); grad_bias->Fill<T>(0); }
    }, "CUDA LinearBackward");

    auto device = input->GetDevice();
    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();
    cublasHandle_t handle = dynamic_cast<infini_train::core::cuda::CudaBlasHandle *>(
                                infini_train::core::GetDeviceGuardImpl(device.type())->GetBlasHandle(device))
                                ->cublas_handle();

    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
    const float alpha = 1.0f, beta = 0.0f;

    auto trans_a1 = transpose ? CUBLAS_OP_N : CUBLAS_OP_T;
    auto lda1 = transpose ? in_features : out_features;
    switch (promoted_type) {
        DISPATCH_CASE(WRAP(CUBLAS_CHECK(cublasSgemm(handle, trans_a1, CUBLAS_OP_N, in_features, bs, out_features, &alpha, static_cast<const float *>(weight->DataPtr()), lda1, static_cast<const float *>(grad_output->DataPtr()), out_features, &beta, static_cast<float *>(grad_input->DataPtr()), in_features));), DataType::kFLOAT32)
        DISPATCH_CASE(WRAP(CUBLAS_CHECK(cublasGemmEx(handle, trans_a1, CUBLAS_OP_N, in_features, bs, out_features, &alpha, weight->DataPtr(), CUDA_R_16BF, lda1, grad_output->DataPtr(), CUDA_R_16BF, out_features, &beta, grad_input->DataPtr(), CUDA_R_16BF, in_features, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));), DataType::kBFLOAT16)
    }

    int m2 = transpose ? in_features : out_features;
    int n2 = transpose ? out_features : in_features;
    auto trans_a2 = CUBLAS_OP_N, trans_b2 = CUBLAS_OP_T;
    const void *a2 = transpose ? input->DataPtr() : grad_output->DataPtr();
    const void *b2 = transpose ? grad_output->DataPtr() : input->DataPtr();
    int lda2 = transpose ? in_features : out_features, ldb2 = transpose ? out_features : in_features, ldc2 = transpose ? in_features : out_features;

    switch (promoted_type) {
        DISPATCH_CASE(WRAP(CUBLAS_CHECK(cublasSgemm(handle, trans_a2, trans_b2, m2, n2, bs, &alpha, static_cast<const float *>(a2), lda2, static_cast<const float *>(b2), ldb2, &beta, static_cast<float *>(grad_weight->DataPtr()), ldc2));), DataType::kFLOAT32)
        DISPATCH_CASE(WRAP(CUBLAS_CHECK(cublasGemmEx(handle, trans_a2, trans_b2, m2, n2, bs, &alpha, a2, CUDA_R_16BF, lda2, b2, CUDA_R_16BF, ldb2, &beta, grad_weight->DataPtr(), CUDA_R_16BF, ldc2, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));), DataType::kBFLOAT16)
    }

    if (bias) {
        constexpr int BLOCK_SIZE = 256;
        int num_blocks = out_features;
        DispatchFunc<DataType::kFLOAT32, DataType::kBFLOAT16>(promoted_type, [=]<typename T>() { ReduceColumnsKernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE, 0, cuda_stream>>>(static_cast<const T *>(grad_output->DataPtr()), static_cast<T *>(grad_bias->DataPtr()), out_features, bs); }, "CUDA LinearBackward");
    }

    return {grad_input, grad_weight, grad_bias};
}

} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_LINEAR_KERNEL(kernel_name) \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_LINEAR_KERNEL(MatmulForward)
REGISTER_CUDA_LINEAR_KERNEL(MatmulBackward)
REGISTER_CUDA_LINEAR_KERNEL(LinearForward)
REGISTER_CUDA_LINEAR_KERNEL(LinearBackward)

#undef REGISTER_CUDA_LINEAR_KERNEL