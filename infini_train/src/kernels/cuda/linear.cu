#include "infini_train/include/device.h"
#include "infini_train/include/kernels/cuda/linear.h"

#include <cstddef>
#include <memory>
#include <tuple>

#include "cublas_v2.h"
#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {

#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t status = call;                                                                                     \
        if (status != cudaSuccess) {                                                                                   \
            LOG(FATAL) << "CUDA Error: " << cudaGetErrorString(status) << " at " << __FILE__ << ":" << __LINE__;       \
        }                                                                                                              \
    } while (0)

#define CUBLAS_CHECK(call)                                                                                             \
    do {                                                                                                               \
        cublasStatus_t status = call;                                                                                  \
        if (status != CUBLAS_STATUS_SUCCESS) {                                                                         \
            LOG(FATAL) << "CUBLAS Error: " << cublasGetStatusString(status) << " at " << __FILE__ << ":" << __LINE__;  \
        }                                                                                                              \
    } while (0)

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
    cublasCreate(&handle);

    // cuBLAS is colmun-major
    // output = input * other --> output.T = other.T * input.T
    // C = A * B ==> output.T[*, n, m] = other.T[*, n, k] * input.T[*, k, m]
    // C = output.T[*, n, m]
    // A = other.T[*, n, k]
    // B = input.T[*, k, m]
    int lda = n;
    int ldb = k;
    int ldc = n;
    int64_t strideA = n * k;
    int64_t strideB = k * m;
    int64_t strideC = m * n;
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, other->DataPtr(), CUDA_R_32F, lda,
                               strideA, input->DataPtr(), CUDA_R_32F, ldb, strideB, &beta, output->DataPtr(),
                               CUDA_R_32F, ldc, strideC, bs, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);

    cublasDestroy(handle);
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

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);
    auto grad_other = std::make_shared<Tensor>(other_dims, DataType::kFLOAT32);
    grad_input->Fill<float>(0.0f);
    grad_other->Fill<float>(0.0f);

    float alpha = 1.0f, beta = 0.0f;
    cublasHandle_t handle;
    cublasCreate(&handle);

    {
        // cuBLAS is colmun-major
        // grad_input = grad_output * other.T --> grad_input.T = other * grad_output.T
        // C = A.T * B ==> grad_input.T[*, k, m] = other[*, k, n] * grad_output.T[*, n, m]
        // C = grad_input.T[*, k, m]
        // A = other.T[*, n, k]
        // B = grad_output.T[*, n, m]
        const int lda = n, ldb = n, ldc = k;
        const int64_t strideA = k * n;
        const int64_t strideB = n * m;
        const int64_t strideC = m * k;
        cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, k, m, n, &alpha, other->DataPtr(), CUDA_R_32F, lda,
                                   strideA, grad_output->DataPtr(), CUDA_R_32F, ldb, strideB, &beta,
                                   grad_input->DataPtr(), CUDA_R_32F, ldc, strideC, bs, CUDA_R_32F,
                                   CUBLAS_GEMM_DEFAULT);
    }

    {
        // cuBLAS is colmun-major
        // grad_other = input.T * grad_output --> grad_other.T =  grad_output.T * input
        // C = A * B.T ==> grad_other.T[*, n, k] = grad_output.T[*, n, m] * input[*, m, k]
        // C = grad_other.T[*, k, m]
        // A = grad_output.T[*, n, k]
        // B = input.T[*, n, m]
        const int lda = n, ldb = n, ldc = k;
        const int64_t strideA = k * n;
        const int64_t strideB = n * m;
        const int64_t strideC = m * k;
        cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, k, m, &alpha, grad_output->DataPtr(),
                                   CUDA_R_32F, lda, strideA, input->DataPtr(), CUDA_R_32F, ldb, strideB, &beta,
                                   grad_other->DataPtr(), CUDA_R_32F, ldc, strideC, bs, CUDA_R_32F,
                                   CUBLAS_GEMM_DEFAULT);
    }

    cublasDestroy(handle);
    return {grad_input, grad_other};
}

std::shared_ptr<Tensor> LinearForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                                      bool transpose, const std::shared_ptr<Tensor> &bias) {

    /*
        !transpose: output = input * weight + bias
        output[*, out_features] = input[*, in_features] * weight[in_features, out_features] + bias[out_features]

        transpose:  output = input * weight^T + bias
        output[*, out_features] = input[*, in_features] * weight[out_features, in_features]^T + bias[out_features]
    */

    CHECK_EQ(input->Dims().size(), 2);
    const int64_t bs = input->Dims()[0];
    const int64_t in_features = input->Dims()[1];
    CHECK_EQ(weight->Dims().size(), 2);

    // As for cublas:
    // C = alpha * op(B) * op(A) + beta * C
    // Dimensions:
    //   input:  (bs, in_features)
    //   weight: (in_features, out_features) or (out_features, in_features) if transposed
    //   output: (bs, out_features)
    int64_t out_features = 0;
    cublasOperation_t op_weight = CUBLAS_OP_N;

    if (transpose) {
        // weight: (out_features, in_features)
        CHECK_EQ(in_features, weight->Dims()[1]);
        out_features = weight->Dims()[0];
        op_weight = CUBLAS_OP_T;
    } else {
        // weight: (in_features, out_features)
        CHECK_EQ(in_features, weight->Dims()[0]);
        out_features = weight->Dims()[1];
        op_weight = CUBLAS_OP_N;
    }

    auto output = std::make_shared<Tensor>(std::vector<int64_t>{bs, out_features}, DataType::kFLOAT32,
                                           Device(DeviceType::kCUDA, 0));

    if (bias) {
        CHECK_EQ(bias->Dims().size(), 1);
        CHECK_EQ(bias->Dims()[0], out_features);
        for (int i = 0; i < bs; ++i) {
            cudaMemcpy(static_cast<float *>(output->DataPtr()) + i * out_features, bias->DataPtr(),
                       out_features * sizeof(float), cudaMemcpyDeviceToDevice);
        }
    } else {
        output->Fill<float>(0.0f);
    }

    const float alpha = 1.0f;
    const float beta = 1.0f;
    cublasHandle_t handle;
    cublasCreate(&handle);

    // C = alpha * op(B) * op(A) + beta * C
    // output = alpha * (input * weight) + beta * output
    cublasSgemm(handle, op_weight, CUBLAS_OP_N, out_features, bs, in_features, &alpha,
                static_cast<const float *>(weight->DataPtr()), (op_weight == CUBLAS_OP_N) ? out_features : in_features,
                static_cast<const float *>(input->DataPtr()), in_features, &beta,
                static_cast<float *>(output->DataPtr()), out_features);

    cublasDestroy(handle);

    return {output};
}

__global__ void set_ones(float *data, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        data[idx] = 1.0f;
    }
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
LinearBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight, bool transpose,
               int64_t out_features, const std::shared_ptr<Tensor> &grad_output, const bool bias) {
    CHECK_EQ(input->Dims().size(), 2);
    const int bs = input->Dims()[0];
    const int in_features = input->Dims()[1];
    CHECK_EQ(weight->Dims().size(), 2);

    auto grad_input = std::make_shared<Tensor>(input->Dims(), DataType::kFLOAT32, Device(DeviceType::kCUDA, 0));
    auto grad_weight = std::make_shared<Tensor>(weight->Dims(), DataType::kFLOAT32, Device(DeviceType::kCUDA, 0));
    grad_weight->Fill<float>(0.0f);
    std::shared_ptr<Tensor> grad_bias = nullptr;
    if (bias) {
        grad_bias = std::make_shared<Tensor>(std::vector<int64_t>{out_features}, DataType::kFLOAT32,
                                             Device(DeviceType::kCUDA, 0));
        grad_bias->Fill<float>(0.0f);
    }

    float alpha = 1.0f;
    float beta = 0.0f;
    cublasHandle_t handle;
    cublasCreate(&handle);

    if (transpose) {
        // d_input = d_output * weight --> d_input.T = weight * d_output.T
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, in_features, bs, out_features, &alpha,
                                 static_cast<const float *>(weight->DataPtr()), in_features,
                                 static_cast<const float *>(grad_output->DataPtr()), out_features, &beta,
                                 static_cast<float *>(grad_input->DataPtr()), in_features));

        // d_weight = d_output.T * input --> d_weight.T = input.T * d_output
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, out_features, in_features, bs, &alpha,
                                 static_cast<const float *>(grad_output->DataPtr()), out_features,
                                 static_cast<const float *>(input->DataPtr()), in_features, &beta,
                                 static_cast<float *>(grad_weight->DataPtr()), out_features));
    } else {
        // d_input = d_output * weight.T --> d_input.T = weight * d_output.T
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, in_features, bs, out_features, &alpha,
                                 static_cast<const float *>(weight->DataPtr()), out_features,
                                 static_cast<const float *>(grad_output->DataPtr()), out_features, &beta,
                                 static_cast<float *>(grad_input->DataPtr()), in_features));

        // d_weight = input.T * d_output --> d_weight.T = d_output.T * input
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, out_features, in_features, bs, &alpha,
                                 static_cast<const float *>(grad_output->DataPtr()), out_features,
                                 static_cast<const float *>(input->DataPtr()), in_features, &beta,
                                 static_cast<float *>(grad_weight->DataPtr()), out_features));
    }

    // FIXME(dcj): remove this sync
    // CUDA_CHECK(cudaDeviceSynchronize());

    // d_bias = \sum_i(i=0, bs-1) d_output[i]
    // TODO(dcj): use thrust::fill or reduce kernel do this
    if (bias) {
        auto ones
            = std::make_shared<Tensor>(std::vector<int64_t>{bs}, DataType::kFLOAT32, Device(DeviceType::kCUDA, 0));
        float *ones_ptr = static_cast<float *>(ones->DataPtr());

        int threads_per_block = 256;
        int num_blocks = (bs + threads_per_block - 1) / threads_per_block;

        set_ones<<<num_blocks, threads_per_block>>>(ones_ptr, bs);

        CUBLAS_CHECK(cublasSgemv(
            handle, CUBLAS_OP_N, out_features, bs, &alpha, static_cast<const float *>(grad_output->DataPtr()),
            out_features, static_cast<float *>(ones_ptr), 1, &beta, static_cast<float *>(grad_bias->DataPtr()), 1));
    }

    cublasDestroy(handle);

    return {grad_input, grad_weight, grad_bias};
}
} // namespace infini_train::kernels::cuda
