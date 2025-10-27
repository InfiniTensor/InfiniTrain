#include <memory>
#include <tuple>
#include <vector>

#include <cublas_v2.h>

#include "glog/logging.h"

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {
std::shared_ptr<Tensor> OuterForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
    /*
    Computes outer product: output[i, j] = input[i] * other[j]
    Equivalent to: input: [M, 1], other: [1, N] → output: [M, N]
    */

    const auto &in_dims = input->Dims();
    const auto &ot_dims = other->Dims();
    // TODO(zbl): support batched outer?
    CHECK_EQ(in_dims.size(), 1);
    CHECK_EQ(ot_dims.size(), 1);

    const int64_t M = in_dims[0];
    const int64_t N = ot_dims[0];

    auto output = std::make_shared<Tensor>(std::vector<int64_t>{M, N}, input->Dtype(), input->GetDevice());

    const auto *cuda_device = dynamic_cast<const CudaDevice *>(input->GetDevice());
    // reinterpret input: [M] as column vector [M, 1]
    // reinterpret other: [N] as row vector [1, N]
    // output[M, N] = input[M, 1] * other.T[1, N]
    // output.T[N, M] = other[N, 1] * input.T[1, M]
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasHandle_t handle = cuda_device->CublasHandle();

    switch (input->Dtype()) {
        DISPATCH_CASE(WRAP({
                          CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, 1, &alpha,
                                                   static_cast<const float *>(other->DataPtr()), N,
                                                   static_cast<const float *>(input->DataPtr()), 1, &beta,
                                                   static_cast<float *>(output->DataPtr()), N));
                      }),
                      DataType::kFLOAT32)
        DISPATCH_CASE(WRAP({
                          CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, 1, &alpha, other->DataPtr(),
                                                    CUDA_R_16BF, N, input->DataPtr(), CUDA_R_16BF, 1, &beta,
                                                    output->DataPtr(), CUDA_R_16BF, N, CUDA_R_32F,
                                                    CUBLAS_GEMM_DEFAULT));
                      }),
                      DataType::kBFLOAT16)
    }

    return output;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> OuterBackward(const std::shared_ptr<Tensor> &input,
                                                                           const std::shared_ptr<Tensor> &other,
                                                                           const std::shared_ptr<Tensor> &grad_output) {
    /*
    grad_input: [M] = grad_output: [M, N] × other: [N]
    grad_other: [N] = grad_output.T: [N, M] × input: [M]
    */
    const int64_t M = input->Dims()[0];
    const int64_t N = other->Dims()[0];
    // TODO(zbl): support batched outer?
    CHECK_EQ(grad_output->Dims().size(), 2);
    CHECK_EQ(grad_output->Dims()[0], M);
    CHECK_EQ(grad_output->Dims()[1], N);

    auto input_dtype = input->Dtype();
    auto other_dtype = other->Dtype();

    DataType promoted_type = DispatchFunc<DataTypeList<INFINI_ALL_TYPES>, DataTypeList<INFINI_ALL_TYPES>>(
        {input_dtype, other_dtype}, [=]<typename Tin, typename To>() { return DataTypeMap_v<WidestType_t<Tin, To>>; },
        "CUDA OuterForward");

    auto input_ = input_dtype == promoted_type ? input : std::make_shared<Tensor>(input->To(promoted_type));
    auto other_ = other_dtype == promoted_type ? other : std::make_shared<Tensor>(other->To(promoted_type));

    auto grad_input = std::make_shared<Tensor>(std::vector<int64_t>{M}, promoted_type, grad_output->GetDevice());
    auto grad_other = std::make_shared<Tensor>(std::vector<int64_t>{N}, promoted_type, grad_output->GetDevice());

    DispatchFunc<DataType::kFLOAT32, DataType::kBFLOAT16>(
        promoted_type,
        [=]<typename T>() {
            grad_input->Fill<T>(0);
            grad_other->Fill<T>(0);
        },
        "CUDA OuterBackward");

    const auto *cuda_device = dynamic_cast<const CudaDevice *>(input->GetDevice());
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasHandle_t handle = cuda_device->CublasHandle();

    switch (promoted_type) {
        DISPATCH_CASE(WRAP({
                          // grad_input[M, 1] = grad_output[M, N] × other[N, 1]
                          // y = grad_input[M]
                          // A = grad_output.T[N, M]
                          // x = other[N]
                          CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_T, N, M, &alpha,
                                                   static_cast<const float *>(grad_output->DataPtr()), N,
                                                   static_cast<const float *>(other->DataPtr()), 1, &beta,
                                                   static_cast<float *>(grad_input->DataPtr()), 1));

                          // grad_other[N, 1] = grad_output.T[N, M] × input[M, 1]
                          // y = grad_other[N]
                          // A = grad_output.T[N, M]
                          // x = input[M]
                          CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_N, N, M, &alpha,
                                                   static_cast<const float *>(grad_output->DataPtr()), N,
                                                   static_cast<const float *>(input->DataPtr()), 1, &beta,
                                                   static_cast<float *>(grad_other->DataPtr()), 1));
                      }),
                      DataType::kFLOAT32)
        DISPATCH_CASE(
            // cublas<t>gemv does not support bf16, use cublasGemmEx to workaround
            WRAP({
                // grad_input[M, 1] = grad_output[M, N] × other[N, 1]
                // grad_input.T[1, M] = other.T[1, N] × grad_output.T[N, M]
                // C = grad_input.T[1, M]
                // A = other.T[1, N]
                // B = grad_output.T[N, M]
                CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, M, N, &alpha, other->DataPtr(),
                                          CUDA_R_16BF, 1, grad_output->DataPtr(), CUDA_R_16BF, N, &beta,
                                          grad_input->DataPtr(), CUDA_R_16BF, 1, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
                // grad_other[N, 1] = grad_output.T[N, M] × input[M, 1]
                // grad_other.T[1, N] = input.T[1, M] × grad_output[M, N]
                // C = grad_other.T[1, N]
                // A = input.T[1, M]
                // B = grad_output.T[N, M]
                CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, 1, N, M, &alpha, input->DataPtr(),
                                          CUDA_R_16BF, 1, grad_output->DataPtr(), CUDA_R_16BF, N, &beta,
                                          grad_other->DataPtr(), CUDA_R_16BF, 1, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
            }),
            DataType::kBFLOAT16)
    }

    return {grad_input, grad_other};
}

} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_OUTER_KERNEL(kernel_name)                                                                        \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_OUTER_KERNEL(OuterForward)
REGISTER_CUDA_OUTER_KERNEL(OuterBackward)

#undef REGISTER_CUDA_OUTER_KERNEL
