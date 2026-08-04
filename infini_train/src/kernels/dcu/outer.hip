#include <memory>
#include <tuple>
#include <vector>

#include <hipblas/hipblas.h>

#include "glog/logging.h"

#include "infini_train/include/common/dcu/common_dcu.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/dcu/dcu_dispatch.h"
#include "infini_train/src/core/runtime/dcu/dcu_runtime_common.h"

namespace infini_train::kernels::dcu {
std::shared_ptr<Tensor> OuterForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
    /*
    Computes outer product: output[i, j] = input[i] * other[j]
    Equivalent to: input: [M, 1], other: [1, N] 鈫?output: [M, N]
    */

    const auto &in_dims = input->Dims();
    const auto &ot_dims = other->Dims();
    // TODO(zbl): support batched outer?
    CHECK_EQ(in_dims.size(), 1);
    CHECK_EQ(ot_dims.size(), 1);

    const int64_t M = in_dims[0];
    const int64_t N = ot_dims[0];

    auto output = std::make_shared<Tensor>(std::vector<int64_t>{M, N}, input->Dtype(), input->GetDevice());

    auto device = input->GetDevice();
    // reinterpret input: [M] as column vector [M, 1]
    // reinterpret other: [N] as row vector [1, N]
    // output[M, N] = input[M, 1] * other.T[1, N]
    // output.T[N, M] = other[N, 1] * input.T[1, M]
    float alpha = 1.0f;
    float beta = 0.0f;
    hipblasHandle_t handle = dynamic_cast<infini_train::core::dcu::DcuBlasHandle *>(
                                infini_train::core::GetDeviceGuardImpl(device.type())->GetBlasHandle(device))
                                ->hipblas_handle();

    switch (input->Dtype()) {
        DISPATCH_CASE(WRAP({
                          HIPBLAS_CHECK(hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, N, M, 1, &alpha,
                                                   static_cast<const float *>(other->DataPtr()), N,
                                                   static_cast<const float *>(input->DataPtr()), 1, &beta,
                                                   static_cast<float *>(output->DataPtr()), N));
                      }),
                      DataType::kFLOAT32)
        DISPATCH_CASE(WRAP({
                          HIPBLAS_CHECK(hipblasGemmEx(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, N, M, 1, &alpha, other->DataPtr(),
                                                    HIPBLAS_R_16B, N, input->DataPtr(), HIPBLAS_R_16B, 1, &beta,
                                                    output->DataPtr(), HIPBLAS_R_16B, N, HIPBLAS_R_32F, HIPBLAS_GEMM_DEFAULT));
                      }),
                      DataType::kBFLOAT16)
    }

    return output;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> OuterBackward(const std::shared_ptr<Tensor> &input,
                                                                           const std::shared_ptr<Tensor> &other,
                                                                           const std::shared_ptr<Tensor> &grad_output) {
    /*
    grad_input: [M] = grad_output: [M, N] 脳 other: [N]
    grad_other: [N] = grad_output.T: [N, M] 脳 input: [M]
    */
    const int64_t M = input->Dims()[0];
    const int64_t N = other->Dims()[0];
    // TODO(zbl): support batched outer?
    CHECK_EQ(grad_output->Dims().size(), 2);
    CHECK_EQ(grad_output->Dims()[0], M);
    CHECK_EQ(grad_output->Dims()[1], N);

    auto input_dtype = input->Dtype();
    auto other_dtype = other->Dtype();
    auto grad_output_dtype = grad_output->Dtype();

    // Compute dtype determined by saved tensors (forward compute dtype), not grad_output
    DataType promoted_type = PromoteDataTypes(input_dtype, other_dtype);

    auto input_promoted = input_dtype == promoted_type ? input : std::make_shared<Tensor>(input->To(promoted_type));
    auto other_promoted = other_dtype == promoted_type ? other : std::make_shared<Tensor>(other->To(promoted_type));
    auto grad_output_promoted
        = grad_output_dtype == promoted_type ? grad_output : std::make_shared<Tensor>(grad_output->To(promoted_type));

    // For bf16 compute, output in fp32 to preserve accumulation precision (matches PyTorch behavior)
    auto output_dtype = (promoted_type == DataType::kBFLOAT16) ? DataType::kFLOAT32 : promoted_type;
    auto grad_input = std::make_shared<Tensor>(std::vector<int64_t>{M}, output_dtype, grad_output->GetDevice());
    auto grad_other = std::make_shared<Tensor>(std::vector<int64_t>{N}, output_dtype, grad_output->GetDevice());

    core::dcu::DispatchDcuFunc<DataType::kFLOAT32, DataType::kBFLOAT16>(
        promoted_type,
        [=]<typename T>() {
            grad_input->Fill(0.0);
            grad_other->Fill(0.0);
        },
        "HIP OuterBackward");

    auto device = input->GetDevice();
    float alpha = 1.0f;
    float beta = 0.0f;
    hipblasHandle_t handle = dynamic_cast<infini_train::core::dcu::DcuBlasHandle *>(
                                infini_train::core::GetDeviceGuardImpl(device.type())->GetBlasHandle(device))
                                ->hipblas_handle();

    switch (promoted_type) {
        DISPATCH_CASE(WRAP({
                          // grad_input[M, 1] = grad_output[M, N] 脳 other[N, 1]
                          // y = grad_input[M]
                          // A = grad_output.T[N, M]
                          // x = other[N]
                          HIPBLAS_CHECK(hipblasSgemv(handle, HIPBLAS_OP_T, N, M, &alpha,
                                                   static_cast<const float *>(grad_output_promoted->DataPtr()), N,
                                                   static_cast<const float *>(other_promoted->DataPtr()), 1, &beta,
                                                   static_cast<float *>(grad_input->DataPtr()), 1));

                          // grad_other[N, 1] = grad_output.T[N, M] 脳 input[M, 1]
                          // y = grad_other[N]
                          // A = grad_output.T[N, M]
                          // x = input[M]
                          HIPBLAS_CHECK(hipblasSgemv(handle, HIPBLAS_OP_N, N, M, &alpha,
                                                   static_cast<const float *>(grad_output_promoted->DataPtr()), N,
                                                   static_cast<const float *>(input_promoted->DataPtr()), 1, &beta,
                                                   static_cast<float *>(grad_other->DataPtr()), 1));
                      }),
                      DataType::kFLOAT32)
        DISPATCH_CASE(
            // hipblas<t>gemv does not support bf16, use hipblasGemmEx to workaround
            WRAP({
                // grad_input[M, 1] = grad_output[M, N] 脳 other[N, 1]
                // grad_input.T[1, M] = other.T[1, N] 脳 grad_output.T[N, M]
                // C = grad_input.T[1, M]
                // A = other.T[1, N]
                // B = grad_output.T[N, M]
                HIPBLAS_CHECK(hipblasGemmEx(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, 1, M, N, &alpha, other_promoted->DataPtr(),
                                          HIPBLAS_R_16B, 1, grad_output_promoted->DataPtr(), HIPBLAS_R_16B, N, &beta,
                                          grad_input->DataPtr(), HIPBLAS_R_32F, 1, HIPBLAS_R_32F, HIPBLAS_GEMM_DEFAULT));
                // grad_other[N, 1] = grad_output.T[N, M] 脳 input[M, 1]
                // grad_other.T[1, N] = input.T[1, M] 脳 grad_output[M, N]
                // C = grad_other.T[1, N]
                // A = input.T[1, M]
                // B = grad_output.T[N, M]
                HIPBLAS_CHECK(hipblasGemmEx(handle, HIPBLAS_OP_N, HIPBLAS_OP_T, 1, N, M, &alpha, input_promoted->DataPtr(),
                                          HIPBLAS_R_16B, 1, grad_output_promoted->DataPtr(), HIPBLAS_R_16B, N, &beta,
                                          grad_other->DataPtr(), HIPBLAS_R_32F, 1, HIPBLAS_R_32F, HIPBLAS_GEMM_DEFAULT));
            }),
            DataType::kBFLOAT16)
    }

    return {grad_input, grad_other};
}

} // namespace infini_train::kernels::dcu

#define REGISTER_HIP_OUTER_KERNEL(kernel_name)                                                                        \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kDCU, kernel_name, infini_train::kernels::dcu::kernel_name)

REGISTER_HIP_OUTER_KERNEL(OuterForward)
REGISTER_HIP_OUTER_KERNEL(OuterBackward)

#undef REGISTER_HIP_OUTER_KERNEL
