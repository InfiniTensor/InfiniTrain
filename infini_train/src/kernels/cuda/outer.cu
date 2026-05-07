#include <memory>
#include <tuple>
#include <vector>

#include <cublas_v2.h>

#include "glog/logging.h"

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/common/cuda/gemm.cuh"
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

    auto dtype = input->Dtype();
    auto output = std::make_shared<Tensor>(std::vector<int64_t>{M, N}, dtype, input->GetDevice());

    auto device = input->GetDevice();

    // reinterpret input: [M] as column vector [M, 1]
    // reinterpret other: [N] as row vector [1, N]
    // output[M, N] = input[M, 1] * other.T[1, N]
    // output.T[N, M] = other[N, 1] * input.T[1, M]
    // This is a GEMM with k=1: C[N,M] = A[N,1] * B[1,M]
    GemmCuda(device, GemmParams{
                         .trans_a = CUBLAS_OP_N,
                         .trans_b = CUBLAS_OP_N,
                         .m = static_cast<int>(N),
                         .n = static_cast<int>(M),
                         .k = 1,
                         .A = other->DataPtr(),
                         .lda = static_cast<int>(N),
                         .B = input->DataPtr(),
                         .ldb = 1,
                         .C = output->DataPtr(),
                         .ldc = static_cast<int>(N),
                         .alpha = 1.0f,
                         .beta = 0.0f,
                         .batch_count = 1,
                         .input_dtype = dtype,
                         .output_dtype = dtype,
                     });

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
    auto grad_output_dtype = grad_output->Dtype();

    // Compute dtype determined by saved tensors (forward compute dtype), not grad_output
    DataType promoted_type = PromoteDataTypes(input_dtype, other_dtype);

    auto input_promoted = input_dtype == promoted_type ? input : std::make_shared<Tensor>(input->To(promoted_type));
    auto other_promoted = other_dtype == promoted_type ? other : std::make_shared<Tensor>(other->To(promoted_type));
    auto grad_output_promoted
        = grad_output_dtype == promoted_type ? grad_output : std::make_shared<Tensor>(grad_output->To(promoted_type));

    // For bf16 compute, output in fp32 to preserve accumulation precision.
    auto output_dtype = (promoted_type == DataType::kBFLOAT16) ? DataType::kFLOAT32 : promoted_type;
    auto grad_input = std::make_shared<Tensor>(std::vector<int64_t>{M}, output_dtype, grad_output->GetDevice());
    auto grad_other = std::make_shared<Tensor>(std::vector<int64_t>{N}, output_dtype, grad_output->GetDevice());

    auto device = input->GetDevice();

    switch (promoted_type) {
    case DataType::kFLOAT32: {
        // grad_input[M] = grad_output[M, N] × other[N]
        SgemvCuda(device, SgemvParams{
                              .trans = CUBLAS_OP_T,
                              .m = static_cast<int>(N),
                              .n = static_cast<int>(M),
                              .A = static_cast<const float *>(grad_output_promoted->DataPtr()),
                              .lda = static_cast<int>(N),
                              .x = static_cast<const float *>(other_promoted->DataPtr()),
                              .y = static_cast<float *>(grad_input->DataPtr()),
                          });

        // grad_other[N] = grad_output.T[N, M] × input[M]
        SgemvCuda(device, SgemvParams{
                              .trans = CUBLAS_OP_N,
                              .m = static_cast<int>(N),
                              .n = static_cast<int>(M),
                              .A = static_cast<const float *>(grad_output_promoted->DataPtr()),
                              .lda = static_cast<int>(N),
                              .x = static_cast<const float *>(input_promoted->DataPtr()),
                              .y = static_cast<float *>(grad_other->DataPtr()),
                          });
        break;
    }
    case DataType::kBFLOAT16: {
        // bf16: cublasSgemv does not support bf16; use GemmCuda (GEMM with k=M or k=N).

        // grad_input[M] = grad_output[M, N] × other[N]
        // grad_input.T[1, M] = other.T[1, N] × grad_output.T[N, M]
        // C[1,M] = A[1,N] * B[N,M]
        GemmCuda(device, GemmParams{
                             .trans_a = CUBLAS_OP_N,
                             .trans_b = CUBLAS_OP_N,
                             .m = 1,
                             .n = static_cast<int>(M),
                             .k = static_cast<int>(N),
                             .A = other_promoted->DataPtr(),
                             .lda = 1,
                             .B = grad_output_promoted->DataPtr(),
                             .ldb = static_cast<int>(N),
                             .C = grad_input->DataPtr(),
                             .ldc = 1,
                             .alpha = 1.0f,
                             .beta = 0.0f,
                             .batch_count = 1,
                             .input_dtype = promoted_type,
                             .output_dtype = output_dtype,
                         });

        // grad_other[N] = grad_output.T[N, M] × input[M]
        // grad_other.T[1, N] = input.T[1, M] × grad_output[M, N]
        // C[1,N] = A[1,M] * B[M,N]  (B stored as grad_output.T[N,M], so ldb=N, trans_b=T)
        GemmCuda(device, GemmParams{
                             .trans_a = CUBLAS_OP_N,
                             .trans_b = CUBLAS_OP_T,
                             .m = 1,
                             .n = static_cast<int>(N),
                             .k = static_cast<int>(M),
                             .A = input_promoted->DataPtr(),
                             .lda = 1,
                             .B = grad_output_promoted->DataPtr(),
                             .ldb = static_cast<int>(N),
                             .C = grad_other->DataPtr(),
                             .ldc = 1,
                             .alpha = 1.0f,
                             .beta = 0.0f,
                             .batch_count = 1,
                             .input_dtype = promoted_type,
                             .output_dtype = output_dtype,
                         });
        break;
    }
    default:
        LOG(FATAL) << "CUDA OuterBackward: unsupported dtype";
    }

    return {grad_input, grad_other};
}

} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_OUTER_KERNEL(kernel_name)                                                                        \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_OUTER_KERNEL(OuterForward)
REGISTER_CUDA_OUTER_KERNEL(OuterBackward)

#undef REGISTER_CUDA_OUTER_KERNEL
