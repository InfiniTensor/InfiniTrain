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
    GemmParams p;
    p.trans_a = CUBLAS_OP_N;
    p.trans_b = CUBLAS_OP_N;
    p.m = static_cast<int>(N);
    p.n = static_cast<int>(M);
    p.k = 1;
    p.A = other->DataPtr();
    p.lda = static_cast<int>(N);
    p.B = input->DataPtr();
    p.ldb = 1;
    p.C = output->DataPtr();
    p.ldc = static_cast<int>(N);
    p.alpha = 1.0f;
    p.beta = 0.0f;
    p.batch_count = 1;
    p.input_dtype = dtype;
    p.output_dtype = dtype;
    p.blas_handle = GetCublasHandle(device);

    GemmCuda(p);

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
    DataType promoted_type = DispatchFunc<DataTypeList<INFINI_ALL_TYPES>, DataTypeList<INFINI_ALL_TYPES>>(
        {input_dtype, other_dtype}, [=]<typename Tin, typename To>() { return DataTypeMap_v<WidestType_t<Tin, To>>; },
        "CUDA OuterBackward");

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
        // fp32: use cublasSgemv (specialized matrix-vector kernel, more efficient than GEMM for this shape)
        // cublasSgemv does not support bf16, so bf16 falls through to GemmCuda below.
        float alpha = 1.0f, beta = 0.0f;
        cublasHandle_t handle = GetCublasHandle(device);

        // grad_input[M] = grad_output[M, N] × other[N]
        // y = grad_input[M], A = grad_output.T[N, M], x = other[N]
        CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_T, N, M, &alpha,
                                 static_cast<const float *>(grad_output_promoted->DataPtr()), N,
                                 static_cast<const float *>(other_promoted->DataPtr()), 1, &beta,
                                 static_cast<float *>(grad_input->DataPtr()), 1));

        // grad_other[N] = grad_output.T[N, M] × input[M]
        // y = grad_other[N], A = grad_output.T[N, M], x = input[M]
        CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_N, N, M, &alpha,
                                 static_cast<const float *>(grad_output_promoted->DataPtr()), N,
                                 static_cast<const float *>(input_promoted->DataPtr()), 1, &beta,
                                 static_cast<float *>(grad_other->DataPtr()), 1));
        break;
    }
    case DataType::kBFLOAT16: {
        // bf16: cublasSgemv does not support bf16; use GemmCuda (GEMM with k=M or k=N).

        // grad_input[M] = grad_output[M, N] × other[N]
        // grad_input.T[1, M] = other.T[1, N] × grad_output.T[N, M]
        // C[1,M] = A[1,N] * B[N,M]
        GemmParams p_input;
        p_input.trans_a = CUBLAS_OP_N;
        p_input.trans_b = CUBLAS_OP_N;
        p_input.m = 1;
        p_input.n = static_cast<int>(M);
        p_input.k = static_cast<int>(N);
        p_input.A = other_promoted->DataPtr();
        p_input.lda = 1;
        p_input.B = grad_output_promoted->DataPtr();
        p_input.ldb = static_cast<int>(N);
        p_input.C = grad_input->DataPtr();
        p_input.ldc = 1;
        p_input.alpha = 1.0f;
        p_input.beta = 0.0f;
        p_input.batch_count = 1;
        p_input.input_dtype = promoted_type;
        p_input.output_dtype = output_dtype;
        p_input.blas_handle = GetCublasHandle(device);
        GemmCuda(p_input);

        // grad_other[N] = grad_output.T[N, M] × input[M]
        // grad_other.T[1, N] = input.T[1, M] × grad_output[M, N]
        // C[1,N] = A[1,M] * B[M,N]  (B stored as grad_output.T[N,M], so ldb=N, trans_b=T)
        GemmParams p_other;
        p_other.trans_a = CUBLAS_OP_N;
        p_other.trans_b = CUBLAS_OP_T;
        p_other.m = 1;
        p_other.n = static_cast<int>(N);
        p_other.k = static_cast<int>(M);
        p_other.A = input_promoted->DataPtr();
        p_other.lda = 1;
        p_other.B = grad_output_promoted->DataPtr();
        p_other.ldb = static_cast<int>(N);
        p_other.C = grad_other->DataPtr();
        p_other.ldc = 1;
        p_other.alpha = 1.0f;
        p_other.beta = 0.0f;
        p_other.batch_count = 1;
        p_other.input_dtype = promoted_type;
        p_other.output_dtype = output_dtype;
        p_other.blas_handle = GetCublasHandle(device);
        GemmCuda(p_other);
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
