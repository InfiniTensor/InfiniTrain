#include <functional>
#include <memory>
#include <numeric>
#include <vector>

#include <cublas_v2.h>

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/common/cuda/gemm.cuh"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

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
    for (int64_t i = 0; i < static_cast<int64_t>(input_dims.size()) - 2; ++i) {
        CHECK_EQ(input_dims[i], other_dims[i]) << "Batch dims must match";
    }

    auto dtype = input->Dtype();
    std::vector<int64_t> output_dims = input_dims;
    output_dims[output_dims.size() - 1] = n;
    auto output = std::make_shared<Tensor>(output_dims, dtype, input->GetDevice());

    auto device = input->GetDevice();

    // cuBLAS is colmun-major
    // output = input * other --> output.T = other.T * input.T
    // C = A * B ==> output.T[*, n, m] = other.T[*, n, k] * input.T[*, k, m]
    // C = output.T[*, n, m]
    // A = other.T[*, n, k]
    // B = input.T[*, k, m]
    // NOTE(zbl): the last cublasGemmAlgo_t param has no effect on GPU arch >= sm_80(Ampere)
    GemmParams p;
    p.trans_a = CUBLAS_OP_N;
    p.trans_b = CUBLAS_OP_N;
    p.m = static_cast<int>(n);
    p.n = static_cast<int>(m);
    p.k = static_cast<int>(k);
    p.A = other->DataPtr();
    p.lda = static_cast<int>(n);
    p.stride_a = n * k;
    p.B = input->DataPtr();
    p.ldb = static_cast<int>(k);
    p.stride_b = k * m;
    p.C = output->DataPtr();
    p.ldc = static_cast<int>(n);
    p.stride_c = m * n;
    p.alpha = 1.0f;
    p.beta = 0.0f;
    p.batch_count = static_cast<int>(bs);
    p.input_dtype = dtype;
    p.output_dtype = dtype;
    p.blas_handle = GetCublasHandle(device);

    GemmCuda(p);

    return output;
}

std::shared_ptr<Tensor> MatmulBackwardInput(const std::shared_ptr<Tensor> &other,
                                            const std::shared_ptr<Tensor> &grad_output,
                                            const std::vector<int64_t> &input_dims) {
    /*
       grad_input[*, m, k] = grad_output[*, m, n] * other[*, k, n]^T
    */

    const auto &other_dims = other->Dims();
    const auto &grad_output_dims = grad_output->Dims();

    CHECK_GE(other_dims.size(), 2);
    CHECK_EQ(other_dims.size(), grad_output_dims.size());

    const int64_t m = grad_output_dims[grad_output_dims.size() - 2];
    const int64_t k = other_dims[other_dims.size() - 2];
    const int64_t n = other_dims[other_dims.size() - 1];
    CHECK_EQ(k, other_dims[other_dims.size() - 2]);
    CHECK_EQ(m, grad_output_dims[grad_output_dims.size() - 2]);
    CHECK_EQ(n, grad_output_dims[grad_output_dims.size() - 1]);

    const int64_t bs
        = std::accumulate(grad_output_dims.rbegin() + 2, grad_output_dims.rend(), 1, std::multiplies<int64_t>{});
    for (int64_t i = 0; i < static_cast<int64_t>(grad_output_dims.size()) - 2; ++i) {
        CHECK_EQ(grad_output_dims[i], other_dims[i]) << "Batch dims must match";
    }

    auto compute_dtype = other->Dtype();
    auto grad_output_dtype = grad_output->Dtype();
    auto grad_output_promoted
        = grad_output_dtype == compute_dtype ? grad_output : std::make_shared<Tensor>(grad_output->To(compute_dtype));

    // For bf16 compute, output in fp32 to preserve accumulation precision.
    auto output_dtype = (compute_dtype == DataType::kBFLOAT16) ? DataType::kFLOAT32 : compute_dtype;
    auto grad_input = std::make_shared<Tensor>(input_dims, output_dtype, grad_output->GetDevice());

    // No Fill(0) needed: cuBLAS beta=0.0f means C is fully overwritten, never read.

    auto device = grad_output->GetDevice();

    // cuBLAS is colmun-major
    // grad_input = grad_output * other.T --> grad_input.T = other * grad_output.T
    // C = A.T * B ==> grad_input.T[*, k, m] = other[*, k, n] * grad_output.T[*, n, m]
    // C = grad_input.T[*, k, m]
    // A = other.T[*, n, k]
    // B = grad_output.T[*, n, m]
    GemmParams p;
    p.trans_a = CUBLAS_OP_T;
    p.trans_b = CUBLAS_OP_N;
    p.m = static_cast<int>(k);
    p.n = static_cast<int>(m);
    p.k = static_cast<int>(n);
    p.A = other->DataPtr();
    p.lda = static_cast<int>(n);
    p.stride_a = k * n;
    p.B = grad_output_promoted->DataPtr();
    p.ldb = static_cast<int>(n);
    p.stride_b = n * m;
    p.C = grad_input->DataPtr();
    p.ldc = static_cast<int>(k);
    p.stride_c = m * k;
    p.alpha = 1.0f;
    p.beta = 0.0f;
    p.batch_count = static_cast<int>(bs);
    p.input_dtype = compute_dtype;
    p.output_dtype = output_dtype;
    p.blas_handle = GetCublasHandle(device);

    GemmCuda(p);

    return grad_input;
}

std::shared_ptr<Tensor> MatmulBackwardOther(const std::shared_ptr<Tensor> &input1,
                                            const std::shared_ptr<Tensor> &grad_output,
                                            const std::vector<int64_t> &other_dims) {
    /*
       grad_other[*, k, n] = input[*, m, k]^T * grad_output[*, m, n]
    */

    const auto &input_dims = input1->Dims();
    const auto &grad_output_dims = grad_output->Dims();
    CHECK_GE(input_dims.size(), 2);
    CHECK_EQ(input_dims.size(), grad_output_dims.size());

    const int64_t m = input_dims[input_dims.size() - 2];
    const int64_t k = input_dims[input_dims.size() - 1];
    const int64_t n = grad_output_dims[grad_output_dims.size() - 1];
    CHECK_EQ(m, grad_output_dims[grad_output_dims.size() - 2]);
    CHECK_EQ(n, grad_output_dims[grad_output_dims.size() - 1]);
    CHECK_EQ(input_dims[input_dims.size() - 1], other_dims[other_dims.size() - 2]);

    const int64_t bs = std::accumulate(input_dims.rbegin() + 2, input_dims.rend(), 1, std::multiplies<int64_t>{});
    for (int64_t i = 0; i < static_cast<int64_t>(input_dims.size()) - 2; ++i) {
        CHECK_EQ(input_dims[i], grad_output_dims[i]) << "Batch dims must match";
        CHECK_EQ(input_dims[i], other_dims[i]) << "Batch dims must match";
    }

    auto compute_dtype = input1->Dtype();
    auto grad_output_dtype = grad_output->Dtype();
    auto grad_output_promoted
        = grad_output_dtype == compute_dtype ? grad_output : std::make_shared<Tensor>(grad_output->To(compute_dtype));

    // For bf16 compute, output in fp32 to preserve accumulation precision.
    auto output_dtype = (compute_dtype == DataType::kBFLOAT16) ? DataType::kFLOAT32 : compute_dtype;
    auto grad_other = std::make_shared<Tensor>(other_dims, output_dtype, grad_output->GetDevice());

    // No Fill(0) needed: cuBLAS beta=0.0f means C is fully overwritten, never read.

    auto device = grad_output->GetDevice();

    // cuBLAS is colmun-major
    // grad_other = input.T * grad_output --> grad_other.T = grad_output.T * input
    // C = A * B.T ==> grad_other.T[*, n, k] = grad_output.T[*, n, m] * input[*, m, k]
    // C = grad_other.T[*, n, k]
    // A = grad_output.T[*, n, m]
    // B = input.T[*, k, m]
    GemmParams p;
    p.trans_a = CUBLAS_OP_N;
    p.trans_b = CUBLAS_OP_T;
    p.m = static_cast<int>(n);
    p.n = static_cast<int>(k);
    p.k = static_cast<int>(m);
    p.A = grad_output_promoted->DataPtr();
    p.lda = static_cast<int>(n);
    p.stride_a = n * m;
    p.B = input1->DataPtr();
    p.ldb = static_cast<int>(k);
    p.stride_b = k * m;
    p.C = grad_other->DataPtr();
    p.ldc = static_cast<int>(n);
    p.stride_c = n * k;
    p.alpha = 1.0f;
    p.beta = 0.0f;
    p.batch_count = static_cast<int>(bs);
    p.input_dtype = compute_dtype;
    p.output_dtype = output_dtype;
    p.blas_handle = GetCublasHandle(device);

    GemmCuda(p);

    return grad_other;
}

} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_MATMUL_KERNEL(kernel_name)                                                                       \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_MATMUL_KERNEL(MatmulForward)
REGISTER_CUDA_MATMUL_KERNEL(MatmulBackwardInput)
REGISTER_CUDA_MATMUL_KERNEL(MatmulBackwardOther)

#undef REGISTER_CUDA_MATMUL_KERNEL
