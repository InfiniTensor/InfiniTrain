#include <memory>
#include <tuple>

#include "cublas_v2.h"
#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
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

    auto output = std::make_shared<Tensor>(std::vector<int64_t>{M, N}, DataType::kFLOAT32, input->GetDevice());

    const auto *cuda_device = dynamic_cast<const CudaDevice *>(input->GetDevice());
    // reinterpret input: [M] as column vector [M, 1]
    // reinterpret other: [N] as row vector [1, N]
    // output[M, N] = input[M, 1] * other.T[1, N]
    // output.T[N, M] = other[N, 1] * input.T[1, M]
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, cuda_device->Stream()));

    CUBLAS_CHECK(cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, 1, &alpha, static_cast<const float *>(other->DataPtr()), N,
        static_cast<const float *>(input->DataPtr()), 1, &beta, static_cast<float *>(output->DataPtr()), N));

    CUBLAS_CHECK(cublasDestroy(handle));
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

    auto grad_input = std::make_shared<Tensor>(std::vector<int64_t>{M}, DataType::kFLOAT32, grad_output->GetDevice());
    auto grad_other = std::make_shared<Tensor>(std::vector<int64_t>{N}, DataType::kFLOAT32, grad_output->GetDevice());
    grad_input->Fill<float>(0.0f);
    grad_other->Fill<float>(0.0f);

    const auto *cuda_device = dynamic_cast<const CudaDevice *>(input->GetDevice());
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, cuda_device->Stream()));

    // grad_input[M, 1] = grad_output[M, N] × other[N, 1]
    // y = grad_input[M]
    // A = grad_output.T[N, M]
    // x = other[N]
    CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_T, N, M, &alpha, static_cast<const float *>(grad_output->DataPtr()), N,
                             static_cast<const float *>(other->DataPtr()), 1, &beta,
                             static_cast<float *>(grad_input->DataPtr()), 1));

    // grad_other[N, 1] = grad_output.T[N, M] × input[M, 1]
    // y = grad_other[N]
    // A = grad_output.T[N, M]
    // x = input[M]
    CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_N, N, M, &alpha, static_cast<const float *>(grad_output->DataPtr()), N,
                             static_cast<const float *>(input->DataPtr()), 1, &beta,
                             static_cast<float *>(grad_other->DataPtr()), 1));

    CUBLAS_CHECK(cublasDestroy(handle));
    return {grad_input, grad_other};
}

} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_OUTER_KERNEL(kernel_name)                                                                        \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_OUTER_KERNEL(OuterForward)
REGISTER_CUDA_OUTER_KERNEL(OuterBackward)

#undef REGISTER_CUDA_OUTER_KERNEL
