#include "glog/logging.h"

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/cuda/cuda_dispatch.h"
#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

namespace infini_train::kernels::cuda {

template <typename T>
__global__ void ScatterForwardKernel(const T *__restrict__ values, const int64_t *__restrict__ indices,
                                     T *__restrict__ output, int64_t rows, int64_t topk, int64_t num_experts) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = rows * topk;
    if (idx >= total) {
        return;
    }

    const int64_t row = idx / topk;
    const int64_t expert_idx = indices[idx];
    output[row * num_experts + expert_idx] = values[idx];
}

std::shared_ptr<Tensor> ScatterForward(const std::shared_ptr<Tensor> &values, const std::shared_ptr<Tensor> &indices,
                                       const std::vector<int64_t> &output_dims) {
    CHECK(indices->Dtype() == DataType::kINT64) << "CUDA ScatterForward expects int64 indices";
    CHECK(values->Dims() == indices->Dims());
    CHECK(!output_dims.empty());
    CHECK_EQ(values->Dims().size(), output_dims.size());
    CHECK_GT(values->Dims().back(), 0);
    CHECK_GT(output_dims.back(), 0);

    const int64_t topk = values->Dims().back();
    const int64_t num_experts = output_dims.back();
    CHECK_GT(num_experts, 0);
    const int64_t rows = values->NumElements() / topk;
    const int64_t output_numel = std::accumulate(output_dims.begin(), output_dims.end(), 1, std::multiplies<int64_t>());
    CHECK_EQ(output_numel, static_cast<size_t>(rows * num_experts));

    auto output = std::make_shared<Tensor>(output_dims, values->Dtype(), values->GetDevice());

    auto device = values->GetDevice();
    const auto &stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                             infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                             ->cuda_stream();
    CUDA_CHECK(cudaMemsetAsync(output->DataPtr(), 0, output->SizeInBytes(), stream));
    const int threads = 256;
    const int blocks = static_cast<int>(((rows * topk) + threads - 1) / threads);
    core::cuda::DispatchCudaFunc<INFINI_ALL_TYPES>(
        values->Dtype(),
        [=]<typename T>() {
            ScatterForwardKernel<T><<<blocks, threads, 0, stream>>>(
                static_cast<const T *>(values->DataPtr()), static_cast<const int64_t *>(indices->DataPtr()),
                static_cast<T *>(output->DataPtr()), rows, topk, num_experts);
        },
        "CUDA ScatterForward");
    return output;
}

template <typename T>
__global__ void ScatterBackwardKernel(const T *__restrict__ grad_output, const int64_t *__restrict__ indices,
                                      T *__restrict__ grad_values, int64_t rows, int64_t topk, int64_t num_experts) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = rows * topk;
    if (idx >= total) {
        return;
    }
    const int64_t row = idx / topk;
    const int64_t expert_idx = indices[idx];
    grad_values[idx] = grad_output[row * num_experts + expert_idx];
}

std::shared_ptr<Tensor> ScatterBackward(const std::shared_ptr<Tensor> &grad_output,
                                        const std::shared_ptr<Tensor> &indices) {
    CHECK(indices->Dtype() == DataType::kINT64) << "CUDA ScatterBackward expects int64 indices";
    CHECK_GE(grad_output->Dims().size(), 1);
    CHECK_GE(indices->Dims().size(), 1);
    const int64_t num_experts = grad_output->Dims().back();
    const int64_t topk = indices->Dims().back();
    const int64_t rows = indices->NumElements() / topk;
    CHECK_EQ(grad_output->NumElements(), static_cast<size_t>(rows * num_experts));

    auto grad_values = std::make_shared<Tensor>(indices->Dims(), grad_output->Dtype(), grad_output->GetDevice());

    auto device = grad_output->GetDevice();
    const auto &stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                             infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                             ->cuda_stream();
    const int threads = 256;
    const int blocks = static_cast<int>(((rows * topk) + threads - 1) / threads);

    core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
        grad_output->Dtype(),
        [=]<typename T>() {
            ScatterBackwardKernel<T><<<blocks, threads, 0, stream>>>(
                static_cast<const T *>(grad_output->DataPtr()), static_cast<const int64_t *>(indices->DataPtr()),
                static_cast<T *>(grad_values->DataPtr()), rows, topk, num_experts);
        },
        "CUDA ScatterBackward");

    return grad_values;
}

} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_SCATTER_KERNEL(kernel_name)                                                                      \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_SCATTER_KERNEL(ScatterForward)
REGISTER_CUDA_SCATTER_KERNEL(ScatterBackward)

#undef REGISTER_CUDA_SCATTER_KERNEL
