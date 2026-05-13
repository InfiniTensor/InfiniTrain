#include "glog/logging.h"

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/cuda/cuda_dispatch.h"
#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

namespace infini_train::kernels::cuda {

template <typename T>
__global__ void TopKMaskForwardKernel(const T *__restrict__ input, T *__restrict__ output, int64_t rows,
                                      int64_t num_experts, int64_t topk) {
    int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    const int64_t offset = row * num_experts;
    float selected_sum = 0.0f;
    for (int64_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
        const float value = static_cast<float>(input[offset + expert_idx]);
        int64_t rank = 0;
        for (int64_t other_idx = 0; other_idx < num_experts; ++other_idx) {
            const float other_value = static_cast<float>(input[offset + other_idx]);
            if (other_value > value || (other_value == value && other_idx < expert_idx)) {
                ++rank;
            }
        }
        const bool selected = rank < topk;
        output[offset + expert_idx] = selected ? input[offset + expert_idx] : T(0.0f);
        selected_sum += selected ? value : 0.0f;
    }
    if (topk > 1 && selected_sum != 0.0f) {
        for (int64_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
            if (static_cast<float>(output[offset + expert_idx]) != 0.0f) {
                output[offset + expert_idx] = T(static_cast<float>(output[offset + expert_idx]) / selected_sum);
            }
        }
    }
}

std::shared_ptr<Tensor> TopKMaskForward(const std::shared_ptr<Tensor> &input, int64_t topk) {
    CHECK_GE(input->Dims().size(), 1);
    const auto &dims = input->Dims();
    const int64_t num_experts = dims.back();
    CHECK_GT(num_experts, 0);
    CHECK_GT(topk, 0);
    CHECK_LE(topk, num_experts);
    const int64_t rows = input->NumElements() / num_experts;

    auto output = std::make_shared<Tensor>(dims, input->Dtype(), input->GetDevice());

    auto device = input->GetDevice();
    const auto &stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                             infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                             ->cuda_stream();
    const int threads = 256;
    const int blocks = static_cast<int>((rows + threads - 1) / threads);

    core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
        input->Dtype(),
        [=]<typename T>() {
            TopKMaskForwardKernel<T><<<blocks, threads, 0, stream>>>(
                static_cast<const T *>(input->DataPtr()), static_cast<T *>(output->DataPtr()), rows, num_experts, topk);
        },
        "CUDA TopKMaskForward");

    return output;
}

template <typename T>
__global__ void TopKMaskBackwardKernel(const T *__restrict__ grad_output, const T *__restrict__ mask_values,
                                       T *__restrict__ grad_input, int64_t total_elements) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) {
        return;
    }
    grad_input[idx] = static_cast<float>(mask_values[idx]) != 0.0f ? grad_output[idx] : T(0.0f);
}

std::shared_ptr<Tensor> TopKMaskBackward(const std::shared_ptr<Tensor> &grad_output,
                                         const std::shared_ptr<Tensor> &mask_values) {
    CHECK(grad_output->Dims() == mask_values->Dims());
    CHECK(grad_output->Dtype() == mask_values->Dtype());
    auto grad_input = std::make_shared<Tensor>(grad_output->Dims(), grad_output->Dtype(), grad_output->GetDevice());

    auto device = grad_output->GetDevice();
    const auto &stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                             infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                             ->cuda_stream();
    const int64_t total_elements = grad_output->NumElements();
    const int threads = 256;
    const int blocks = static_cast<int>((total_elements + threads - 1) / threads);

    core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
        grad_output->Dtype(),
        [=]<typename T>() {
            TopKMaskBackwardKernel<T><<<blocks, threads, 0, stream>>>(
                static_cast<const T *>(grad_output->DataPtr()), static_cast<const T *>(mask_values->DataPtr()),
                static_cast<T *>(grad_input->DataPtr()), total_elements);
        },
        "CUDA TopKMaskBackward");

    return grad_input;
}

} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_TOPK_MASK_KERNEL(kernel_name)                                                                    \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_TOPK_MASK_KERNEL(TopKMaskForward)
REGISTER_CUDA_TOPK_MASK_KERNEL(TopKMaskBackward)

#undef REGISTER_CUDA_TOPK_MASK_KERNEL
