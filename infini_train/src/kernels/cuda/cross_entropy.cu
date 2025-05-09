#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {
namespace {
constexpr float kNegativeInfinity = -std::numeric_limits<float>::infinity();
}

template <typename TargetType>
__global__ void CrossEntropyForwardKernel(const float *input_ptr, const TargetType *target_ptr, float *loss_ptr, int bs,
                                          int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < bs) {
        float max_logit = kNegativeInfinity;
        for (int j = 0; j < num_classes; ++j) { max_logit = max(max_logit, input_ptr[idx * num_classes + j]); }
        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; ++j) { sum_exp += expf(input_ptr[idx * num_classes + j] - max_logit); }
        loss_ptr[idx] = -logf(expf(input_ptr[idx * num_classes + target_ptr[idx]] - max_logit) / sum_exp);
    }
}

std::shared_ptr<Tensor> CrossEntropyForward(const std::shared_ptr<Tensor> &input,
                                            const std::shared_ptr<Tensor> &target) {
    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int num_classes = *input_dims.rbegin();

    auto batched_output = std::make_shared<Tensor>(std::vector<int64_t>{bs}, DataType::kFLOAT32, input->GetDevice());
    const float *input_ptr = static_cast<const float *>(input->DataPtr());
    float *batched_loss_ptr = static_cast<float *>(batched_output->DataPtr());

    int threads_per_block = 256;
    int num_blocks = (bs + threads_per_block - 1) / threads_per_block;

    // TODO(dcj): support multi datatypes later
    switch (target->Dtype()) {
    case DataType::kUINT8: {
        const uint8_t *target_ptr = static_cast<const uint8_t *>(target->DataPtr());
        // FIXME(dcj): do reduce on GPU
        CrossEntropyForwardKernel<uint8_t>
            <<<num_blocks, threads_per_block>>>(input_ptr, target_ptr, batched_loss_ptr, bs, num_classes);
        break;
    }
    case DataType::kINT64: {
        const int64_t *target_ptr = static_cast<const int64_t *>(target->DataPtr());
        // FIXME(dcj): do reduce on GPU
        CrossEntropyForwardKernel<int64_t>
            <<<num_blocks, threads_per_block>>>(input_ptr, target_ptr, batched_loss_ptr, bs, num_classes);
        break;
    }
    default:
        LOG(FATAL) << "Unsupported target data type: " << static_cast<int>(target->Dtype());
    }
    cudaDeviceSynchronize();

    auto loss_cpu = batched_output->To(Device());
    auto loss = std::make_shared<Tensor>(std::vector<int64_t>{}, DataType::kFLOAT32, Device());
    static_cast<float *>(loss->DataPtr())[0]
        = std::accumulate(static_cast<const float *>(loss_cpu.DataPtr()),
                          static_cast<const float *>(loss_cpu.DataPtr()) + bs, 0.0f)
        / bs;

    return {std::make_shared<Tensor>(loss->To(input->GetDevice()))};
}

template <typename TargetType>
__global__ void CrossEntropyBackwardKernel(const float *input_ptr, float *input_grad_ptr, const TargetType *target_ptr,
                                           int bs, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < bs) {
        float max_logit = kNegativeInfinity;
        for (int j = 0; j < num_classes; ++j) { max_logit = max(max_logit, input_ptr[idx * num_classes + j]); }
        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; ++j) { sum_exp += expf(input_ptr[idx * num_classes + j] - max_logit); }
        for (int j = 0; j < num_classes; ++j) {
            int idx_grad = idx * num_classes + j;
            input_grad_ptr[idx_grad]
                = (expf(input_ptr[idx_grad] - max_logit) / sum_exp - (j == target_ptr[idx] ? 1.0f : 0.0f)) / bs;
        }
    }
}

std::shared_ptr<Tensor> CrossEntropyBackward(const std::shared_ptr<Tensor> &input,
                                             const std::shared_ptr<Tensor> &target,
                                             const std::shared_ptr<Tensor> &grad_output) {
    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int num_classes = *input_dims.rbegin();

    CHECK_EQ(grad_output->Dims().size(), 0);
    auto grad_input = std::make_shared<Tensor>(input->Dims(), DataType::kFLOAT32, grad_output->GetDevice());
    grad_input->Fill<float>(0.0f);
    const float *input_ptr = static_cast<const float *>(input->DataPtr());
    float *input_grad_ptr = static_cast<float *>(grad_input->DataPtr());

    int threads_per_block = 256;
    int num_blocks = (bs + threads_per_block - 1) / threads_per_block;

    // TODO(dcj): support multi datatypes later
    switch (target->Dtype()) {
    case DataType::kUINT8: {
        const uint8_t *target_ptr = static_cast<const uint8_t *>(target->DataPtr());
        CrossEntropyBackwardKernel<uint8_t>
            <<<num_blocks, threads_per_block>>>(input_ptr, input_grad_ptr, target_ptr, bs, num_classes);
        break;
    }
    case DataType::kINT64: {
        const int64_t *target_ptr = static_cast<const int64_t *>(target->DataPtr());
        CrossEntropyBackwardKernel<int64_t>
            <<<num_blocks, threads_per_block>>>(input_ptr, input_grad_ptr, target_ptr, bs, num_classes);
        break;
    }
    default:
        LOG(FATAL) << "Unsupported target data type: " << static_cast<int>(target->Dtype());
    }

    return {grad_input};
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_CROSS_ENTROPY_KERNEL(kernel_name)                                                                \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_CROSS_ENTROPY_KERNEL(CrossEntropyForward)
REGISTER_CUDA_CROSS_ENTROPY_KERNEL(CrossEntropyBackward)

#undef REGISTER_CUDA_CROSS_ENTROPY_KERNEL
