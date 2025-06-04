#include <cmath>
#include <cstdint>
#include <cub/block/block_reduce.cuh>
#include <cuda_runtime.h>
#include <limits>
#include <numeric>

#include "glog/logging.h"

#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {
namespace {
constexpr float kNegativeInfinity = -std::numeric_limits<float>::infinity();
}

template <size_t BLOCK_SIZE, typename TargetType>
__global__ void CrossEntropyForwardKernel(const float *__restrict__ input_ptr,
                                          const TargetType *__restrict__ target_ptr, float *__restrict__ loss_ptr,
                                          int bs, int num_classes) {
    __shared__ struct {
        float max_logit;
        float sum_exp;
        TargetType target_class;
        typename cub::BlockReduce<float, BLOCK_SIZE>::TempStorage reduce;
    } shared;

    const int sample_idx = blockIdx.x;
    if (sample_idx >= bs) {
        return;
    }

    const int tid = threadIdx.x;
    const size_t base = sample_idx * num_classes;

    if (tid == 0) {
        shared.target_class = target_ptr[sample_idx];
    }
    __syncthreads();

    // calculate the max
    float thread_max = kNegativeInfinity;
    for (int i = tid; i < num_classes; i += BLOCK_SIZE) { thread_max = fmaxf(thread_max, input_ptr[base + i]); }
    shared.max_logit = cub::BlockReduce<float, BLOCK_SIZE>(shared.reduce).Reduce(thread_max, cub::Max());
    __syncthreads();

    // calculate the sum of exponents
    float thread_sum = 0.0f;
    for (int i = tid; i < num_classes; i += BLOCK_SIZE) { thread_sum += expf(input_ptr[base + i] - shared.max_logit); }
    shared.sum_exp = cub::BlockReduce<float, BLOCK_SIZE>(shared.reduce).Sum(thread_sum);
    __syncthreads();

    // calculate the loss
    if (tid == 0) {
        const float target_val = input_ptr[base + shared.target_class] - shared.max_logit;
        loss_ptr[sample_idx] = logf(shared.sum_exp) - target_val;
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

    constexpr int threads_per_block = 256;
    int num_blocks = bs;

    const auto *cuda_device = dynamic_cast<const CudaDevice *>(target->GetDevice());
    // TODO(dcj): support multi datatypes later
    switch (target->Dtype()) {
    case DataType::kUINT8: {
        const uint8_t *target_ptr = static_cast<const uint8_t *>(target->DataPtr());
        // FIXME(dcj): do reduce on GPU

        CrossEntropyForwardKernel<threads_per_block, uint8_t>
            <<<num_blocks, threads_per_block, 0, cuda_device->Stream()>>>(input_ptr, target_ptr, batched_loss_ptr, bs,
                                                                          num_classes);
        break;
    }
    case DataType::kINT64: {
        const int64_t *target_ptr = static_cast<const int64_t *>(target->DataPtr());
        // FIXME(dcj): do reduce on GPU
        CrossEntropyForwardKernel<threads_per_block, int64_t>
            <<<num_blocks, threads_per_block, 0, cuda_device->Stream()>>>(input_ptr, target_ptr, batched_loss_ptr, bs,
                                                                          num_classes);
        break;
    }
    default:
        LOG(FATAL) << "Unsupported target data type: " << static_cast<int>(target->Dtype());
    }
    cudaDeviceSynchronize();

    auto loss_cpu = batched_output->To(DeviceManager::Instance()->GetDevice(DeviceType::kCPU, 0));
    auto loss = std::make_shared<Tensor>(std::vector<int64_t>{}, DataType::kFLOAT32,
                                         DeviceManager::Instance()->GetDevice(DeviceType::kCPU, 0));
    static_cast<float *>(loss->DataPtr())[0]
        = std::accumulate(static_cast<const float *>(loss_cpu.DataPtr()),
                          static_cast<const float *>(loss_cpu.DataPtr()) + bs, 0.0f)
        / bs;

    return {std::make_shared<Tensor>(loss->To(input->GetDevice()))};
}

template <typename TargetType, size_t BLOCK_SIZE>
__global__ void CrossEntropyBackwardKernel(const float *__restrict__ input_ptr, float *__restrict__ input_grad_ptr,
                                           const TargetType *__restrict__ target_ptr,
                                           const float *__restrict__ output_grad_ptr, int bs, int num_classes) {
    __shared__ struct {
        float max_logit;
        float sum_exp;
        int target_class;
        typename cub::BlockReduce<float, BLOCK_SIZE>::TempStorage reduce;
    } shared;

    const int tid = threadIdx.x;
    const int idx = blockIdx.x;

    if (idx >= bs) {
        return;
    }

    const size_t idx_base = idx * num_classes;

    if (tid == 0) {
        shared.target_class = static_cast<int>(target_ptr[idx]);
    }
    __syncthreads();

    // calculate the max
    float thread_max = kNegativeInfinity;
    for (int i = tid; i < num_classes; i += BLOCK_SIZE) { thread_max = fmaxf(thread_max, input_ptr[idx_base + i]); }
    shared.max_logit = cub::BlockReduce<float, BLOCK_SIZE>(shared.reduce).Reduce(thread_max, cub::Max());
    __syncthreads();

    // calculate the sum
    float thread_sum = 0.0f;
    for (int i = tid; i < num_classes; i += BLOCK_SIZE) {
        thread_sum += expf(input_ptr[idx_base + i] - shared.max_logit);
    }
    shared.sum_exp = cub::BlockReduce<float, BLOCK_SIZE>(shared.reduce).Sum(thread_sum);
    __syncthreads();

    // calculate the gradient
    const float inv_bs = 1.0f / bs;
    const float scale = 1.0f / shared.sum_exp;
    const int target = shared.target_class;

    for (int i = tid; i < num_classes; i += BLOCK_SIZE) {
        const int global_idx = idx_base + i;
        const float exp_val = expf(input_ptr[global_idx] - shared.max_logit);
        input_grad_ptr[global_idx] = (exp_val * scale - (i == target)) * inv_bs * output_grad_ptr[0];
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
    const float *output_grad_ptr = static_cast<const float *>(grad_output->DataPtr());
    const float *input_ptr = static_cast<const float *>(input->DataPtr());
    float *input_grad_ptr = static_cast<float *>(grad_input->DataPtr());

    constexpr int threads_per_block = 256;
    int num_blocks = bs;

    const auto *cuda_device = dynamic_cast<const CudaDevice *>(target->GetDevice());
    // TODO(dcj): support multi datatypes later
    switch (target->Dtype()) {
    case DataType::kUINT8: {
        const uint8_t *target_ptr = static_cast<const uint8_t *>(target->DataPtr());
        CrossEntropyBackwardKernel<uint8_t, threads_per_block>
            <<<num_blocks, threads_per_block, 0, cuda_device->Stream()>>>(input_ptr, input_grad_ptr, target_ptr,
                                                                          output_grad_ptr, bs, num_classes);
        break;
    }
    case DataType::kINT64: {
        const int64_t *target_ptr = static_cast<const int64_t *>(target->DataPtr());
        CrossEntropyBackwardKernel<int64_t, threads_per_block>
            <<<num_blocks, threads_per_block, 0, cuda_device->Stream()>>>(input_ptr, input_grad_ptr, target_ptr,
                                                                          output_grad_ptr, bs, num_classes);
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
