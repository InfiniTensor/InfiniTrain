#include <cmath>
#include <limits>
#include <numeric>

#include <cub/block/block_reduce.cuh>
#include <cuda_runtime.h>

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/common/cuda/kernel_helper.cuh"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {
namespace {
constexpr float kNegativeInfinity = -std::numeric_limits<float>::infinity();
}

template <size_t BLOCK_SIZE, typename TargetType, typename InputType>
__global__ void CrossEntropyForwardKernel(const InputType *__restrict__ input_ptr,
                                          const TargetType *__restrict__ target_ptr, InputType *__restrict__ loss_ptr,
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
    for (int i = tid; i < num_classes; i += BLOCK_SIZE) {
        thread_max = fmaxf(thread_max, common::cuda::Cast<float>(input_ptr[base + i]));
    }
    const float block_max = cub::BlockReduce<float, BLOCK_SIZE>(shared.reduce).Reduce(thread_max, cub::Max());
    if (tid == 0) {
        shared.max_logit = block_max;
    }
    __syncthreads();

    // calculate the sum of exponents
    float thread_sum = 0.0f;
    for (int i = tid; i < num_classes; i += BLOCK_SIZE) {
        thread_sum += expf(common::cuda::Cast<float>(input_ptr[base + i]) - shared.max_logit);
    }
    const float block_sum = cub::BlockReduce<float, BLOCK_SIZE>(shared.reduce).Sum(thread_sum);
    if (tid == 0) {
        shared.sum_exp = block_sum;
    }
    __syncthreads();

    // calculate the loss
    if (tid == 0) {
        const float target_val
            = common::cuda::Cast<float>(input_ptr[base + common::cuda::Cast<size_t>(shared.target_class)])
            - shared.max_logit;
        loss_ptr[sample_idx] = logf(shared.sum_exp) - target_val;
    }
}

std::shared_ptr<Tensor> CrossEntropyForward(const std::shared_ptr<Tensor> &input,
                                            const std::shared_ptr<Tensor> &target) {
    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int num_classes = *input_dims.rbegin();

    auto batched_output = std::make_shared<Tensor>(std::vector<int64_t>{bs}, input->Dtype(), input->GetDevice());

    constexpr int threads_per_block = 256;
    int num_blocks = bs;

    const auto *cuda_device = dynamic_cast<const CudaDevice *>(target->GetDevice());
    return DispatchFunc<DataTypeList<DataType::kUINT8, DataType::kINT64>, DataTypeList<INFINI_ALL_FLOATING_TYPES>>(
        {target->Dtype(), input->Dtype()},
        [=]<typename Ttarget, typename Tinput>() {
            const Ttarget *target_ptr = static_cast<const Ttarget *>(target->DataPtr());
            const Tinput *input_ptr = static_cast<const Tinput *>(input->DataPtr());
            Tinput *batched_loss_ptr = static_cast<Tinput *>(batched_output->DataPtr());
            // FIXME(dcj): do reduce on GPU
            CrossEntropyForwardKernel<threads_per_block, Ttarget, Tinput>
                <<<num_blocks, threads_per_block, 0, cuda_device->Stream()>>>(input_ptr, target_ptr, batched_loss_ptr,
                                                                              bs, num_classes);

            auto loss_cpu = batched_output->To(DeviceManager::Instance()->GetDefaultDevice());
            auto loss = std::make_shared<Tensor>(std::vector<int64_t>{}, input->Dtype(),
                                                 DeviceManager::Instance()->GetDefaultDevice());
            auto loss_cpu_typed_ptr = static_cast<const Tinput *>(loss_cpu.DataPtr());
            static_cast<Tinput *>(loss->DataPtr())[0]
                = std::accumulate(loss_cpu_typed_ptr, loss_cpu_typed_ptr + bs, 0.0f,
                                  [](float acc, const Tinput &val) { return acc + common::cuda::Cast<float>(val); })
                / bs;

            return std::make_shared<Tensor>(loss->To(input->GetDevice()));
        },
        "CUDA CrossEntropyForward");
}

template <size_t BLOCK_SIZE, typename TargetType, typename InputType>
__global__ void CrossEntropyBackwardKernel(const InputType *__restrict__ input_ptr,
                                           InputType *__restrict__ input_grad_ptr,
                                           const TargetType *__restrict__ target_ptr,
                                           const InputType *__restrict__ output_grad_ptr, int bs, int num_classes) {
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
    for (int i = tid; i < num_classes; i += BLOCK_SIZE) {
        thread_max = fmaxf(thread_max, common::cuda::Cast<float>(input_ptr[idx_base + i]));
    }
    const float block_max = cub::BlockReduce<float, BLOCK_SIZE>(shared.reduce).Reduce(thread_max, cub::Max());
    if (tid == 0) {
        shared.max_logit = block_max;
    }
    __syncthreads();

    // calculate the sum
    float thread_sum = 0.0f;
    for (int i = tid; i < num_classes; i += BLOCK_SIZE) {
        thread_sum += expf(common::cuda::Cast<float>(input_ptr[idx_base + i]) - shared.max_logit);
    }

    const float block_sum = cub::BlockReduce<float, BLOCK_SIZE>(shared.reduce).Sum(thread_sum);
    if (tid == 0) {
        shared.sum_exp = block_sum;
    }
    __syncthreads();

    // calculate the gradient
    const float inv_bs = 1.0f / bs;
    const float scale = 1.0f / shared.sum_exp;
    const int target = shared.target_class;

    for (int i = tid; i < num_classes; i += BLOCK_SIZE) {
        const int global_idx = idx_base + i;
        const float exp_val = expf(common::cuda::Cast<float>(input_ptr[global_idx]) - shared.max_logit);
        input_grad_ptr[global_idx] = common::cuda::Cast<InputType>((exp_val * scale - (i == target)) * inv_bs
                                                                   * common::cuda::Cast<float>(output_grad_ptr[0]));
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
    auto grad_input = std::make_shared<Tensor>(input->Dims(), input->Dtype(), grad_output->GetDevice());

    constexpr int threads_per_block = 256;
    int num_blocks = bs;

    const auto *cuda_device = dynamic_cast<const CudaDevice *>(target->GetDevice());
    DispatchFunc<DataTypeList<DataType::kUINT8, DataType::kINT64>, DataTypeList<INFINI_ALL_FLOATING_TYPES>>(
        {target->Dtype(), input->Dtype()},
        [=]<typename Ttarget, typename Tinput>() {
            grad_input->Fill<Tinput>(0);
            const Tinput *output_grad_ptr = static_cast<const Tinput *>(grad_output->DataPtr());
            const Ttarget *target_ptr = static_cast<const Ttarget *>(target->DataPtr());
            const Tinput *input_ptr = static_cast<const Tinput *>(input->DataPtr());
            Tinput *input_grad_ptr = static_cast<Tinput *>(grad_input->DataPtr());
            CrossEntropyBackwardKernel<threads_per_block, Ttarget, Tinput>
                <<<num_blocks, threads_per_block, 0, cuda_device->Stream()>>>(input_ptr, input_grad_ptr, target_ptr,
                                                                              output_grad_ptr, bs, num_classes);
        },
        "CUDA CrossEntropyBackward");

    return {grad_input};
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_CROSS_ENTROPY_KERNEL(kernel_name)                                                                \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_CROSS_ENTROPY_KERNEL(CrossEntropyForward)
REGISTER_CUDA_CROSS_ENTROPY_KERNEL(CrossEntropyBackward)

#undef REGISTER_CUDA_CROSS_ENTROPY_KERNEL
