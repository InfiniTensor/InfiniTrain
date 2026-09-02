#include <cmath>
#include <limits>
#include <numeric>

#include <hipcub/block/block_reduce.hpp>
#include <hip/hip_runtime.h>

#include "infini_train/include/common/dcu/common_dcu.h"
#include "infini_train/include/common/dcu/cub_compat.cuh"
#include "infini_train/include/common/dcu/kernel_helper.cuh"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/dcu/dcu_dispatch.h"
#include "infini_train/src/core/runtime/dcu/dcu_runtime_common.h"

namespace infini_train::kernels::dcu {
namespace {
constexpr float kNegativeInfinity = -std::numeric_limits<float>::infinity();
}

template <size_t BLOCK_SIZE, typename TargetType, typename InputType>
__global__ void CrossEntropyForwardKernel(const InputType *__restrict__ input_ptr,
                                          const TargetType *__restrict__ target_ptr, float *__restrict__ loss_ptr,
                                          int bs, int num_classes) {
    __shared__ struct {
        float max_logit;
        float sum_exp;
        TargetType target_class;
        typename hipcub::BlockReduce<float, BLOCK_SIZE>::TempStorage reduce;
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
        thread_max = fmaxf(thread_max, common::dcu::Cast<float>(input_ptr[base + i]));
    }
    const float block_max = hipcub::BlockReduce<float, BLOCK_SIZE>(shared.reduce).Reduce(thread_max, CubMaxOp());
    if (tid == 0) {
        shared.max_logit = block_max;
    }
    __syncthreads();

    // calculate the sum of exponents
    float thread_sum = 0.0f;
    for (int i = tid; i < num_classes; i += BLOCK_SIZE) {
        thread_sum += expf(common::dcu::Cast<float>(input_ptr[base + i]) - shared.max_logit);
    }
    const float block_sum = hipcub::BlockReduce<float, BLOCK_SIZE>(shared.reduce).Sum(thread_sum);
    if (tid == 0) {
        shared.sum_exp = block_sum;
    }
    __syncthreads();

    // calculate the loss
    if (tid == 0) {
        const float target_val
            = common::dcu::Cast<float>(input_ptr[base + common::dcu::Cast<size_t>(shared.target_class)])
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

    // Keep logits in their input dtype while accumulating and returning loss in FP32.
    auto batched_output
        = std::make_shared<Tensor>(std::vector<int64_t>{bs}, DataType::kFLOAT32, input->GetDevice());

    constexpr int threads_per_block = 256;
    int num_blocks = bs;

    auto device = target->GetDevice();
    const auto &hip_stream = dynamic_cast<infini_train::core::dcu::DcuStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->hip_stream();

    return core::dcu::DispatchDcuFunc<DataTypeList<DataType::kUINT8, DataType::kINT64>,
                                        DataTypeList<INFINI_ALL_FLOATING_TYPES>>(
        {target->Dtype(), input->Dtype()},
        [=]<typename Ttarget, typename Tinput>() {
            const Ttarget *target_ptr = static_cast<const Ttarget *>(target->DataPtr());
            const Tinput *input_ptr = static_cast<const Tinput *>(input->DataPtr());
            float *batched_loss_ptr = static_cast<float *>(batched_output->DataPtr());
            // FIXME(dcj): do reduce on GPU
            CrossEntropyForwardKernel<threads_per_block, Ttarget, Tinput>
                <<<num_blocks, threads_per_block, 0, hip_stream>>>(input_ptr, target_ptr, batched_loss_ptr, bs,
                                                                    num_classes);

            auto loss_cpu = batched_output->To(Device());
            auto loss = std::make_shared<Tensor>(std::vector<int64_t>{}, DataType::kFLOAT32, Device());
            auto loss_cpu_typed_ptr = static_cast<const float *>(loss_cpu.DataPtr());
            static_cast<float *>(loss->DataPtr())[0]
                = std::accumulate(loss_cpu_typed_ptr, loss_cpu_typed_ptr + bs, 0.0f) / bs;

            return std::make_shared<Tensor>(loss->To(input->GetDevice()));
        },
        "HIP CrossEntropyForward");
}

template <size_t BLOCK_SIZE, typename TargetType, typename InputType, typename OutputGradType>
__global__ void CrossEntropyBackwardKernel(const InputType *__restrict__ input_ptr,
                                           InputType *__restrict__ input_grad_ptr,
                                           const TargetType *__restrict__ target_ptr,
                                           const OutputGradType *__restrict__ output_grad_ptr, int bs,
                                           int num_classes) {
    __shared__ struct {
        float max_logit;
        float sum_exp;
        int target_class;
        typename hipcub::BlockReduce<float, BLOCK_SIZE>::TempStorage reduce;
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
        thread_max = fmaxf(thread_max, common::dcu::Cast<float>(input_ptr[idx_base + i]));
    }
    const float block_max = hipcub::BlockReduce<float, BLOCK_SIZE>(shared.reduce).Reduce(thread_max, CubMaxOp());
    if (tid == 0) {
        shared.max_logit = block_max;
    }
    __syncthreads();

    // calculate the sum
    float thread_sum = 0.0f;
    for (int i = tid; i < num_classes; i += BLOCK_SIZE) {
        thread_sum += expf(common::dcu::Cast<float>(input_ptr[idx_base + i]) - shared.max_logit);
    }

    const float block_sum = hipcub::BlockReduce<float, BLOCK_SIZE>(shared.reduce).Sum(thread_sum);
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
        const float exp_val = expf(common::dcu::Cast<float>(input_ptr[global_idx]) - shared.max_logit);
        input_grad_ptr[global_idx] = common::dcu::Cast<InputType>((exp_val * scale - (i == target)) * inv_bs
                                                                   * common::dcu::Cast<float>(output_grad_ptr[0]));
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
    auto grad_input = std::make_shared<Tensor>(input_dims, input->Dtype(), input->GetDevice());

    constexpr int threads_per_block = 256;
    int num_blocks = bs;

    auto device = target->GetDevice();
    const auto &hip_stream = dynamic_cast<infini_train::core::dcu::DcuStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->hip_stream();

    core::dcu::DispatchDcuFunc<DataTypeList<DataType::kUINT8, DataType::kINT64>,
                                DataTypeList<INFINI_ALL_FLOATING_TYPES>,
                                DataTypeList<INFINI_ALL_FLOATING_TYPES>>(
        {target->Dtype(), input->Dtype(), grad_output->Dtype()},
        [=]<typename Ttarget, typename Tinput, typename Toutput_grad>() {
            const Toutput_grad *output_grad_ptr = static_cast<const Toutput_grad *>(grad_output->DataPtr());
            const Ttarget *target_ptr = static_cast<const Ttarget *>(target->DataPtr());
            const Tinput *input_ptr = static_cast<const Tinput *>(input->DataPtr());
            Tinput *input_grad_ptr = static_cast<Tinput *>(grad_input->DataPtr());

            CrossEntropyBackwardKernel<threads_per_block, Ttarget, Tinput, Toutput_grad>
                <<<num_blocks, threads_per_block, 0, hip_stream>>>(input_ptr, input_grad_ptr, target_ptr,
                                                                    output_grad_ptr, bs, num_classes);
        },
        "HIP CrossEntropyBackward");

    return {grad_input};
}
} // namespace infini_train::kernels::dcu

#define REGISTER_HIP_CROSS_ENTROPY_KERNEL(kernel_name)                                                                \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kDCU, kernel_name, infini_train::kernels::dcu::kernel_name)

REGISTER_HIP_CROSS_ENTROPY_KERNEL(CrossEntropyForward)
REGISTER_HIP_CROSS_ENTROPY_KERNEL(CrossEntropyBackward)

#undef REGISTER_HIP_CROSS_ENTROPY_KERNEL
