#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "glog/logging.h"

#include "infini_train/include/core/cuda_generator.h"
#include "infini_train/include/core/generator.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"
#include "infini_train/src/kernels/cuda/common/policy_helper.cuh"
namespace infini_train::kernels::cuda {

namespace {

constexpr int kUnroll = 4;
constexpr int kBlockSize = 256;

__global__ void DropoutKernel(float *data, int64_t numel, uint64_t seed, uint64_t offset, float p) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, static_cast<uint64_t>(idx), offset, &state);

    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const int64_t rounded_size = ((numel - 1) / (stride * kUnroll) + 1) * stride * kUnroll;

    const float keep_scale = 1.0f / (1.0f - p);

    for (int64_t linear_index = idx; linear_index < rounded_size; linear_index += stride * kUnroll) {
        float4 rand = curand_uniform4(&state);

#pragma unroll
        for (int ii = 0; ii < kUnroll; ++ii) {
            const int64_t li = linear_index + stride * ii;
            if (li < numel) {
                const float r = (&rand.x)[ii];
                data[li] = (r < p) ? 0.0f : data[li] * keep_scale;
            }
        }
    }
}

} // namespace

std::shared_ptr<Tensor> Dropout(std::shared_ptr<Tensor> tensor, float p, core::CUDAGeneratorImpl *impl) {
    auto device = tensor->GetDevice();
    core::DeviceGuard guard(device);

    CHECK(device.IsCUDA()) << "CUDA Dropout kernel requires a CUDA tensor";
    CHECK_EQ(static_cast<int>(tensor->Dtype()), static_cast<int>(DataType::kFLOAT32))
        << "CUDA Dropout currently only supports FLOAT32 tensors";

    const int64_t numel = tensor->NumElements();
    if (numel == 0) {
        return tensor;
    }

    auto launch = common::MakeElementwiseLaunch(numel, kBlockSize);

    uint64_t counter_offset
        = common::CalcPhiloxCounterOffset(numel, static_cast<int64_t>(launch.grid.x) * launch.block.x, kUnroll);

    const auto &cuda_stream
        = dynamic_cast<infini_train::core::cuda::CudaStream *>(
              infini_train::core::GetDeviceGuardImpl(tensor->GetDevice().type())->GetStream(tensor->GetDevice()))
              ->cuda_stream();

    uint64_t seed = 0;
    uint64_t offset = 0;
    {
        std::lock_guard<std::mutex> lock(impl->mutex_);
        std::tie(seed, offset) = impl->PhiloxEngineInputs(counter_offset);
    }

    DropoutKernel<<<launch.grid.x, launch.block.x, 0, cuda_stream>>>(static_cast<float *>(tensor->DataPtr()), numel,
                                                                     seed, offset, p);

    return tensor;
}

} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_DROPOUT_KERNEL(kernel_name)                                                                      \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_DROPOUT_KERNEL(Dropout)

#undef REGISTER_CUDA_DROPOUT_KERNEL