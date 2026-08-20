#include <algorithm>
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
#include "infini_train/src/kernels/cuda/common/policy_helper.cuh"

#include "infini_train/src/core/runtime/cuda/cuda_dispatch.h"
#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

namespace infini_train::kernels::cuda {

namespace {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// Philox generates 128 bits = 4 × 32-bit floats per curand_uniform4 /
// curand_normal4 call.  UNROLL must equal 4.
// (See cuda_DistributionTemplates.h — const int UNROLL = 4)
constexpr int kUnroll = 4;
constexpr int kBlockSize = 256;

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

/**
 * UniformPhiloxKernel
 *
 * Fills data[0..numel) with values drawn from Uniform[from, to).
 *
 * Design (mirrors distribution_elementwise_grid_stride_kernel in cuda_DistributionTemplates.h):
 *
 *   - Each thread initialises its own curandStatePhilox4_32_10_t using the
 *     (seed, offset) pair supplied by the host.  The unique per-thread
 *     subsequence index is the thread's global linear index `idx`, so all
 *     threads generate non-overlapping random sequences.
 *
 *   - The grid-stride outer loop runs for `rounded_size / stride` iterations,
 *     where rounded_size is padded to a multiple of stride*kUnroll so every
 *     thread executes the same number of curand4 calls — a requirement for
 *     correct Philox offset accounting.
 *
 *   - curand_uniform4 returns 4 floats in (0, 1].  We map to [from, to) by
 *     scaling and flipping the boundary (matches PyTorch's uniform_kernel).
 */
template <typename T>
__global__ void UniformKernel(T *data, T from, T to, int64_t numel, uint64_t seed, uint64_t offset) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    // Initialise per-thread Philox state.
    // curand_init(seed, sequence=idx, offset, &state):
    //   - `sequence` selects an independent 2^67-element subsequence per thread.
    //   - `offset` skips the first `offset` values within that subsequence,
    //     allowing successive kernel launches to continue from where the
    //     previous one left off without overlap.
    curandStatePhilox4_32_10_t state;
    curand_init(seed, static_cast<uint64_t>(idx), offset, &state);

    const float range = to - from;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    // Pad to a multiple of stride*kUnroll so every thread runs the same number
    // of outer iterations (and thus the same number of curand4 calls).
    const int64_t rounded_size = ((numel - 1) / (stride * kUnroll) + 1) * stride * kUnroll;

    for (int64_t linear_index = idx; linear_index < rounded_size; linear_index += stride * kUnroll) {
        // curand_uniform4 advances the Philox counter by kUnroll=4 positions
        // and returns 4 floats in (0, 1].
        float4 rand = curand_uniform4(&state);

#pragma unroll
        for (int ii = 0; ii < kUnroll; ++ii) {
            const int64_t li = linear_index + stride * ii;
            if (li < numel) {
                const float r = (&rand.x)[ii];
                // Map (0, 1] → [from, to).  If the scaled value hits exactly
                // `to` (rare due to floating-point rounding), clamp to `from`
                // to preserve the half-open interval semantics.
                // (See cuda_DistributionTemplates.h uniform_kernel.)
                float val = r * range + from;
                val = (val == to) ? from : val;
                data[li] = val;
            }
        }
    }
}

/**
 * NormalPhiloxKernel
 *
 * Fills data[0..numel) with values drawn from N(mean, std_val).
 *
 * Uses curand_normal4, which applies the Box-Muller transform internally to
 * produce 4 standard-normal floats per call.  We then scale by std_val and
 * shift by mean.
 *
 * (See cuda_DistributionTemplates.h normal_and_transform /
 *  normal_kernel for the PyTorch equivalent.)
 */
template <typename T>
__global__ void NormalKernel(T *data, T mean, T std_val, int64_t numel, uint64_t seed, uint64_t offset) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, static_cast<uint64_t>(idx), offset, &state);

    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const int64_t rounded_size = ((numel - 1) / (stride * kUnroll) + 1) * stride * kUnroll;

    for (int64_t linear_index = idx; linear_index < rounded_size; linear_index += stride * kUnroll) {
        // curand_normal4 returns 4 standard-normal floats (mean=0, std=1).
        float4 rand = curand_normal4(&state);

#pragma unroll
        for (int ii = 0; ii < kUnroll; ++ii) {
            const int64_t li = linear_index + stride * ii;
            if (li < numel) {
                data[li] = (&rand.x)[ii] * std_val + mean;
            }
        }
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Host-side launcher implementations
// ---------------------------------------------------------------------------

void Uniform(std::shared_ptr<Tensor> tensor, float mean, float std_val, core::CUDAGeneratorImpl *impl) {

    CHECK(impl != nullptr) << "Generator must be a CUDAGeneratorImpl for CUDA kernels";
    CHECK(tensor != nullptr) << "Tensor must not be null";
    const int64_t numel = tensor->NumElements();
    if (numel == 0) {
        return;
    }

    auto launch = common::MakeElementwiseLaunch(numel, kBlockSize);

    uint64_t counter_offset
        = common::CalcPhiloxCounterOffset(numel, static_cast<int64_t>(launch.grid.x) * launch.block.x, kUnroll);

    const auto &cuda_stream
        = dynamic_cast<infini_train::core::cuda::CudaStream *>(
              infini_train::core::GetDeviceGuardImpl(tensor->GetDevice().type())->GetStream(tensor->GetDevice()))
              ->cuda_stream();
    // Acquire the generator lock only for the brief PhiloxEngineInputs call,
    // then release before launching the kernel.  The kernel runs purely on
    // local (seed, offset) values; no device-side synchronisation is needed.
    // (See "Note [Acquire lock when using random generators]" in PyTorch.)
    uint64_t seed = 0;
    uint64_t offset = 0;
    {
        std::lock_guard<std::mutex> lock(impl->mutex_);
        std::tie(seed, offset) = impl->PhiloxEngineInputs(counter_offset);
    }
    // core::cuda::DispatchCudaFunc<INFINI_FLOATING_TYPES>(
    //     tensor->Dtype(),
    //     [=]<typename T>() {
    //         const T casted_mean = mean.to<T>();
    //         const T casted_std  = std_val.to<T>();
    UniformKernel<float><<<launch.grid, launch.block, 0, cuda_stream>>>(static_cast<float *>(tensor->DataPtr()), mean,
                                                                        std_val, numel, seed, offset);
    //     }
    // );
}

void Normal(std::shared_ptr<Tensor> tensor, float mean, float std_val, core::CUDAGeneratorImpl *impl) {

    CHECK(impl != nullptr) << "Generator must be a CUDAGeneratorImpl for CUDA kernels";
    CHECK(tensor != nullptr) << "Tensor must not be null";
    const int64_t numel = tensor->NumElements();
    if (numel == 0) {
        return;
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
    // core::cuda::DispatchCudaFunc<INFINI_FLOATING_TYPES>(
    //     tensor->Dtype(),
    //     [=]<typename T>() {
    //         const T casted_mean = static_cast<T>(mean);
    //         const T casted_std  = static_cast<T>(std_val);
    NormalKernel<float><<<launch.grid, launch.block, 0, cuda_stream>>>(static_cast<float *>(tensor->DataPtr()), mean,
                                                                       std_val, numel, seed, offset);
    // }
    // );
}

} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_DISTRIBUTION_KERNEL(kernel_name)                                                                 \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_DISTRIBUTION_KERNEL(Uniform)
REGISTER_CUDA_DISTRIBUTION_KERNEL(Normal)

#undef REGISTER_CUDA_DISTRIBUTION_KERNEL