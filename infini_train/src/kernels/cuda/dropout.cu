#include <cstdint>
#include <mutex>
#include <optional>
#include <type_traits>

#include <curand_kernel.h>

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/common/cuda/kernel_helper.cuh"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/core/runtime/dropout_stubs.h"
#include "infini_train/include/generator.h"
#include "infini_train/include/tensor.h"
#include "infini_train/src/core/runtime/cuda/cuda_dispatch.h"
#include "infini_train/src/core/runtime/cuda/cuda_generator_impl.h"
#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

namespace infini_train::kernels::cuda {
namespace {

constexpr int kThreadsPerBlock = 256;

template <typename random_t> __device__ random_t uniform_sample(curandStatePhilox4_32_10_t *state) {
    if constexpr (std::is_same_v<random_t, double>) {
        return 1.0 - curand_uniform_double(state);
    } else {
        return static_cast<float>(curand(state)) * 0x1p-32f;
    }
}

template <typename storage_t, typename random_t>
__global__ void DropoutForwardKernel(storage_t *output, uint8_t *mask, const storage_t *input, int64_t n,
                                     random_t p, uint64_t seed, uint64_t subsequence) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }
    if (p == static_cast<random_t>(0)) {
        mask[index] = 1;
        output[index] = input[index];
        return;
    }
    if (p == static_cast<random_t>(1)) {
        mask[index] = 0;
        output[index] = common::cuda::Cast<storage_t>(0.0f);
        return;
    }

    curandStatePhilox4_32_10_t state;
    curand_init(seed, subsequence + static_cast<uint64_t>(index), 0, &state);
    const bool keep = uniform_sample<random_t>(&state) >= p;
    const random_t scale = static_cast<random_t>(1) / (static_cast<random_t>(1) - p);
    mask[index] = keep ? 1 : 0;
    output[index] = keep
        ? common::cuda::Cast<storage_t>(common::cuda::Cast<random_t>(input[index]) * scale)
        : common::cuda::Cast<storage_t>(0.0f);
}

template <typename storage_t, typename random_t>
__global__ void DropoutBackwardKernel(storage_t *grad_input, const storage_t *grad_output, const uint8_t *mask,
                                      int64_t n, random_t p) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }
    const random_t scale = p == static_cast<random_t>(1)
        ? static_cast<random_t>(0)
        : static_cast<random_t>(1) / (static_cast<random_t>(1) - p);
    grad_input[index] = mask[index]
        ? common::cuda::Cast<storage_t>(common::cuda::Cast<random_t>(grad_output[index]) * scale)
        : common::cuda::Cast<storage_t>(0.0f);
}

const core::cuda::CudaStream *get_cuda_stream(const Device &device) {
    return dynamic_cast<const core::cuda::CudaStream *>(
        core::GetDeviceGuardImpl(device.type())->GetStream(device));
}

void dropout_forward_cuda(Tensor &output, Tensor &mask, const Tensor &input, double p,
                          const std::optional<Generator> &generator) {
    const Device device = input.GetDevice();
    CHECK(device.IsCUDA());
    const int64_t n = input.NumElements();
    if (n == 0) {
        return;
    }
    core::DeviceGuard guard(device);

    uint64_t seed = 0;
    uint64_t subsequence = 0;
    if (p > 0.0 && p < 1.0) {
        auto *cuda_generator = get_generator_or_default<core::cuda::CUDAGeneratorImpl>(
            generator, core::cuda::getDefaultCUDAGenerator(device.index()));
        std::lock_guard<std::mutex> lock(cuda_generator->mutex_);
        seed = cuda_generator->current_seed();
        subsequence = cuda_generator->philox_subsequence(static_cast<uint64_t>(n));
    }

    const int blocks = static_cast<int>((n + kThreadsPerBlock - 1) / kThreadsPerBlock);
    const auto *stream = get_cuda_stream(device);
    core::cuda::DispatchCudaFunc<DataType::kFLOAT16, DataType::kBFLOAT16, DataType::kFLOAT32, DataType::kFLOAT64>(
        input.Dtype(),
        [&]<typename storage_t>() {
            using random_t = std::conditional_t<std::is_same_v<storage_t, double>, double, float>;
            DropoutForwardKernel<storage_t, random_t><<<blocks, kThreadsPerBlock, 0, stream->cuda_stream()>>>(
                static_cast<storage_t *>(output.DataPtr()), static_cast<uint8_t *>(mask.DataPtr()),
                static_cast<const storage_t *>(input.DataPtr()), n, static_cast<random_t>(p), seed, subsequence);
        },
        "CUDA dropout forward");
    CUDA_CHECK(cudaGetLastError());
}

void dropout_backward_cuda(Tensor &grad_input, const Tensor &grad_output, const Tensor &mask, double p) {
    const Device device = grad_output.GetDevice();
    CHECK(device.IsCUDA());
    const int64_t n = grad_output.NumElements();
    if (n == 0) {
        return;
    }
    core::DeviceGuard guard(device);

    const int blocks = static_cast<int>((n + kThreadsPerBlock - 1) / kThreadsPerBlock);
    const auto *stream = get_cuda_stream(device);
    core::cuda::DispatchCudaFunc<DataType::kFLOAT16, DataType::kBFLOAT16, DataType::kFLOAT32, DataType::kFLOAT64>(
        grad_output.Dtype(),
        [&]<typename storage_t>() {
            using random_t = std::conditional_t<std::is_same_v<storage_t, double>, double, float>;
            DropoutBackwardKernel<storage_t, random_t><<<blocks, kThreadsPerBlock, 0, stream->cuda_stream()>>>(
                static_cast<storage_t *>(grad_input.DataPtr()), static_cast<const storage_t *>(grad_output.DataPtr()),
                static_cast<const uint8_t *>(mask.DataPtr()), n, static_cast<random_t>(p));
        },
        "CUDA dropout backward");
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

using infini_train::dropout_backward_stub;
using infini_train::dropout_forward_stub;

REGISTER_DISPATCH(dropout_forward_stub, Device::DeviceType::kCUDA, &dropout_forward_cuda);
REGISTER_DISPATCH(dropout_backward_stub, Device::DeviceType::kCUDA, &dropout_backward_cuda);

} // namespace infini_train::kernels::cuda
