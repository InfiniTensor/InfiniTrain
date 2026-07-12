#include <cstdint>
#include <mutex>
#include <optional>

#include <curand_kernel.h>

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/common/cuda/kernel_helper.cuh"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/core/runtime/distribution_stubs.h"
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

template <typename random_t> __device__ random_t normal_sample(curandStatePhilox4_32_10_t *state) {
    if constexpr (std::is_same_v<random_t, double>) {
        return curand_normal_double(state);
    } else {
        return curand_normal(state);
    }
}

template <typename storage_t, typename random_t>
__global__ void UniformKernel(storage_t *data, int64_t n, random_t from, random_t to,
                              uint64_t seed, uint64_t subsequence) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }

    curandStatePhilox4_32_10_t state;
    curand_init(seed, subsequence + static_cast<uint64_t>(index), 0, &state);
    const storage_t from_value = common::cuda::Cast<storage_t>(from);
    const storage_t to_value = common::cuda::Cast<storage_t>(to);
    const storage_t value = common::cuda::Cast<storage_t>(from + uniform_sample<random_t>(&state) * (to - from));
    data[index] = common::cuda::Cast<random_t>(value) == common::cuda::Cast<random_t>(to_value) ? from_value : value;
}

template <typename storage_t, typename random_t>
__global__ void NormalKernel(storage_t *data, int64_t n, random_t mean, random_t std,
                             uint64_t seed, uint64_t subsequence) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }

    curandStatePhilox4_32_10_t state;
    curand_init(seed, subsequence + static_cast<uint64_t>(index), 0, &state);
    data[index] = common::cuda::Cast<storage_t>(mean + normal_sample<random_t>(&state) * std);
}

const core::cuda::CudaStream *get_cuda_stream(const Device &device) {
    return dynamic_cast<const core::cuda::CudaStream *>(
        core::GetDeviceGuardImpl(device.type())->GetStream(device));
}

void uniform_cuda_kernel(Tensor &tensor, double from, double to,
                         const std::optional<Generator> &generator) {
    const Device device = tensor.GetDevice();
    CHECK(device.IsCUDA());
    const int64_t n = tensor.NumElements();
    if (n == 0) {
        return;
    }
    core::DeviceGuard guard(device);
    auto *cuda_generator = get_generator_or_default<core::cuda::CUDAGeneratorImpl>(
        generator, core::cuda::getDefaultCUDAGenerator(device.index()));

    uint64_t seed = 0;
    uint64_t subsequence = 0;
    {
        std::lock_guard<std::mutex> lock(cuda_generator->mutex_);
        seed = cuda_generator->current_seed();
        subsequence = cuda_generator->philox_subsequence(static_cast<uint64_t>(n));
    }

    const int blocks = static_cast<int>((n + kThreadsPerBlock - 1) / kThreadsPerBlock);
    const auto *stream = get_cuda_stream(device);
    core::cuda::DispatchCudaFunc<DataType::kFLOAT16, DataType::kBFLOAT16, DataType::kFLOAT32, DataType::kFLOAT64>(
        tensor.Dtype(),
        [&]<typename storage_t>() {
            using random_t = std::conditional_t<std::is_same_v<storage_t, double>, double, float>;
            UniformKernel<storage_t, random_t><<<blocks, kThreadsPerBlock, 0, stream->cuda_stream()>>>(
                static_cast<storage_t *>(tensor.DataPtr()), n, static_cast<random_t>(from), static_cast<random_t>(to),
                seed, subsequence);
        },
        "CUDA uniform");
    CUDA_CHECK(cudaGetLastError());
}

void normal_cuda_kernel(Tensor &tensor, double mean, double std,
                        const std::optional<Generator> &generator) {
    const Device device = tensor.GetDevice();
    CHECK(device.IsCUDA());
    const int64_t n = tensor.NumElements();
    if (n == 0) {
        return;
    }
    core::DeviceGuard guard(device);
    auto *cuda_generator = get_generator_or_default<core::cuda::CUDAGeneratorImpl>(
        generator, core::cuda::getDefaultCUDAGenerator(device.index()));

    uint64_t seed = 0;
    uint64_t subsequence = 0;
    {
        std::lock_guard<std::mutex> lock(cuda_generator->mutex_);
        seed = cuda_generator->current_seed();
        subsequence = cuda_generator->philox_subsequence(static_cast<uint64_t>(n));
    }

    const int blocks = static_cast<int>((n + kThreadsPerBlock - 1) / kThreadsPerBlock);
    const auto *stream = get_cuda_stream(device);
    core::cuda::DispatchCudaFunc<DataType::kFLOAT16, DataType::kBFLOAT16, DataType::kFLOAT32, DataType::kFLOAT64>(
        tensor.Dtype(),
        [&]<typename storage_t>() {
            using random_t = std::conditional_t<std::is_same_v<storage_t, double>, double, float>;
            NormalKernel<storage_t, random_t><<<blocks, kThreadsPerBlock, 0, stream->cuda_stream()>>>(
                static_cast<storage_t *>(tensor.DataPtr()), n, static_cast<random_t>(mean), static_cast<random_t>(std),
                seed, subsequence);
        },
        "CUDA normal");
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

using infini_train::normal_stub;
using infini_train::uniform_stub;

REGISTER_DISPATCH(uniform_stub, Device::DeviceType::kCUDA, &uniform_cuda_kernel);
REGISTER_DISPATCH(normal_stub, Device::DeviceType::kCUDA, &normal_cuda_kernel);

} // namespace infini_train::kernels::cuda
