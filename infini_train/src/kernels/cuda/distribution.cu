#include <cstdint>
#include <mutex>
#include <optional>

#include <curand_kernel.h>

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/core/runtime/distribution_stubs.h"
#include "infini_train/include/generator.h"
#include "infini_train/include/tensor.h"
#include "infini_train/src/core/runtime/cuda/cuda_generator_impl.h"
#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

namespace infini_train::kernels::cuda {
namespace {

constexpr int kThreadsPerBlock = 256;

__global__ void UniformKernel(float *data, int64_t n, float from, float to,
                              uint64_t seed, uint64_t subsequence) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }

    curandStatePhilox4_32_10_t state;
    curand_init(seed, subsequence + static_cast<uint64_t>(index), 0, &state);
    const float unit = static_cast<float>(curand(&state)) * 0x1p-32f;
    data[index] = from + unit * (to - from);
}

__global__ void NormalKernel(float *data, int64_t n, float mean, float std,
                             uint64_t seed, uint64_t subsequence) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }

    curandStatePhilox4_32_10_t state;
    curand_init(seed, subsequence + static_cast<uint64_t>(index), 0, &state);
    data[index] = mean + curand_normal(&state) * std;
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
    UniformKernel<<<blocks, kThreadsPerBlock, 0, stream->cuda_stream()>>>(
        static_cast<float *>(tensor.DataPtr()), n, static_cast<float>(from), static_cast<float>(to), seed, subsequence);
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
    NormalKernel<<<blocks, kThreadsPerBlock, 0, stream->cuda_stream()>>>(
        static_cast<float *>(tensor.DataPtr()), n, static_cast<float>(mean), static_cast<float>(std), seed, subsequence);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

using infini_train::normal_stub;
using infini_train::uniform_stub;

REGISTER_DISPATCH(uniform_stub, Device::DeviceType::kCUDA, &uniform_cuda_kernel);
REGISTER_DISPATCH(normal_stub, Device::DeviceType::kCUDA, &normal_cuda_kernel);

} // namespace infini_train::kernels::cuda
