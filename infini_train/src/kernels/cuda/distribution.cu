#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <type_traits>

#include <curand_kernel.h>

#include "glog/logging.h"

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/common/cuda/kernel_helper.cuh"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/generator.h"
#include "infini_train/include/generator_impl.h"
#include "infini_train/include/tensor.h"
#include "infini_train/src/core/runtime/cuda/cuda_dispatch.h"
#include "infini_train/src/core/runtime/cuda/cuda_generator_impl.h"
#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

namespace infini_train::kernels::cuda {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kMaxUniformAttempts = 3;

template <typename RandomT> __device__ RandomT UniformSample(curandStatePhilox4_32_10_t *state) {
    if constexpr (std::is_same_v<RandomT, double>) {
        double sample;
        do {
            const double2 values = curand_uniform2_double(state);
            sample = 1.0 - values.x;
            // Subtraction can round a tiny positive draw to 1.0.
        } while (sample >= 1.0);
        return sample;
    } else {
        return static_cast<float>(curand(state)) * 0x1p-32f;
    }
}

template <typename RandomT> __device__ RandomT NormalSample(curandStatePhilox4_32_10_t *state) {
    if constexpr (std::is_same_v<RandomT, double>) {
        return curand_normal_double(state);
    } else {
        return curand_normal(state);
    }
}

template <typename StorageT, typename RandomT>
__global__ void UniformKernel(StorageT *data, int64_t n, double from, double to, uint64_t seed, uint64_t subsequence) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }

    const StorageT from_value = common::cuda::Cast<StorageT>(from);
    if (from == to) {
        data[index] = from_value;
        return;
    }
    curandStatePhilox4_32_10_t state;
    curand_init(seed, subsequence + static_cast<uint64_t>(index), 0, &state);
    const RandomT sample_from = static_cast<RandomT>(from);
    const RandomT sample_to = static_cast<RandomT>(to);
    StorageT value;
    int attempt = 0;
    do {
        value = common::cuda::Cast<StorageT>(sample_from + UniformSample<RandomT>(&state) * (sample_to - sample_from));
        ++attempt;
    } while (attempt < kMaxUniformAttempts
             && (common::cuda::Cast<double>(value) < from || common::cuda::Cast<double>(value) >= to));
    // Bounded retries may still leave out-of-range values in very narrow intervals.
    data[index] = common::cuda::Cast<double>(value) == to ? from_value : value;
}

template <typename StorageT, typename RandomT>
__global__ void NormalKernel(StorageT *data, int64_t n, RandomT mean, RandomT std, uint64_t seed,
                             uint64_t subsequence) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }

    curandStatePhilox4_32_10_t state;
    curand_init(seed, subsequence + static_cast<uint64_t>(index), 0, &state);
    data[index] = common::cuda::Cast<StorageT>(mean + NormalSample<RandomT>(&state) * std);
}

const core::cuda::CudaStream *GetCudaStream(const Device &device) {
    return dynamic_cast<const core::cuda::CudaStream *>(core::GetDeviceGuardImpl(device.type())->GetStream(device));
}

} // namespace

void Uniform(const std::shared_ptr<Tensor> tensor, double from, double to, const std::optional<Generator> gen) {
    const Device device = tensor->GetDevice();
    CHECK(device.IsCUDA());
    const int64_t n = tensor->NumElements();
    if (n == 0) {
        return;
    }
    core::DeviceGuard guard(device);
    auto &cuda_generator = GetGeneratorOrDefault<core::cuda::CUDAGeneratorImpl>(
        gen, core::cuda::GetDefaultCudaGenerator(device.index()), device);

    uint64_t seed = 0;
    uint64_t subsequence = 0;
    if (from != to) {
        std::lock_guard<std::mutex> lock(cuda_generator.mutex_);
        seed = cuda_generator.current_seed();
        subsequence = cuda_generator.ReservePhiloxSubsequence(static_cast<uint64_t>(n));
    }

    const int blocks = static_cast<int>((n + kThreadsPerBlock - 1) / kThreadsPerBlock);
    const auto *stream = GetCudaStream(device);
    core::cuda::DispatchCudaFunc<DataType::kFLOAT16, DataType::kBFLOAT16, DataType::kFLOAT32, DataType::kFLOAT64>(
        tensor->Dtype(),
        [&]<typename StorageT>() {
            using RandomT = std::conditional_t<std::is_same_v<StorageT, double>, double, float>;
            UniformKernel<StorageT, RandomT><<<blocks, kThreadsPerBlock, 0, stream->cuda_stream()>>>(
                static_cast<StorageT *>(tensor->DataPtr()), n, from, to, seed, subsequence);
        },
        "CUDA uniform");
    CUDA_CHECK(cudaGetLastError());
}

void Normal(const std::shared_ptr<Tensor> tensor, double mean, double std, const std::optional<Generator> gen) {
    const Device device = tensor->GetDevice();
    CHECK(device.IsCUDA());
    const int64_t n = tensor->NumElements();
    if (n == 0) {
        return;
    }
    core::DeviceGuard guard(device);
    auto &cuda_generator = GetGeneratorOrDefault<core::cuda::CUDAGeneratorImpl>(
        gen, core::cuda::GetDefaultCudaGenerator(device.index()), device);

    uint64_t seed = 0;
    uint64_t subsequence = 0;
    {
        std::lock_guard<std::mutex> lock(cuda_generator.mutex_);
        seed = cuda_generator.current_seed();
        subsequence = cuda_generator.ReservePhiloxSubsequence(static_cast<uint64_t>(n));
    }

    const int blocks = static_cast<int>((n + kThreadsPerBlock - 1) / kThreadsPerBlock);
    const auto *stream = GetCudaStream(device);
    core::cuda::DispatchCudaFunc<DataType::kFLOAT16, DataType::kBFLOAT16, DataType::kFLOAT32, DataType::kFLOAT64>(
        tensor->Dtype(),
        [&]<typename StorageT>() {
            using RandomT = std::conditional_t<std::is_same_v<StorageT, double>, double, float>;
            NormalKernel<StorageT, RandomT><<<blocks, kThreadsPerBlock, 0, stream->cuda_stream()>>>(
                static_cast<StorageT *>(tensor->DataPtr()), n, static_cast<RandomT>(mean), static_cast<RandomT>(std),
                seed, subsequence);
        },
        "CUDA normal");
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_DISTRIBUTION_KERNEL(kernel_name)                                                                 \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_DISTRIBUTION_KERNEL(Uniform)
REGISTER_CUDA_DISTRIBUTION_KERNEL(Normal)

#undef REGISTER_CUDA_DISTRIBUTION_KERNEL
