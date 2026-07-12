#include <cmath>
#include <cstdint>
#include <limits>

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/device.h"

#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

namespace infini_train::kernels::cuda {
namespace {
constexpr int kThreadsPerBlock = 256;
constexpr float kTwoPi = 6.283185307179586476925286766559f;

void CheckUniformBounds(float from, float to) {
    CHECK_LE(from, to);
    CHECK(std::isfinite(from)) << "Uniform lower bound must be finite";
    CHECK(std::isfinite(to)) << "Uniform upper bound must be finite";
    const double range = static_cast<double>(to) - static_cast<double>(from);
    CHECK_LE(range, static_cast<double>(std::numeric_limits<float>::max()))
        << "Uniform bounds range exceeds float maximum";
}

struct Philox4x32State {
    uint32_t c0;
    uint32_t c1;
    uint32_t c2;
    uint32_t c3;
};

__device__ uint32_t MulHi(uint32_t a, uint32_t b) {
    return static_cast<uint32_t>((static_cast<uint64_t>(a) * b) >> 32);
}

__device__ Philox4x32State PhiloxRound(Philox4x32State counter, uint32_t key0, uint32_t key1) {
    constexpr uint32_t kPhiloxM0 = 0xD2511F53;
    constexpr uint32_t kPhiloxM1 = 0xCD9E8D57;

    const uint32_t lo0 = counter.c0 * kPhiloxM0;
    const uint32_t hi0 = MulHi(counter.c0, kPhiloxM0);
    const uint32_t lo1 = counter.c2 * kPhiloxM1;
    const uint32_t hi1 = MulHi(counter.c2, kPhiloxM1);

    return {hi1 ^ counter.c1 ^ key0, lo1, hi0 ^ counter.c3 ^ key1, lo0};
}

__device__ Philox4x32State Philox(uint64_t seed, uint64_t subsequence) {
    constexpr uint32_t kPhiloxW0 = 0x9E3779B9;
    constexpr uint32_t kPhiloxW1 = 0xBB67AE85;

    Philox4x32State counter{static_cast<uint32_t>(subsequence), static_cast<uint32_t>(subsequence >> 32), 0, 0};
    uint32_t key0 = static_cast<uint32_t>(seed);
    uint32_t key1 = static_cast<uint32_t>(seed >> 32);

    for (int round = 0; round < 10; ++round) {
        counter = PhiloxRound(counter, key0, key1);
        key0 += kPhiloxW0;
        key1 += kPhiloxW1;
    }
    return counter;
}

__device__ uint32_t PhiloxRandomUint(uint64_t seed, uint64_t offset) {
    const Philox4x32State values = Philox(seed, offset / 4);
    switch (offset % 4) {
    case 0:
        return values.c0;
    case 1:
        return values.c1;
    case 2:
        return values.c2;
    default:
        return values.c3;
    }
}

__device__ float UintToUniform(uint32_t value) { return static_cast<float>(value >> 8) * 0x1.0p-24f; }

__global__ void RandomUniformFloat32Kernel(float *data, int64_t num_elements, float from, float to, uint64_t seed,
                                           uint64_t offset) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_elements) {
        return;
    }
    const float u = UintToUniform(PhiloxRandomUint(seed, offset + static_cast<uint64_t>(idx)));
    data[idx] = from + (to - from) * u;
}

__global__ void RandomNormalFloat32Kernel(float *data, int64_t num_elements, float mean, float std, uint64_t seed,
                                          uint64_t offset) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_elements) {
        return;
    }

    const uint64_t element_offset = offset + static_cast<uint64_t>(idx) * 2;
    float u1 = UintToUniform(PhiloxRandomUint(seed, element_offset));
    const float u2 = UintToUniform(PhiloxRandomUint(seed, element_offset + 1));
    u1 = fmaxf(u1, 0x1.0p-24f);
    const float z = sqrtf(-2.0f * logf(u1)) * cosf(kTwoPi * u2);
    data[idx] = mean + std * z;
}

cudaStream_t GetCudaStream(Device device) {
    auto *stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
        infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device));
    CHECK_NOTNULL(stream);
    return stream->cuda_stream();
}
} // namespace

void RandomUniformFloat32(void *data, int64_t num_elements, float from, float to, uint64_t seed, uint64_t offset,
                          Device device) {
    CHECK_GE(num_elements, 0);
    CheckUniformBounds(from, to);
    if (num_elements == 0) {
        return;
    }
    CHECK_LE(num_elements, static_cast<int64_t>(std::numeric_limits<int>::max()) * kThreadsPerBlock)
        << "Random uniform tensor is too large";
    const int blocks = static_cast<int>((num_elements + kThreadsPerBlock - 1) / kThreadsPerBlock);
    RandomUniformFloat32Kernel<<<blocks, kThreadsPerBlock, 0, GetCudaStream(device)>>>(
        static_cast<float *>(data), num_elements, from, to, seed, offset);
    CUDA_CHECK(cudaGetLastError());
}

void RandomNormalFloat32(void *data, int64_t num_elements, float mean, float std, uint64_t seed, uint64_t offset,
                         Device device) {
    CHECK_GE(num_elements, 0);
    CHECK_GE(std, 0.0f);
    if (num_elements == 0) {
        return;
    }
    CHECK_LE(num_elements, static_cast<int64_t>(std::numeric_limits<int>::max()) * kThreadsPerBlock)
        << "Random normal tensor is too large";
    const int blocks = static_cast<int>((num_elements + kThreadsPerBlock - 1) / kThreadsPerBlock);
    RandomNormalFloat32Kernel<<<blocks, kThreadsPerBlock, 0, GetCudaStream(device)>>>(
        static_cast<float *>(data), num_elements, mean, std, seed, offset);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace infini_train::kernels::cuda
