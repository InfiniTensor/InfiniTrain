#include <cmath>
#include <cstdint>
#include <limits>

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/device.h"

#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"
#include "infini_train/src/random_utils.h"

namespace infini_train::kernels::cuda {
namespace {
constexpr int kThreadsPerBlock = 256;
constexpr float kTwoPi = 6.283185307179586476925286766559f;
constexpr int kUniformValuesPerThread = 4;
constexpr int kNormalValuesPerThread = 2;

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

__device__ Philox4x32State Philox(uint64_t seed, uint64_t counter_index) {
    constexpr uint32_t kPhiloxW0 = 0x9E3779B9;
    constexpr uint32_t kPhiloxW1 = 0xBB67AE85;

    Philox4x32State counter{static_cast<uint32_t>(counter_index), static_cast<uint32_t>(counter_index >> 32), 0, 0};
    uint32_t key0 = static_cast<uint32_t>(seed);
    uint32_t key1 = static_cast<uint32_t>(seed >> 32);

    for (int round = 0; round < 10; ++round) {
        counter = PhiloxRound(counter, key0, key1);
        key0 += kPhiloxW0;
        key1 += kPhiloxW1;
    }
    return counter;
}

__device__ uint32_t PhiloxStateWord(const Philox4x32State &values, uint32_t lane) {
    switch (lane) {
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

__device__ void PhiloxRandomWords(uint64_t seed, uint64_t offset, uint32_t *output, int count) {
    uint64_t counter_index = offset / 4;
    uint32_t lane = static_cast<uint32_t>(offset % 4);
    Philox4x32State values = Philox(seed, counter_index);

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (i >= count) {
            return;
        }
        output[i] = PhiloxStateWord(values, lane++);
        if (lane == 4 && i + 1 < count) {
            values = Philox(seed, ++counter_index);
            lane = 0;
        }
    }
}

__device__ float UintToUnitFloat(uint32_t value) { return static_cast<float>(value >> 8) * 0x1.0p-24f; }

__global__ void RandomUniformFloat32Kernel(float *data, int64_t num_elements, float from, float to, uint64_t seed,
                                           uint64_t offset) {
    const int64_t thread_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t first_element = thread_index * kUniformValuesPerThread;
    if (first_element >= num_elements) {
        return;
    }

    const int count
        = static_cast<int>(num_elements - first_element < kUniformValuesPerThread ? num_elements - first_element
                                                                                  : kUniformValuesPerThread);
    uint32_t random_words[kUniformValuesPerThread];
    PhiloxRandomWords(seed, offset + static_cast<uint64_t>(first_element), random_words, count);

#pragma unroll
    for (int i = 0; i < kUniformValuesPerThread; ++i) {
        if (i >= count) {
            return;
        }
        const float u = UintToUnitFloat(random_words[i]);
        const float value = from + (to - from) * u;
        // Preserve the half-open interval if the final float rounding reaches the upper bound.
        data[first_element + i] = value == to ? from : value;
    }
}

__global__ void RandomNormalFloat32Kernel(float *data, int64_t num_elements, float mean, float stddev, uint64_t seed,
                                          uint64_t offset) {
    const int64_t thread_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t first_element = thread_index * kNormalValuesPerThread;
    if (first_element >= num_elements) {
        return;
    }

    const int count = static_cast<int>(
        num_elements - first_element < kNormalValuesPerThread ? num_elements - first_element : kNormalValuesPerThread);
    uint32_t random_words[kNormalValuesPerThread * 2];
    PhiloxRandomWords(seed, offset + static_cast<uint64_t>(first_element) * 2, random_words, count * 2);

#pragma unroll
    for (int i = 0; i < kNormalValuesPerThread; ++i) {
        if (i >= count) {
            return;
        }
        const float u1 = 1.0f - UintToUnitFloat(random_words[i * 2]);
        const float u2 = UintToUnitFloat(random_words[i * 2 + 1]);
        const float z = sqrtf(-2.0f * logf(u1)) * cosf(kTwoPi * u2);
        data[first_element + i] = mean + stddev * z;
    }
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
    infini_train::detail::CheckUniformBounds(from, to);
    if (num_elements == 0) {
        return;
    }
    const int64_t work_items = num_elements / kUniformValuesPerThread + (num_elements % kUniformValuesPerThread != 0);
    CHECK_LE(work_items, static_cast<int64_t>(std::numeric_limits<int>::max()) * kThreadsPerBlock)
        << "Random uniform tensor is too large";
    const int blocks = static_cast<int>((work_items + kThreadsPerBlock - 1) / kThreadsPerBlock);
    RandomUniformFloat32Kernel<<<blocks, kThreadsPerBlock, 0, GetCudaStream(device)>>>(
        static_cast<float *>(data), num_elements, from, to, seed, offset);
    CUDA_CHECK(cudaGetLastError());
}

void RandomNormalFloat32(void *data, int64_t num_elements, float mean, float stddev, uint64_t seed, uint64_t offset,
                         Device device) {
    CHECK_GE(num_elements, 0);
    CHECK_GE(stddev, 0.0f);
    if (num_elements == 0) {
        return;
    }
    const int64_t work_items = num_elements / kNormalValuesPerThread + (num_elements % kNormalValuesPerThread != 0);
    CHECK_LE(work_items, static_cast<int64_t>(std::numeric_limits<int>::max()) * kThreadsPerBlock)
        << "Random normal tensor is too large";
    const int blocks = static_cast<int>((work_items + kThreadsPerBlock - 1) / kThreadsPerBlock);
    RandomNormalFloat32Kernel<<<blocks, kThreadsPerBlock, 0, GetCudaStream(device)>>>(
        static_cast<float *>(data), num_elements, mean, stddev, seed, offset);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace infini_train::kernels::cuda
