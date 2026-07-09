#pragma once

// distribution.h — Minimal distribution template infrastructure.
//
// Mirrors PyTorch's DistributionTemplates.h / cpu_DistributionTemplates.h /
// cuda_DistributionTemplates.h three-layer design, simplified for InfiniTrain's
// current needs:
//
//   1. distribution_impl  — parameter validation + backend dispatch
//   2. cpu::*_kernel       — CPU backend: locks generator, fills elements
//   3. cuda::*_kernel      — CUDA backend: host-side fallback for now
//
// The key improvement over the old FillBuffer is:
//   - Distribution logic is decoupled from init.cc consumers
//   - Generator resolution (explicit vs default) follows PyTorch's
//     check_generator / GetGeneratorOrDefault pattern
//   - Each backend owns its locking strategy
//   - Ready for future device-kernel extension on CUDA

#include <cstdint>
#include <mutex>
#include <optional>
#include <random>
#include <vector>

#include "infini_train/include/core/cpu_generator.h"
#include "infini_train/include/core/generator.h"
#ifdef USE_CUDA
#include "infini_train/include/core/cuda_generator.h"
#endif
#include "infini_train/include/device.h"

namespace infini_train::core::distribution {

// ============================== CPU backend ==============================

namespace cpu {

// Fills `buffer` with values drawn from `dist`, using the CPU generator's
// engine directly. The generator mutex is held for the entire draw loop
// because Engine() returns a reference to the generator's internal state.
//
// Corresponds to PyTorch's cpu::uniform_kernel / cpu::normal_kernel pattern,
// simplified to operate on a flat float buffer instead of a TensorIterator.
template <typename Dist>
void FillKernel(float *data, int64_t numel, CPUGeneratorImpl *impl, Dist dist) {
    std::lock_guard<std::mutex> lock(impl->mutex_);
    auto &engine = impl->Engine();
    for (int64_t i = 0; i < numel; ++i) {
        data[i] = dist(engine);
    }
}

}  // namespace cpu

// ============================== CUDA backend =============================

#ifdef USE_CUDA
namespace cuda {

// Host-side fallback: generates random values on the host using a temporary
// mt19937_64 engine derived from the CUDA generator's (seed, offset) pair,
// then the caller copies the result to the device.
//
// A future device-kernel path will generate values directly on the GPU via
// Philox (mirroring PyTorch's distribution_nullary_kernel).
//
// Locking strategy matches PyTorch: only the offset reservation (PhiloxEngine-
// Inputs) touches impl state; the subsequent host-side generation uses purely
// local variables and runs outside the impl lock to reduce contention.
template <typename Dist>
void FillKernel(float *data, int64_t numel, CUDAGeneratorImpl *impl, Dist dist) {
    uint64_t seed, offset;
    {
        std::lock_guard<std::mutex> lock(impl->mutex_);
        std::tie(seed, offset) = impl->PhiloxEngineInputs(static_cast<uint64_t>(numel));
    }
    // Mix a backend-specific constant so that the derived engine produces a
    // sequence distinct from the CPU generator even when seed and offset
    // values coincide (e.g. both backends seeded identically via ManualSeed).
    constexpr uint64_t kCUDABackendTag = 0xD2511F53D9E08B47ULL;
    std::mt19937_64 engine(seed ^ ((offset + kCUDABackendTag) * 0x9E3779B97F4A7C15ULL));
    for (int64_t i = 0; i < numel; ++i) {
        data[i] = dist(engine);
    }
}

}  // namespace cuda
#endif  // USE_CUDA

// ======================== Distribution dispatch ==========================
//
// These mirror PyTorch's DistributionTemplates.h top-level functions
// (normal_impl_, uniform_impl_, etc.) which validate parameters, resolve the
// generator, and dispatch to the correct backend kernel.

// Fills `buffer` by drawing from the given distribution, using the resolved
// generator for the target device. Replaces the old FillBuffer in init.cc.
//
// Flow:
//   1. Resolve generator (explicit or device default)
//   2. Dispatch to cpu::FillKernel or cuda::FillKernel
//   3. Backend kernel handles locking and element-wise generation
template <typename Dist>
void FillBuffer(std::vector<float> &buffer, const Device &device,
                const std::optional<Generator> &generator, Dist dist) {
    auto *data = buffer.data();
    const auto numel = static_cast<int64_t>(buffer.size());

    if (device.IsCPU()) {
        auto *impl = GetGeneratorOrDefault<CPUGeneratorImpl>(generator, device);
        cpu::FillKernel(data, numel, impl, dist);
        return;
    }

#ifdef USE_CUDA
    auto *impl = GetGeneratorOrDefault<CUDAGeneratorImpl>(generator, device);
    cuda::FillKernel(data, numel, impl, dist);
#else
    LOG(FATAL) << "Random operator on non-CPU device requires CUDA support";
#endif
}

// Convenience wrappers matching PyTorch's named distribution templates.

// uniform_impl_: fills buffer with values from [from, to).
inline void UniformFill(std::vector<float> &buffer, const Device &device,
                        const std::optional<Generator> &generator, float from, float to) {
    FillBuffer(buffer, device, generator, std::uniform_real_distribution<float>(from, to));
}

// normal_impl_: fills buffer with values from N(mean, std).
inline void NormalFill(std::vector<float> &buffer, const Device &device,
                       const std::optional<Generator> &generator, float mean, float std) {
    FillBuffer(buffer, device, generator, std::normal_distribution<float>(mean, std));
}

}  // namespace infini_train::core::distribution
