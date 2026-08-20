#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "infini_train/include/core/generator.h"
#include "infini_train/include/device.h"

namespace infini_train::core {

/**
 * CUDA backend Generator.
 *
 * Unlike the CPU backend, CUDA random kernels are counter (Philox) based. The
 * host side keeps only a lightweight state: the seed and a 64-bit Philox
 * offset. Random kernels read the (seed, offset) pair and advance the offset by
 * the number of random values they consumed, so that subsequent kernel launches
 * do not overlap the random sequence.
 *
 * Each CUDA device maintains its own independent generator (see
 * detail::DefaultCUDAGenerator).
 */
class CUDAGeneratorImpl : public GeneratorImpl {
public:
    static constexpr Device::DeviceType kDeviceType = Device::DeviceType::kCUDA;

    explicit CUDAGeneratorImpl(int8_t device_index = 0, uint64_t seed = kDefaultSeed);

    void SetCurrentSeed(uint64_t seed) override;
    uint64_t CurrentSeed() const override;
    uint64_t Seed() override;

    void SetOffset(uint64_t offset) override;
    uint64_t GetOffset() const override;

    std::vector<uint8_t> GetState() const override;
    void SetState(const std::vector<uint8_t> &state) override;

    Device GetDevice() const override;
    std::shared_ptr<GeneratorImpl> Clone() const override;

    // --- Internal kernel extension interface ----------------------------
    // The methods below expose Philox counter details for use by random
    // operators (e.g. init::Uniform on CUDA tensors). They are NOT part of
    // the user-facing Generator API and should not be called from outside the
    // kernel layer.
    //
    // Reserves `increment` Philox values for the next kernel launch and returns
    // the (seed, offset) pair the kernel must use. The offset is advanced so the
    // next call yields a non-overlapping subsequence.
    std::pair<uint64_t, uint64_t> PhiloxEngineInputs(uint64_t increment);

private:
    static constexpr uint64_t kDefaultSeed = 67280421310721ULL;

    int8_t device_index_ = 0;
    uint64_t seed_ = kDefaultSeed;
    uint64_t philox_offset_ = 0;
};

} // namespace infini_train::core
