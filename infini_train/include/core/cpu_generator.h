#pragma once

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "infini_train/include/core/generator.h"
#include "infini_train/include/device.h"

namespace infini_train::core {

/**
 * CPU backend Generator.
 * State organization: the random sequence is driven by a std::mt19937_64
 * engine. GetState() serializes a self-describing blob containing a backend
 * magic, the current seed and the full engine state, allowing SetState() to
 * resume the exact same sequence and to reject states coming from other
 * backends.
 */  
class CPUGeneratorImpl : public GeneratorImpl {
public:
    static constexpr Device::DeviceType kDeviceType = Device::DeviceType::kCPU;

    explicit CPUGeneratorImpl(uint64_t seed = kDefaultSeed);

    void SetCurrentSeed(uint64_t seed) override;
    uint64_t CurrentSeed() const override;
    uint64_t Seed() override;

    // CPU mt19937 engine is not counter based; only offset 0 is supported.
    void SetOffset(uint64_t offset) override;
    uint64_t GetOffset() const override;

    std::vector<uint8_t> GetState() const override;
    void SetState(const std::vector<uint8_t> &state) override;

    Device GetDevice() const override;
    std::shared_ptr<GeneratorImpl> Clone() const override;

    // --- Internal kernel extension interface ----------------------------
    // The methods below expose backend engine details for use by random
    // operators (e.g. init::Uniform). They are NOT part of the user-facing
    // Generator API and should not be called from outside the kernel layer.
    //
    // Direct access to the underlying engine for random operators. Each draw
    // advances the engine, i.e. advances the generator state.
    std::mt19937_64 &Engine();

private:
    static constexpr uint64_t kDefaultSeed = 67280421310721ULL;

    uint64_t seed_ = kDefaultSeed;
    std::mt19937_64 engine_;
};

} // namespace infini_train::core
