#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "infini_train/include/core/generator/generator_impl.h"
#include "infini_train/include/device.h"

namespace infini_train::core {

class CPUGeneratorImpl : public GeneratorImpl {
public:
    explicit CPUGeneratorImpl(int8_t device_index);

    void SetCurrentSeed(uint64_t seed) override;
    uint64_t CurrentSeed() const override { return seed_; }
    uint64_t InitialSeed() const override { return initial_seed_; }

    std::vector<uint8_t> GetState() const override;
    void SetState(const std::vector<uint8_t> &state) override;
    uint32_t StateMagic() const override { return kMagic; }

    std::mt19937_64 &engine() { return engine_; }

protected:
    void Reseed(uint64_t seed) override;

private:
    static constexpr uint32_t kMagic = 0x47555043; // 'CPUG'
    static constexpr uint32_t kVersion = 1;

    uint64_t initial_seed_{0};
    uint64_t seed_{0};
    std::mt19937_64 engine_{0};
};

} // namespace infini_train::core
