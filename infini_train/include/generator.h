#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "infini_train/include/device.h"

namespace infini_train {

class GeneratorImpl {
public:
    virtual ~GeneratorImpl() = default;

    virtual void ManualSeed(uint64_t seed) = 0;
    virtual uint64_t Seed() = 0;
    virtual uint64_t InitialSeed() const = 0;
    virtual std::vector<uint8_t> GetState() const = 0;
    virtual void SetState(const std::vector<uint8_t> &state) = 0;
    virtual Device GetDevice() const = 0;
    virtual std::pair<uint64_t, uint64_t> ReserveRandomOffset(uint64_t increment) = 0;

    virtual void FillUniform(std::vector<float> &buffer, float from, float to) = 0;
    virtual void FillNormal(std::vector<float> &buffer, float mean, float std) = 0;
};

class Generator {
public:
    explicit Generator(std::shared_ptr<GeneratorImpl> impl);

    void ManualSeed(uint64_t seed);
    uint64_t Seed();
    uint64_t InitialSeed() const;
    std::vector<uint8_t> GetState() const;
    void SetState(const std::vector<uint8_t> &state);
    Device GetDevice() const;
    std::pair<uint64_t, uint64_t> ReserveRandomOffset(uint64_t increment);

    void FillUniform(std::vector<float> &buffer, float from, float to);
    void FillNormal(std::vector<float> &buffer, float mean, float std);

private:
    std::shared_ptr<GeneratorImpl> impl_;
};

std::shared_ptr<Generator> MakeCPUGenerator(uint64_t seed = 42);
std::shared_ptr<Generator> MakeCUDAGenerator(int8_t device_index, uint64_t seed = 42);
std::shared_ptr<Generator> GetDefaultCPUGenerator();
std::shared_ptr<Generator> GetDefaultCUDAGenerator(int8_t device_index);
std::shared_ptr<Generator> GetDefaultGenerator(const Device &device);
void ManualSeed(uint64_t seed);
void ManualSeedAll(uint64_t seed);

} // namespace infini_train
