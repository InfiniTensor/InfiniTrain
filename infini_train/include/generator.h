#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "infini_train/include/device.h"

namespace infini_train {

class GeneratorImpl;
namespace detail {
class GeneratorAccessor;
}

class Generator {
public:
    void ManualSeed(uint64_t seed);
    uint64_t Seed();
    uint64_t InitialSeed() const;
    std::vector<uint8_t> GetState() const;
    void SetState(const std::vector<uint8_t> &state);
    // The CUDA index is the generator's home index. An explicit CUDA generator
    // may be consumed by tensors on another CUDA index.
    Device GetDevice() const;

private:
    explicit Generator(std::shared_ptr<GeneratorImpl> impl);

    std::shared_ptr<GeneratorImpl> impl_;

    friend class detail::GeneratorAccessor;
    friend std::shared_ptr<Generator> MakeCPUGenerator(uint64_t seed);
    friend std::shared_ptr<Generator> MakeCUDAGenerator(int8_t device_index, uint64_t seed);
};

std::shared_ptr<Generator> MakeCPUGenerator(uint64_t seed = 42);
std::shared_ptr<Generator> MakeCUDAGenerator(int8_t device_index, uint64_t seed = 42);
std::shared_ptr<Generator> GetDefaultCPUGenerator();
std::shared_ptr<Generator> GetDefaultCUDAGenerator(int8_t device_index);
std::shared_ptr<Generator> GetDefaultGenerator(const Device &device);
// ManualSeed is an alias for ManualSeedAll.
void ManualSeed(uint64_t seed);
void ManualSeedAll(uint64_t seed);

} // namespace infini_train
