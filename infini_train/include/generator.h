#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "infini_train/include/device.h"

namespace infini_train {
namespace core {
class GeneratorImpl;
class GeneratorImplRegistry;
} // namespace core

class Generator {
public:
    explicit Generator(Device device = Device(Device::DeviceType::kCPU, 0));

    Generator(const Generator &) = default;
    Generator &operator=(const Generator &) = default;
    Generator(Generator &&) = default;
    Generator &operator=(Generator &&) = default;

    void ManualSeed(uint64_t seed);
    uint64_t Seed();
    uint64_t InitialSeed() const;

    std::vector<uint8_t> GetState() const;
    void SetState(const std::vector<uint8_t> &state);

    Device device() const;

    const std::shared_ptr<core::GeneratorImpl> &impl() const { return impl_; }

private:
    explicit Generator(std::shared_ptr<core::GeneratorImpl> impl);
    static Generator FromImpl(std::shared_ptr<core::GeneratorImpl> impl);

    friend class core::GeneratorImplRegistry;
    friend Generator &default_generator(Device device);

    std::shared_ptr<core::GeneratorImpl> impl_;
};

Generator &default_generator(Device device = Device(Device::DeviceType::kCPU, 0));

void manual_seed(uint64_t seed);
void manual_seed_cuda(uint64_t seed);

} // namespace infini_train
