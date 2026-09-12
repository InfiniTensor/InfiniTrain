#pragma once

#include <cstdint>
#include <memory>

#include "infini_train/include/device.h"

namespace infini_train {

class Tensor;
class GeneratorImpl;

// A lightweight handle with shared-copy semantics. Use Clone() for an independent state.
class Generator {
public:
    static constexpr uint64_t kDefaultSeed = 67280421310721;

    Generator() = delete;

    Generator(const Generator &) = default;
    Generator &operator=(const Generator &) = default;
    Generator(Generator &&) = default;
    Generator &operator=(Generator &&) = default;

    ~Generator() = default;

    void set_current_seed(uint64_t seed) const;
    uint64_t current_seed() const;
    uint64_t Seed();
    void set_state(const Tensor &state);
    std::shared_ptr<Tensor> get_state() const;
    Device device() const;
    Generator Clone() const;

    friend bool operator==(const Generator &a, const Generator &b) { return a.impl_ == b.impl_; }
    friend bool operator!=(const Generator &a, const Generator &b) { return !(a == b); }

private:
    friend class GeneratorAccessor;

    explicit Generator(std::shared_ptr<GeneratorImpl> impl);
    std::shared_ptr<GeneratorImpl> impl_;
};

Generator CreateGenerator(const Device &device, uint64_t seed = Generator::kDefaultSeed);

// Returns the lazily initialized default generator for the requested device.
const Generator &GetDefaultGenerator(const Device &device);

// Initializes or resets the default generators for all enabled devices with the given seed.
void ManualSeed(uint64_t seed);

} // namespace infini_train
