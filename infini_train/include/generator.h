#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>

#include "infini_train/include/device.h"

namespace infini_train {

class Tensor;

namespace detail {

// Validates the common Tensor contract for serialized RNG states.
void check_rng_state(const Tensor &state);

} // namespace detail

// Base interface for device-specific random number generators.
class GeneratorImpl {
public:
    explicit GeneratorImpl(Device device) : device_(device) {}
    virtual ~GeneratorImpl() = default;

    GeneratorImpl(const GeneratorImpl &other) = delete;
    GeneratorImpl(GeneratorImpl &&other) = delete;
    GeneratorImpl &operator=(const GeneratorImpl &other) = delete;
    GeneratorImpl &operator=(GeneratorImpl &&other) = delete;

    virtual void set_current_seed(uint64_t seed) = 0;
    virtual uint64_t current_seed() const = 0;
    virtual uint64_t seed() = 0;
    virtual void set_state(const Tensor &state) = 0;
    virtual std::shared_ptr<Tensor> get_state() const = 0;

    std::shared_ptr<GeneratorImpl> clone() const { return std::shared_ptr<GeneratorImpl>(clone_impl()); }

    Device device() const { return device_; }

    // Callers must lock this mutex when an operation spans multiple generator calls.
    std::mutex mutex_;

protected:
    Device device_;

    virtual GeneratorImpl *clone_impl() const = 0;
};

// A lightweight handle with shared-copy semantics. Use clone() for an independent state.
class Generator {
public:
    static constexpr uint64_t kDefaultSeed = 67280421310721;

    Generator() = default;

    explicit Generator(std::shared_ptr<GeneratorImpl> impl);

    Generator(const Generator &) = default;
    Generator &operator=(const Generator &) = default;
    Generator(Generator &&) = default;
    Generator &operator=(Generator &&) = default;

    ~Generator() = default;

    void set_current_seed(uint64_t seed) const { impl_->set_current_seed(seed); }
    uint64_t current_seed() const { return impl_->current_seed(); }
    uint64_t seed() { return impl_->seed(); }

    void set_state(const Tensor &state);
    std::shared_ptr<Tensor> get_state() const;

    Device device() const { return impl_->device(); }

    Generator clone() const { return Generator(impl_->clone()); }

    std::mutex &mutex() const { return impl_->mutex_; }

    // Prefer check_generator<T>(); this unchecked accessor assumes a matching backend.
    template <typename T> T *get() const { return static_cast<T *>(impl_.get()); }

    GeneratorImpl *unsafeGetGeneratorImpl() const { return impl_.get(); }
    bool defined() const { return impl_ != nullptr; }

    friend bool operator==(const Generator &a, const Generator &b) { return a.impl_ == b.impl_; }
    friend bool operator!=(const Generator &a, const Generator &b) { return !(a == b); }

private:
    std::shared_ptr<GeneratorImpl> impl_;
};

// Internal factory for backend implementations.
template <class Impl, class... Args> Generator make_generator(Args &&...args) {
    return Generator(std::make_shared<Impl>(std::forward<Args>(args)...));
}

template <typename T> T *check_generator(const Generator &generator) {
    if (!generator.defined()) {
        throw std::invalid_argument("Generator with undefined implementation is not allowed");
    }
    if (T::device_type() != generator.device().type()) {
        throw std::invalid_argument("Generator device type does not match the requested backend");
    }

    auto *impl = dynamic_cast<T *>(generator.unsafeGetGeneratorImpl());
    if (impl == nullptr) {
        throw std::invalid_argument("Generator implementation does not match the requested backend");
    }
    return impl;
}

template <typename T>
T *get_generator_or_default(const std::optional<Generator> &generator, const Generator &default_generator) {
    return generator.has_value() && generator->defined() ? check_generator<T>(*generator)
                                                         : check_generator<T>(default_generator);
}

// Creates a generator for the requested device without exposing its backend implementation.
Generator CreateGenerator(const Device &device, uint64_t seed = Generator::kDefaultSeed);

// Returns the lazily initialized default generator for the requested device.
const Generator &GetDefaultGenerator(const Device &device);

// Reset the default generators for all enabled devices.
void manual_seed(uint64_t seed);

} // namespace infini_train
