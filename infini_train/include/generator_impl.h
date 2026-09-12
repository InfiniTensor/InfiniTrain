#pragma once

// Internal Generator implementation interface. Application code should include generator.h.

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "infini_train/include/device.h"
#include "infini_train/include/generator.h"
#include "infini_train/include/tensor.h"

namespace infini_train {

class Tensor;

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
    virtual uint64_t Seed() = 0;
    virtual void set_state(const Tensor &state) = 0;
    virtual std::shared_ptr<Tensor> get_state() const = 0;

    std::shared_ptr<GeneratorImpl> Clone() const { return std::shared_ptr<GeneratorImpl>(CloneImpl()); }
    Device device() const { return device_; }

    // Callers must hold this mutex when accessing shared RNG state concurrently.
    std::mutex mutex_;

protected:
    Device device_;
    virtual GeneratorImpl *CloneImpl() const = 0;
};

class GeneratorAccessor {
public:
    static GeneratorImpl &Get(const Generator &generator) {
        if (!generator.impl_) {
            throw std::invalid_argument("Undefined Generator");
        }
        return *generator.impl_;
    }

    static std::mutex &Mutex(const Generator &generator) { return Get(generator).mutex_; }

    static Generator FromImpl(std::shared_ptr<GeneratorImpl> impl) { return Generator(std::move(impl)); }
};

template <typename T> T &CheckedGeneratorImpl(const Generator &generator, const Device &expected_device) {
    static_assert(std::is_base_of_v<GeneratorImpl, T>);
    auto &base = GeneratorAccessor::Get(generator);
    if (base.device() != expected_device) {
        throw std::invalid_argument("Generator device mismatch");
    }
    auto *typed = dynamic_cast<T *>(&base);
    if (typed == nullptr) {
        throw std::invalid_argument("Generator backend mismatch");
    }
    return *typed;
}

template <class Impl, class... Args> Generator MakeGenerator(Args &&...args) {
    return GeneratorAccessor::FromImpl(std::make_shared<Impl>(std::forward<Args>(args)...));
}

template <typename T>
T &GetGeneratorOrDefault(const std::optional<Generator> &generator, const Generator &default_generator,
                         const Device &expected_device) {
    const Generator &chosen = generator.has_value() ? *generator : default_generator;
    return CheckedGeneratorImpl<T>(chosen, expected_device);
}

namespace detail {
void CheckRngState(const Tensor &state);
} // namespace detail

} // namespace infini_train
