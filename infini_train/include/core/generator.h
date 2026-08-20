#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>

#include "glog/logging.h"

#include "infini_train/include/device.h"

namespace infini_train::core {

/**
 * GeneratorImpl — Backend-agnostic polymorphic base class for random number generators.
 */

class GeneratorImpl {
public:
    GeneratorImpl() = default;
    virtual ~GeneratorImpl() = default;

    // Delete copy and move in favor of Clone().
    GeneratorImpl(const GeneratorImpl &) = delete;
    GeneratorImpl(GeneratorImpl &&) = delete;
    GeneratorImpl &operator=(const GeneratorImpl &) = delete;
    GeneratorImpl &operator=(GeneratorImpl &&) = delete;

    // Seed control --------------------------------------------------------
    virtual void SetCurrentSeed(uint64_t seed) = 0;
    virtual uint64_t CurrentSeed() const = 0;
    virtual uint64_t Seed() = 0;
    // Philox-style offset control. Only meaningful for counter based backends
    // (e.g. CUDA). Backends that cannot support random access may reject
    // non-zero offsets.
    virtual void SetOffset(uint64_t offset) = 0;
    virtual uint64_t GetOffset() const = 0;

    // State control -------------------------------------------------------
    virtual std::vector<uint8_t> GetState() const = 0;
    virtual void SetState(const std::vector<uint8_t> &state) = 0;

    virtual Device GetDevice() const = 0;
    virtual std::shared_ptr<GeneratorImpl> Clone() const = 0;

    std::mutex mutex_;
};

/*
 * Generator — User facing handle. Cheap to copy: copies share the same underlying impl,
 * matching the semantics of a generator reference (e.g. torch.Generator).
 */
class Generator {
public:
    Generator() = default;

    explicit Generator(std::shared_ptr<GeneratorImpl> impl) : impl_(std::move(impl)) {
        CHECK(impl_ != nullptr) << "GeneratorImpl with nullptr is not supported";
    }

    bool Defined() const { return static_cast<bool>(impl_); }

    // Seed control --------------------------------------------------------
    void SetCurrentSeed(uint64_t seed) {
        std::lock_guard<std::mutex> lock(impl_->mutex_);
        impl_->SetCurrentSeed(seed);
    }

    void ManualSeed(uint64_t seed) { SetCurrentSeed(seed); }
    uint64_t CurrentSeed() const {
        std::lock_guard<std::mutex> lock(impl_->mutex_);
        return impl_->CurrentSeed();
    }

    uint64_t Seed() {
        std::lock_guard<std::mutex> lock(impl_->mutex_);
        return impl_->Seed();
    }

    void SetOffset(uint64_t offset) {
        std::lock_guard<std::mutex> lock(impl_->mutex_);
        impl_->SetOffset(offset);
    }
    uint64_t GetOffset() const {
        std::lock_guard<std::mutex> lock(impl_->mutex_);
        return impl_->GetOffset();
    }

    // State control -------------------------------------------------------
    std::vector<uint8_t> GetState() const {
        std::lock_guard<std::mutex> lock(impl_->mutex_);
        return impl_->GetState();
    }
    void SetState(const std::vector<uint8_t> &state) {
        std::lock_guard<std::mutex> lock(impl_->mutex_);
        impl_->SetState(state);
    }

    // Misc ----------------------------------------------------------------
    Device GetDevice() const { return impl_->GetDevice(); }
    std::mutex &Mutex() { return impl_->mutex_; }

    GeneratorImpl *UnsafeGetImpl() const { return impl_.get(); }

    template <typename T> T *Get() const { return static_cast<T *>(impl_.get()); }

    Generator Clone() const {
        std::lock_guard<std::mutex> lock(impl_->mutex_);
        return Generator(impl_->Clone());
    }

    bool operator==(const Generator &other) const { return impl_ == other.impl_; }
    bool operator!=(const Generator &other) const { return !(*this == other); }

private:
    std::shared_ptr<GeneratorImpl> impl_;
};

namespace detail {
// Returns the process-wide default Generator for the given device.
const Generator &GetDefaultGenerator(const Device &device);

// Convenience accessors.
const Generator &DefaultCPUGenerator();
const Generator &DefaultCUDAGenerator(int8_t device_index = 0);

// Appends a little-endian uint64_t to the output byte vector.
inline void AppendU64(std::vector<uint8_t> &out, uint64_t value) {
    for (int i = 0; i < 8; ++i) { out.push_back(static_cast<uint8_t>((value >> (i * 8)) & 0xFF)); }
}

// Reads a little-endian uint64_t from the given byte pointer.
inline uint64_t ReadU64(const uint8_t *p) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) { value |= static_cast<uint64_t>(p[i]) << (i * 8); }
    return value;
}
} // namespace detail

// Global random seed entry point.
void ManualSeed(uint64_t seed);

// Casts the given Generator (or the device default when none is supplied) to a
// concrete backend implementation, validating that the device type matches.
template <typename T> T *GetGeneratorOrDefault(const std::optional<Generator> &gen, const Device &device) {
    const Generator &chosen = (gen.has_value() && gen->Defined()) ? *gen : detail::GetDefaultGenerator(device);
    CHECK(chosen.Defined()) << "Generator with undefined implementation is not allowed";
    CHECK(T::kDeviceType == chosen.GetDevice().type())
        << "Generator device type mismatch: expected " << static_cast<int>(T::kDeviceType) << " but found "
        << static_cast<int>(chosen.GetDevice().type());
    return chosen.Get<T>();
}

// Convenience factory — matches PyTorch's at::make_generator<Impl>(args...).
template <class Impl, class... Args> Generator MakeGenerator(Args &&...args) {
    return Generator(std::make_shared<Impl>(std::forward<Args>(args)...));
}

// Validates a Generator optional and casts to the concrete backend type.
// Unlike GetGeneratorOrDefault, this does NOT fall back to a default — the
// caller must supply a defined generator.  Matches PyTorch's check_generator.
template <typename T> T *CheckGenerator(std::optional<Generator> gen) {
    CHECK(gen.has_value()) << "Expected a Generator but received std::nullopt";
    CHECK(gen->Defined()) << "Generator with undefined implementation is not allowed";
    CHECK(T::kDeviceType == gen->GetDevice().type())
        << "Generator device type mismatch: expected " << static_cast<int>(T::kDeviceType) << " but found "
        << static_cast<int>(gen->GetDevice().type());
    return gen->Get<T>();
}

} // namespace infini_train::core