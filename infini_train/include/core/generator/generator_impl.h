#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

#include "infini_train/include/device.h"

namespace infini_train::core {

class GeneratorImpl {
public:
    explicit GeneratorImpl(Device device) : device_(device) {}
    virtual ~GeneratorImpl() = default;

    GeneratorImpl(const GeneratorImpl &) = delete;
    GeneratorImpl &operator=(const GeneratorImpl &) = delete;

    virtual void SetCurrentSeed(uint64_t seed) = 0;
    virtual uint64_t CurrentSeed() const = 0;
    virtual uint64_t InitialSeed() const = 0;
    virtual std::vector<uint8_t> GetState() const = 0;
    virtual void SetState(const std::vector<uint8_t> &state) = 0;
    virtual uint32_t StateMagic() const = 0;

    virtual uint64_t Seed();

    Device device() const { return device_; }
    std::mutex &mutex() { return mutex_; }

protected:
    virtual void Reseed(uint64_t seed) = 0;

    Device device_;
    mutable std::mutex mutex_;
};

using GeneratorImplFactory = std::function<std::shared_ptr<GeneratorImpl>(int8_t /*device_index*/)>;

inline constexpr int kMaxGpus = 8;

class GeneratorImplRegistry {
public:
    static GeneratorImplRegistry &Instance();

    void Register(Device::DeviceType type, GeneratorImplFactory factory);

    std::shared_ptr<GeneratorImpl> Create(Device device) const;

    std::shared_ptr<GeneratorImpl> Default(Device device);

    void ResetAllSeeds(uint64_t seed);

    void ResetCudaSeeds(uint64_t seed);

private:
    GeneratorImplRegistry() = default;

    uint64_t LastUserSeedOrRandom();

    std::unordered_map<Device::DeviceType, GeneratorImplFactory> factories_;
    std::shared_ptr<GeneratorImpl> default_cpu_;
    std::array<std::shared_ptr<GeneratorImpl>, kMaxGpus> default_cuda_{};
    std::once_flag default_cpu_once_;
    std::array<std::once_flag, kMaxGpus> default_cuda_once_{};
    std::atomic<bool> cpu_initialized_{false};
    std::array<std::atomic<bool>, kMaxGpus> cuda_initialized_{};
    std::optional<uint64_t> last_user_seed_;
    mutable std::mutex registry_mutex_;
};

} // namespace infini_train::core

#define INFINI_TRAIN_REGISTER_GENERATOR_IMPL(device_type, FactoryFn)                                                   \
    namespace {                                                                                                        \
    struct GenReg_##device_type {                                                                                      \
        GenReg_##device_type() {                                                                                       \
            ::infini_train::core::GeneratorImplRegistry::Instance().Register(                                          \
                ::infini_train::Device::DeviceType::device_type, FactoryFn);                                           \
        }                                                                                                              \
    };                                                                                                                 \
    static GenReg_##device_type _gen_reg_##device_type;                                                                \
    } // namespace
