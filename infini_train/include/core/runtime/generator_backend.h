#pragma once

// Internal Generator backend registration interface. Application code should include generator.h.

#include <cstdint>
#include <memory>
#include <unordered_map>

#include "infini_train/include/common/common.h"
#include "infini_train/include/device.h"
#include "infini_train/include/generator.h"

namespace infini_train {

class GeneratorBackend {
public:
    virtual ~GeneratorBackend() = default;

    virtual Device::DeviceType Type() const = 0;
    virtual Generator Create(const Device &device, uint64_t seed) = 0;
    virtual const Generator &GetDefault(const Device &device) = 0;
    virtual void ManualSeedAll(uint64_t seed) = 0;
};

// Registers one backend per device type during static initialization.
class GeneratorBackendRegistry {
public:
    static GeneratorBackendRegistry &Instance();

    void Register(Device::DeviceType type, std::unique_ptr<GeneratorBackend> backend);
    // Throws if no backend is registered for the requested device type.
    GeneratorBackend &Get(Device::DeviceType type) const;
    void ManualSeedAll(uint64_t seed) const;

private:
    GeneratorBackendRegistry() = default;
    GeneratorBackendRegistry(const GeneratorBackendRegistry &) = delete;
    GeneratorBackendRegistry &operator=(const GeneratorBackendRegistry &) = delete;

    std::unordered_map<Device::DeviceType, std::unique_ptr<GeneratorBackend>> backends_;
};

} // namespace infini_train

// CAT expands __COUNTER__ before token pasting so each registration has a unique name.
#define INFINI_TRAIN_REGISTER_GENERATOR_BACKEND(device_type, class_impl)                                               \
    [[maybe_unused]] static const bool CAT(infini_train_generator_backend_registered_, __COUNTER__) = []() {           \
        infini_train::GeneratorBackendRegistry::Instance().Register(device_type, std::make_unique<class_impl>());      \
        return true;                                                                                                   \
    }();
