#include "infini_train/include/core/generator/generator_impl.h"

#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>

#include "glog/logging.h"

namespace infini_train::core {

namespace {

uint64_t RandomDeviceSeed64() {
    std::random_device rd;
    const uint64_t hi = static_cast<uint64_t>(rd());
    const uint64_t lo = static_cast<uint64_t>(rd());
    return (hi << 32) ^ lo;
}

} // namespace

uint64_t GeneratorImpl::Seed() {
    std::lock_guard<std::mutex> lk(mutex_);
    const uint64_t s = RandomDeviceSeed64();
    Reseed(s);
    return CurrentSeed();
}

GeneratorImplRegistry &GeneratorImplRegistry::Instance() {
    static GeneratorImplRegistry instance;
    return instance;
}

void GeneratorImplRegistry::Register(Device::DeviceType type, GeneratorImplFactory factory) {
    std::lock_guard<std::mutex> lk(registry_mutex_);
    CHECK(factories_.emplace(type, std::move(factory)).second)
        << "GeneratorImpl already registered for device type " << static_cast<int>(type);
}

std::shared_ptr<GeneratorImpl> GeneratorImplRegistry::Create(Device device) const {
    auto it = factories_.find(device.type());
    if (it == factories_.end()) {
        throw std::runtime_error("No GeneratorImpl factory registered for device type "
                                 + std::to_string(static_cast<int>(device.type()))
                                 + " (build with USE_CUDA=ON for CUDA support).");
    }
    if (device.IsCPU()) {
        CHECK_EQ(device.index(), 0) << "CPU Generator must use device index 0";
        return it->second(0);
    }
    const int8_t idx = device.index();
    CHECK(0 <= idx && idx < kMaxGpus) << "CUDA device index out of range: " << static_cast<int>(idx);
    return it->second(idx);
}

uint64_t GeneratorImplRegistry::LastUserSeedOrRandom() {
    {
        std::lock_guard<std::mutex> lk(registry_mutex_);
        if (last_user_seed_) {
            return *last_user_seed_;
        }
    }
    return RandomDeviceSeed64();
}

std::shared_ptr<GeneratorImpl> GeneratorImplRegistry::Default(Device /*device*/) {
    LOG(FATAL) << "GeneratorImplRegistry::Default() not implemented yet (Task 5)";
    return nullptr;
}

void GeneratorImplRegistry::ResetAllSeeds(uint64_t /*seed*/) {
    LOG(FATAL) << "GeneratorImplRegistry::ResetAllSeeds() not implemented yet (Task 5)";
}

void GeneratorImplRegistry::ResetCudaSeeds(uint64_t /*seed*/) {
    LOG(FATAL) << "GeneratorImplRegistry::ResetCudaSeeds() not implemented yet (Task 5)";
}

} // namespace infini_train::core
