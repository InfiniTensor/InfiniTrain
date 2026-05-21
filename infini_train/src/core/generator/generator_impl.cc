#include "infini_train/include/core/generator/generator_impl.h"

#include <cstdint>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

#include "glog/logging.h"

#include "infini_train/include/generator.h"

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

std::shared_ptr<GeneratorImpl> GeneratorImplRegistry::Default(Device device) {
    if (device.IsCPU()) {
        std::call_once(default_cpu_once_, [&] {
            auto it = factories_.find(Device::DeviceType::kCPU);
            CHECK(it != factories_.end()) << "No CPU GeneratorImpl factory registered";
            default_cpu_ = it->second(0);
            default_cpu_->SetCurrentSeed(LastUserSeedOrRandom());
            cpu_initialized_.store(true, std::memory_order_release);
        });
        return default_cpu_;
    }
    const int8_t idx = device.index();
    CHECK(0 <= idx && idx < kMaxGpus) << "CUDA device index out of range: " << static_cast<int>(idx);
    std::call_once(default_cuda_once_[idx], [&] {
        auto it = factories_.find(Device::DeviceType::kCUDA);
        if (it == factories_.end()) {
            throw std::runtime_error("No CUDA GeneratorImpl factory registered (build with USE_CUDA=ON).");
        }
        default_cuda_[idx] = it->second(idx);
        default_cuda_[idx]->SetCurrentSeed(LastUserSeedOrRandom());
        cuda_initialized_[idx].store(true, std::memory_order_release);
    });
    return default_cuda_[idx];
}

void GeneratorImplRegistry::ResetAllSeeds(uint64_t seed) {
    {
        std::lock_guard<std::mutex> lk(registry_mutex_);
        last_user_seed_ = seed;
    }
    auto cpu_impl = Default(Device(Device::DeviceType::kCPU, 0));
    {
        std::lock_guard<std::mutex> lk(cpu_impl->mutex());
        cpu_impl->SetCurrentSeed(seed);
    }
    for (int i = 0; i < kMaxGpus; ++i) {
        if (!cuda_initialized_[i].load(std::memory_order_acquire)) {
            continue;
        }
        auto &impl = default_cuda_[i];
        std::lock_guard<std::mutex> lk(impl->mutex());
        impl->SetCurrentSeed(seed);
    }
}

void GeneratorImplRegistry::ResetCudaSeeds(uint64_t seed) {
    for (int i = 0; i < kMaxGpus; ++i) {
        if (!cuda_initialized_[i].load(std::memory_order_acquire)) {
            continue;
        }
        auto &impl = default_cuda_[i];
        std::lock_guard<std::mutex> lk(impl->mutex());
        impl->SetCurrentSeed(seed);
    }
}

} // namespace infini_train::core

// ---------------------------------------------------------------------------
// Generator handle (PImpl) — bound to GeneratorImplRegistry::Create().
// ---------------------------------------------------------------------------

namespace infini_train {

Generator::Generator(Device device) : impl_(core::GeneratorImplRegistry::Instance().Create(device)) {}

Generator::Generator(std::shared_ptr<core::GeneratorImpl> impl) : impl_(std::move(impl)) {}

Generator Generator::FromImpl(std::shared_ptr<core::GeneratorImpl> impl) { return Generator(std::move(impl)); }

void Generator::ManualSeed(uint64_t seed) {
    std::lock_guard<std::mutex> lk(impl_->mutex());
    impl_->SetCurrentSeed(seed);
}

uint64_t Generator::Seed() { return impl_->Seed(); }

uint64_t Generator::InitialSeed() const {
    std::lock_guard<std::mutex> lk(impl_->mutex());
    return impl_->InitialSeed();
}

std::vector<uint8_t> Generator::GetState() const {
    std::lock_guard<std::mutex> lk(impl_->mutex());
    return impl_->GetState();
}

void Generator::SetState(const std::vector<uint8_t> &state) {
    std::lock_guard<std::mutex> lk(impl_->mutex());
    impl_->SetState(state);
}

Device Generator::device() const { return impl_->device(); }

namespace {

struct DefaultHandles {
    Generator cpu{Device(Device::DeviceType::kCPU, 0)};
    std::array<Generator, core::kMaxGpus> cuda{};
    std::once_flag cpu_once;
    std::array<std::once_flag, core::kMaxGpus> cuda_once;
};

DefaultHandles &handles() {
    static DefaultHandles h;
    return h;
}

} // namespace

Generator &default_generator(Device device) {
    auto &h = handles();
    auto impl = core::GeneratorImplRegistry::Instance().Default(device);
    if (device.IsCPU()) {
        std::call_once(h.cpu_once, [&] { h.cpu = Generator::FromImpl(impl); });
        DCHECK_EQ(h.cpu.impl().get(), impl.get());
        return h.cpu;
    }
    const int8_t idx = device.index();
    CHECK(0 <= idx && idx < core::kMaxGpus);
    std::call_once(h.cuda_once[idx], [&] { h.cuda[idx] = Generator::FromImpl(impl); });
    DCHECK_EQ(h.cuda[idx].impl().get(), impl.get());
    return h.cuda[idx];
}

void manual_seed(uint64_t seed) { core::GeneratorImplRegistry::Instance().ResetAllSeeds(seed); }

void manual_seed_cuda(uint64_t seed) { core::GeneratorImplRegistry::Instance().ResetCudaSeeds(seed); }

std::shared_ptr<core::GeneratorImpl> ResolveGenerator(const std::optional<Generator> &gen, Device device) {
    if (gen.has_value()) {
        return gen->impl();
    }
    return default_generator(device).impl();
}

} // namespace infini_train
