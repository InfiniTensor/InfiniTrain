#include "infini_train/include/core/generator.h"

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>

#include "glog/logging.h"

#include "infini_train/include/core/cpu_generator.h"
#include "infini_train/include/device.h"

#ifdef USE_CUDA
#include "infini_train/include/core/cuda_generator.h"
#endif

namespace infini_train::core {
namespace detail {
namespace {
// Guards the lazily-created default generators.
std::mutex &DefaultGeneratorMutex() {
    static std::mutex mutex;
    return mutex;
}

std::optional<uint64_t> &DefaultManualSeed() {
    static std::optional<uint64_t> seed;
    return seed;
}

// Process-wide default CPU generator (single shared random state source).
Generator &MutableDefaultCPUGenerator() {
    static Generator generator(std::make_shared<CPUGeneratorImpl>());
    return generator;
}

#ifdef USE_CUDA
// Per-device default CUDA generators. Different devices are fully independent.
std::map<int8_t, Generator> &MutableDefaultCUDAGenerators() {
    static std::map<int8_t, Generator> generators;
    return generators;
}

Generator &MutableDefaultCUDAGenerator(int8_t device_index) {
    auto &generators = MutableDefaultCUDAGenerators();
    auto it = generators.find(device_index);
    if (it == generators.end()) {
        auto manual_seed = DefaultManualSeed();
        Generator generator = manual_seed.has_value()
                                ? Generator(std::make_shared<CUDAGeneratorImpl>(device_index, *manual_seed))
                                : Generator(std::make_shared<CUDAGeneratorImpl>(device_index));
        it = generators.emplace(device_index, std::move(generator)).first;
    }
    return it->second;
}
#endif
} // namespace

const Generator &DefaultCPUGenerator() {
    std::lock_guard<std::mutex> lock(DefaultGeneratorMutex());
    return MutableDefaultCPUGenerator();
}

const Generator &DefaultCUDAGenerator(int8_t device_index) {
#ifdef USE_CUDA
    std::lock_guard<std::mutex> lock(DefaultGeneratorMutex());
    return MutableDefaultCUDAGenerator(device_index);
#else
    LOG(FATAL) << "CUDA default generator requested but the framework was built without CUDA support";
    return DefaultCPUGenerator();
#endif
}

const Generator &GetDefaultGenerator(const Device &device) {
    switch (device.type()) {
    case Device::DeviceType::kCPU:
        return DefaultCPUGenerator();
    case Device::DeviceType::kCUDA:
        return DefaultCUDAGenerator(device.index());
    default:
        LOG(FATAL) << "No default generator for device type " << static_cast<int>(device.type());
        return DefaultCPUGenerator();
    }
}
} // namespace detail

void ManualSeed(uint64_t seed) {
    // Lock ordering: DefaultGeneratorMutex (map guard) is always acquired first;
    // Generator::SetCurrentSeed then acquires impl->mutex_ (impl guard).
    // Callers must not hold impl->mutex_ when calling ManualSeed() to avoid
    // potential deadlock with threads that draw from the default generator.
    std::lock_guard<std::mutex> lock(detail::DefaultGeneratorMutex());
    detail::DefaultManualSeed() = seed;
    detail::MutableDefaultCPUGenerator().SetCurrentSeed(seed);
#ifdef USE_CUDA
    // Seed already materialized CUDA generators. Future CUDA defaults read
    // DefaultManualSeed() when they are lazily created.
    for (auto &[index, generator] : detail::MutableDefaultCUDAGenerators()) { generator.SetCurrentSeed(seed); }
#endif
}

} // namespace infini_train::core
