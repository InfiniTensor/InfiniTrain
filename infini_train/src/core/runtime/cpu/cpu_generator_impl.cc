#include "infini_train/src/core/runtime/cpu/cpu_generator_impl.h"

#include <cstddef>
#include <cstring>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/core/runtime/generator_backend.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/generator_impl.h"
#include "infini_train/include/tensor.h"

namespace infini_train::core::cpu {
namespace {

// Backend tag used to reject states from other generator implementations.
constexpr char kCPUStateMagic[] = "ITRNGCPU";
constexpr std::size_t kStateMagicSize = sizeof(kCPUStateMagic) - 1;

// Fixed footer after the variable-length mt19937 stream. Cache flags are
// serialized as uint8_t; absent cache values are serialized as zero.
constexpr std::size_t kSerializedSeedSize = sizeof(uint64_t);
constexpr std::size_t kSerializedFloatCacheFlagSize = sizeof(uint8_t);
constexpr std::size_t kSerializedFloatCacheValueSize = sizeof(float);
constexpr std::size_t kSerializedDoubleCacheFlagSize = sizeof(uint8_t);
constexpr std::size_t kSerializedDoubleCacheValueSize = sizeof(double);
constexpr std::size_t kStateFooterSize = kSerializedSeedSize + kSerializedFloatCacheFlagSize
                                       + kSerializedFloatCacheValueSize + kSerializedDoubleCacheFlagSize
                                       + kSerializedDoubleCacheValueSize;

// Seed through seed_seq using both 32-bit halves of the seed. This produces a
// different sequence from direct std::mt19937(seed) initialization, even when
// the seed fits in 32 bits.
std::mt19937 MakeSeededEngine(uint64_t seed) {
    std::seed_seq seq{static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32)};
    std::mt19937 engine;
    engine.seed(seq);
    return engine;
}

uint64_t GenerateNonDeterministicSeed() {
    std::random_device rd;
    uint64_t val = (static_cast<uint64_t>(rd()) << 32) | rd();
    return val;
}

// Copies state bytes and advances offset; data and value must reference valid
// buffers of data_size and value_size bytes, and offset must be a valid pointer.
void WriteBytes(uint8_t *data, std::size_t data_size, std::size_t *offset, const void *value, std::size_t value_size) {
    CHECK_LE(*offset, data_size) << "CPU generator state write offset out of bounds";
    CHECK_LE(value_size, data_size - *offset) << "CPU generator state write exceeds buffer";
    if (value_size != 0) {
        std::memcpy(data + *offset, value, value_size);
    }
    *offset += value_size;
}

void ReadBytes(const uint8_t *data, std::size_t data_size, std::size_t *offset, void *value, std::size_t value_size) {
    CHECK_LE(*offset, data_size) << "CPU generator state read offset out of bounds";
    CHECK_LE(value_size, data_size - *offset) << "CPU generator state read exceeds buffer";
    if (value_size != 0) {
        std::memcpy(value, data + *offset, value_size);
    }
    *offset += value_size;
}

} // namespace

CPUGeneratorImpl::CPUGeneratorImpl(uint64_t seed)
    : GeneratorImpl(Device(Device::DeviceType::kCPU, 0)), engine_(MakeSeededEngine(seed)), seed_(seed) {}

void CPUGeneratorImpl::set_current_seed(uint64_t seed) {
    seed_ = seed;
    next_float_normal_sample_.reset();
    next_double_normal_sample_.reset();
    engine_ = MakeSeededEngine(seed);
}

uint64_t CPUGeneratorImpl::current_seed() const { return seed_; }

uint64_t CPUGeneratorImpl::Seed() {
    uint64_t random_seed = GenerateNonDeterministicSeed();
    set_current_seed(random_seed);
    return random_seed;
}

// State layout: magic, serialized engine, seed, and cached float/double normal samples.
// The std::mt19937 stream format is only portable across compatible standard-library builds.

void CPUGeneratorImpl::set_state(const Tensor &state) {
    ::infini_train::detail::CheckRngState(state);

    const size_t data_size = state.SizeInBytes();
    CHECK_GT(data_size, kStateMagicSize + kStateFooterSize) << "CPU generator state is too small";

    const uint8_t *data = static_cast<const uint8_t *>(state.DataPtr());
    CHECK_EQ(std::memcmp(data, kCPUStateMagic, kStateMagicSize), 0)
        << "Invalid RNG state: not a CPU generator state (backend magic mismatch)";

    const size_t engine_size = data_size - kStateMagicSize - kStateFooterSize;
    std::string engine_str(reinterpret_cast<const char *>(data + kStateMagicSize), engine_size);

    std::istringstream iss(engine_str);
    std::mt19937 restored_engine;
    iss >> restored_engine;
    CHECK(!iss.fail()) << "Invalid CPU generator engine state";
    iss >> std::ws;
    CHECK(iss.eof()) << "Invalid trailing bytes in CPU generator engine state";

    size_t offset = kStateMagicSize + engine_size;
    uint64_t restored_seed = 0;
    ReadBytes(data, data_size, &offset, &restored_seed, kSerializedSeedSize);
    uint8_t has_float = 0;
    ReadBytes(data, data_size, &offset, &has_float, kSerializedFloatCacheFlagSize);
    CHECK_LE(has_float, 1) << "Invalid CPU generator float normal cache flag";
    float restored_float = 0.0f;
    ReadBytes(data, data_size, &offset, &restored_float, kSerializedFloatCacheValueSize);
    uint8_t has_double = 0;
    ReadBytes(data, data_size, &offset, &has_double, kSerializedDoubleCacheFlagSize);
    CHECK_LE(has_double, 1) << "Invalid CPU generator double normal cache flag";
    double restored_double = 0.0;
    ReadBytes(data, data_size, &offset, &restored_double, kSerializedDoubleCacheValueSize);
    CHECK_EQ(offset, data_size) << "CPU generator state size mismatch";

    // Do not change the generator until the complete state has been validated.
    engine_ = restored_engine;
    seed_ = restored_seed;
    next_float_normal_sample_ = has_float ? std::optional<float>(restored_float) : std::nullopt;
    next_double_normal_sample_ = has_double ? std::optional<double>(restored_double) : std::nullopt;
}

std::shared_ptr<Tensor> CPUGeneratorImpl::get_state() const {
    std::ostringstream oss;
    oss << engine_;
    std::string engine_str = oss.str();

    const size_t engine_size = engine_str.size();
    const size_t total_size = kStateMagicSize + engine_size + kStateFooterSize;

    auto state_tensor = std::make_shared<Tensor>(std::vector<int64_t>{static_cast<int64_t>(total_size)},
                                                 DataType::kUINT8, Device(Device::DeviceType::kCPU, 0));

    uint8_t *data = static_cast<uint8_t *>(state_tensor->DataPtr());
    size_t offset = 0;

    const uint8_t has_float = next_float_normal_sample_.has_value() ? 1 : 0;
    const float float_val = has_float ? *next_float_normal_sample_ : 0.0f;
    const uint8_t has_double = next_double_normal_sample_.has_value() ? 1 : 0;
    const double double_val = has_double ? *next_double_normal_sample_ : 0.0;

    WriteBytes(data, total_size, &offset, kCPUStateMagic, kStateMagicSize);
    WriteBytes(data, total_size, &offset, engine_str.data(), engine_size);
    WriteBytes(data, total_size, &offset, &seed_, kSerializedSeedSize);
    WriteBytes(data, total_size, &offset, &has_float, kSerializedFloatCacheFlagSize);
    WriteBytes(data, total_size, &offset, &float_val, kSerializedFloatCacheValueSize);
    WriteBytes(data, total_size, &offset, &has_double, kSerializedDoubleCacheFlagSize);
    WriteBytes(data, total_size, &offset, &double_val, kSerializedDoubleCacheValueSize);
    CHECK_EQ(offset, total_size) << "CPU generator state size mismatch";

    return state_tensor;
}

uint32_t CPUGeneratorImpl::Random() { return engine_(); }

uint64_t CPUGeneratorImpl::Random64() {
    uint32_t hi = engine_();
    uint32_t lo = engine_();
    return (static_cast<uint64_t>(hi) << 32) | lo;
}

std::optional<float> CPUGeneratorImpl::next_float_normal_sample() const { return next_float_normal_sample_; }

std::optional<double> CPUGeneratorImpl::next_double_normal_sample() const { return next_double_normal_sample_; }

void CPUGeneratorImpl::set_next_float_normal_sample(std::optional<float> randn) { next_float_normal_sample_ = randn; }

void CPUGeneratorImpl::set_next_double_normal_sample(std::optional<double> randn) {
    next_double_normal_sample_ = randn;
}

std::shared_ptr<CPUGeneratorImpl> CPUGeneratorImpl::Clone() const {
    return std::shared_ptr<CPUGeneratorImpl>(CloneImpl());
}

CPUGeneratorImpl *CPUGeneratorImpl::CloneImpl() const {
    auto *clone = new CPUGeneratorImpl(seed_);
    clone->set_engine(engine_);
    clone->set_next_float_normal_sample(next_float_normal_sample_);
    clone->set_next_double_normal_sample(next_double_normal_sample_);
    return clone;
}

void CPUGeneratorImpl::set_engine(std::mt19937 engine) { engine_ = std::move(engine); }

} // namespace infini_train::core::cpu

namespace infini_train::core::cpu {
namespace {

struct DefaultCpuGeneratorState {
    std::once_flag init_once;
    std::optional<Generator> generator;
};

DefaultCpuGeneratorState &GetDefaultCpuGeneratorState() {
    static DefaultCpuGeneratorState state;
    return state;
}

} // namespace

const Generator &GetDefaultCpuGenerator() {
    auto &state = GetDefaultCpuGeneratorState();
    std::call_once(state.init_once,
                   [&state] { state.generator.emplace(CreateCpuGenerator(GenerateNonDeterministicSeed())); });
    return *state.generator;
}

Generator CreateCpuGenerator(uint64_t seed) { return MakeGenerator<CPUGeneratorImpl>(seed); }

void ManualSeed(uint64_t seed) {
    auto &state = GetDefaultCpuGeneratorState();
    bool initialized_here = false;
    std::call_once(state.init_once, [&state, seed, &initialized_here] {
        state.generator.emplace(CreateCpuGenerator(seed));
        initialized_here = true;
    });
    if (initialized_here) {
        return;
    }
    const Generator &generator = *state.generator;
    std::lock_guard<std::mutex> lock(GeneratorAccessor::Mutex(generator));
    generator.set_current_seed(seed);
}

} // namespace infini_train::core::cpu

namespace infini_train::core::cpu {
namespace {

class CPUGeneratorBackend final : public GeneratorBackend {
public:
    Device::DeviceType Type() const override { return Device::DeviceType::kCPU; }

    Generator Create(const Device & /*device*/, uint64_t seed) override { return CreateCpuGenerator(seed); }

    const Generator &GetDefault(const Device & /*device*/) override { return GetDefaultCpuGenerator(); }

    void ManualSeedAll(uint64_t seed) override { ::infini_train::core::cpu::ManualSeed(seed); }
};

INFINI_TRAIN_REGISTER_GENERATOR_BACKEND(Device::DeviceType::kCPU, CPUGeneratorBackend);

} // namespace
} // namespace infini_train::core::cpu
