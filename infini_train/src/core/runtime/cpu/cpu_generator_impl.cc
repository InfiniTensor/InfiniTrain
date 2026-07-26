#include "infini_train/src/core/runtime/cpu/cpu_generator_impl.h"

#include <cstring>
#include <mutex>
#include <random>
#include <sstream>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace infini_train::core::cpu {
namespace {

// Backend tag used to reject states from other generator implementations.
constexpr char kCPUStateMagic[] = "ITRNGCPU";
constexpr size_t kStateMagicSize = sizeof(kCPUStateMagic) - 1;

constexpr size_t kStateFooterSize
    = sizeof(uint64_t) + sizeof(uint8_t) + sizeof(float) + sizeof(uint8_t) + sizeof(double);

} // namespace

// std::mt19937(seed) truncates to 32 bits, so seed both halves explicitly.
static std::mt19937 make_seeded_engine(uint64_t seed) {
    std::seed_seq seq{static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32)};
    std::mt19937 engine;
    engine.seed(seq);
    return engine;
}

static uint64_t getNonDeterministicRandom() {
    std::random_device rd;
    uint64_t val = (static_cast<uint64_t>(rd()) << 32) | rd();
    return val;
}

CPUGeneratorImpl::CPUGeneratorImpl(uint64_t seed)
    : GeneratorImpl(Device(Device::DeviceType::kCPU, 0)), engine_(make_seeded_engine(seed)), seed_(seed) {}

void CPUGeneratorImpl::set_current_seed(uint64_t seed) {
    seed_ = seed;
    next_float_normal_sample_.reset();
    next_double_normal_sample_.reset();
    engine_ = make_seeded_engine(seed);
}

uint64_t CPUGeneratorImpl::current_seed() const { return seed_; }

uint64_t CPUGeneratorImpl::seed() {
    uint64_t random_seed = getNonDeterministicRandom();
    set_current_seed(random_seed);
    return random_seed;
}

// State layout: magic, serialized engine, seed, and cached float/double normal samples.
// The std::mt19937 stream format is only portable across compatible standard-library builds.

void CPUGeneratorImpl::set_state(const Tensor &state) {
    ::infini_train::detail::check_rng_state(state);

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
    std::memcpy(&restored_seed, data + offset, sizeof(restored_seed));
    offset += sizeof(restored_seed);

    const uint8_t has_float = data[offset++];
    CHECK_LE(has_float, 1) << "Invalid CPU generator float normal cache flag";
    float restored_float = 0.0f;
    std::memcpy(&restored_float, data + offset, sizeof(restored_float));
    offset += sizeof(restored_float);

    const uint8_t has_double = data[offset++];
    CHECK_LE(has_double, 1) << "Invalid CPU generator double normal cache flag";
    double restored_double = 0.0;
    std::memcpy(&restored_double, data + offset, sizeof(restored_double));

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

    std::memcpy(data + offset, kCPUStateMagic, kStateMagicSize);
    offset += kStateMagicSize;

    std::memcpy(data + offset, engine_str.data(), engine_size);
    offset += engine_size;

    std::memcpy(data + offset, &seed_, sizeof(seed_));
    offset += sizeof(seed_);

    bool has_float = next_float_normal_sample_.has_value();
    data[offset++] = has_float ? 1 : 0;
    float float_val = has_float ? *next_float_normal_sample_ : 0.0f;
    std::memcpy(data + offset, &float_val, sizeof(float_val));
    offset += sizeof(float_val);

    bool has_double = next_double_normal_sample_.has_value();
    data[offset++] = has_double ? 1 : 0;
    double double_val = has_double ? *next_double_normal_sample_ : 0.0;
    std::memcpy(data + offset, &double_val, sizeof(double_val));

    return state_tensor;
}

uint32_t CPUGeneratorImpl::random() { return engine_(); }

uint64_t CPUGeneratorImpl::random64() {
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

std::shared_ptr<CPUGeneratorImpl> CPUGeneratorImpl::clone() const {
    return std::shared_ptr<CPUGeneratorImpl>(clone_impl());
}

CPUGeneratorImpl *CPUGeneratorImpl::clone_impl() const {
    auto *clone = new CPUGeneratorImpl(seed_);
    clone->set_engine(engine_);
    clone->set_next_float_normal_sample(next_float_normal_sample_);
    clone->set_next_double_normal_sample(next_double_normal_sample_);
    return clone;
}

void CPUGeneratorImpl::set_engine(std::mt19937 engine) { engine_ = std::move(engine); }

Device::DeviceType CPUGeneratorImpl::device_type() { return Device::DeviceType::kCPU; }

} // namespace infini_train::core::cpu

namespace infini_train::core::cpu {

const Generator &getDefaultCPUGenerator() {
    static auto default_gen = createCPUGenerator(getNonDeterministicRandom());
    return default_gen;
}

Generator createCPUGenerator(uint64_t seed) { return make_generator<CPUGeneratorImpl>(seed); }

void manual_seed(uint64_t seed) {
    const auto &default_gen = getDefaultCPUGenerator();
    std::lock_guard<std::mutex> lock(default_gen.mutex());
    default_gen.set_current_seed(seed);
}

} // namespace infini_train::core::cpu
