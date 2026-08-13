#include "infini_train/include/generator.h"
#include "infini_train/src/generator_internal.h"
#include "infini_train/src/random_utils.h"

#include <bit>
#include <charconv>
#include <cmath>
#include <limits>
#include <locale>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

#include "glog/logging.h"

namespace infini_train {
namespace {
constexpr char kCPUStateHeader[] = "InfiniTrain CPU Generator";
constexpr char kCUDAStateHeader[] = "InfiniTrain CUDA Generator";
constexpr float kTwoPi = 6.283185307179586476925286766559f;

// Keep the raw word-to-unit-interval mapping explicit: std::uniform_real_distribution does not specify an identical
// value sequence across standard library implementations.
float UintToUnitFloat(uint32_t value) { return static_cast<float>(value >> 8) * 0x1.0p-24f; }

std::pair<float, float> BoxMullerPair(uint32_t first, uint32_t second) {
    const float uniform1 = 1.0f - UintToUnitFloat(first);
    const float uniform2 = UintToUnitFloat(second);
    const float radius = std::sqrt(-2.0f * std::log(uniform1));
    const float angle = kTwoPi * uniform2;
    return {radius * std::cos(angle), radius * std::sin(angle)};
}

void SeedEngine(std::mt19937 &engine, uint64_t seed) {
    std::seed_seq seed_seq{static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32)};
    engine.seed(seed_seq);
}

uint64_t MakeRandomSeed() {
    std::random_device rd;
    return (static_cast<uint64_t>(rd()) << 32) ^ rd();
}

std::string ToString(const std::vector<uint8_t> &state) { return std::string(state.begin(), state.end()); }

std::vector<uint8_t> ToBytes(const std::string &state) { return std::vector<uint8_t>(state.begin(), state.end()); }

template <typename T> T ParseStateIntegerToken(const std::string &token, const char *error_message) {
    T value = 0;
    const auto [ptr, error] = std::from_chars(token.data(), token.data() + token.size(), value);
    CHECK(error == std::errc{} && ptr == token.data() + token.size()) << error_message;
    return value;
}

template <typename T> T ParseStateIntegerLine(std::istringstream &iss, const char *error_message) {
    std::string line;
    CHECK(std::getline(iss, line)) << error_message;
    return ParseStateIntegerToken<T>(line, error_message);
}

void ValidateCPUEngineState(const std::string &serialized) {
    std::istringstream iss(serialized);
    iss.imbue(std::locale::classic());
    bool any_nonzero_word = false;
    for (size_t i = 0; i < std::mt19937::state_size; ++i) {
        std::string token;
        CHECK(iss >> token) << "Invalid CPU generator engine state: missing state word";
        const uint64_t word = ParseStateIntegerToken<uint64_t>(token, "Invalid CPU generator engine word in state");
        CHECK_LE(word, std::numeric_limits<uint32_t>::max()) << "Invalid CPU generator engine word in state";
        any_nonzero_word = any_nonzero_word || word != 0;
    }

    std::string position_token;
    CHECK(iss >> position_token) << "Invalid CPU generator engine state: missing position";
    const uint64_t position
        = ParseStateIntegerToken<uint64_t>(position_token, "Invalid CPU generator engine position in state");
    CHECK_LE(position, std::mt19937::state_size) << "Invalid CPU generator engine position in state";

    std::string trailing_token;
    CHECK(!(iss >> trailing_token)) << "Invalid CPU generator state: trailing data";
    CHECK(any_nonzero_word) << "Invalid CPU generator engine state: all-zero state";
}

void CheckNoTrailingStateData(std::istringstream &iss, const char *generator_name) {
    CHECK_EQ(iss.peek(), std::char_traits<char>::eof())
        << "Invalid " << generator_name << " generator state: trailing data";
}

std::mutex &DefaultGeneratorMutex() {
    static std::mutex mutex;
    return mutex;
}

uint64_t ProcessDefaultSeed() {
    static const uint64_t seed = MakeRandomSeed();
    return seed;
}

uint64_t &DefaultGeneratorSeed() {
    static uint64_t seed = ProcessDefaultSeed();
    return seed;
}

std::unordered_map<int8_t, std::shared_ptr<Generator>> &DefaultCUDAGenerators() {
    static std::unordered_map<int8_t, std::shared_ptr<Generator>> generators;
    return generators;
}
} // namespace

class CPUGeneratorImpl : public GeneratorImpl {
public:
    explicit CPUGeneratorImpl(uint64_t seed);

    void ManualSeed(uint64_t seed) override;
    uint64_t Seed() override;
    uint64_t InitialSeed() const override;
    std::vector<uint8_t> GetState() const override;
    void SetState(const std::vector<uint8_t> &state) override;
    Device GetDevice() const override;

    void FillUniform(float *data, size_t num_elements, float from, float to);
    void FillNormal(float *data, size_t num_elements, float mean, float stddev);

private:
    mutable std::mutex mutex_;
    uint64_t initial_seed_ = 0;
    std::mt19937 engine_;
    // Box-Muller produces two samples at a time. Preserve the second so the
    // stream does not depend on how callers partition output tensors.
    bool has_next_normal_sample_ = false;
    float next_normal_sample_ = 0.0f;
};

class CUDAGeneratorImpl : public GeneratorImpl {
public:
    CUDAGeneratorImpl(int8_t device_index, uint64_t seed);

    void ManualSeed(uint64_t seed) override;
    uint64_t Seed() override;
    uint64_t InitialSeed() const override;
    std::vector<uint8_t> GetState() const override;
    void SetState(const std::vector<uint8_t> &state) override;
    Device GetDevice() const override;
    std::pair<uint64_t, uint64_t> ReserveRandomOffset(uint64_t increment);

private:
    mutable std::mutex mutex_;
    Device device_;
    uint64_t initial_seed_ = 0;
    uint64_t offset_ = 0;
};

Generator::Generator(std::shared_ptr<GeneratorImpl> impl) : impl_(std::move(impl)) { CHECK(impl_ != nullptr); }

void Generator::ManualSeed(uint64_t seed) { impl_->ManualSeed(seed); }

uint64_t Generator::Seed() { return impl_->Seed(); }

uint64_t Generator::InitialSeed() const { return impl_->InitialSeed(); }

std::vector<uint8_t> Generator::GetState() const { return impl_->GetState(); }

void Generator::SetState(const std::vector<uint8_t> &state) { impl_->SetState(state); }

Device Generator::GetDevice() const { return impl_->GetDevice(); }

std::pair<uint64_t, uint64_t>
detail::GeneratorAccessor::ReserveCUDARandomOffset(const std::shared_ptr<Generator> &generator, uint64_t increment) {
    CHECK(generator != nullptr);
    auto *cuda_generator = dynamic_cast<CUDAGeneratorImpl *>(generator->impl_.get());
    CHECK(cuda_generator != nullptr) << "CUDA random offset requires a CUDA generator";
    return cuda_generator->ReserveRandomOffset(increment);
}

void detail::GeneratorAccessor::FillCPUUniform(const std::shared_ptr<Generator> &generator, float *data,
                                               size_t num_elements, float from, float to) {
    CHECK(generator != nullptr);
    auto *cpu_generator = dynamic_cast<CPUGeneratorImpl *>(generator->impl_.get());
    CHECK(cpu_generator != nullptr) << "CPU random fill requires a CPU generator";
    cpu_generator->FillUniform(data, num_elements, from, to);
}

void detail::GeneratorAccessor::FillCPUNormal(const std::shared_ptr<Generator> &generator, float *data,
                                              size_t num_elements, float mean, float stddev) {
    CHECK(generator != nullptr);
    auto *cpu_generator = dynamic_cast<CPUGeneratorImpl *>(generator->impl_.get());
    CHECK(cpu_generator != nullptr) << "CPU random fill requires a CPU generator";
    cpu_generator->FillNormal(data, num_elements, mean, stddev);
}

CPUGeneratorImpl::CPUGeneratorImpl(uint64_t seed) { ManualSeed(seed); }

void CPUGeneratorImpl::ManualSeed(uint64_t seed) {
    std::lock_guard<std::mutex> lock(mutex_);
    initial_seed_ = seed;
    SeedEngine(engine_, seed);
    has_next_normal_sample_ = false;
    next_normal_sample_ = 0.0f;
}

uint64_t CPUGeneratorImpl::Seed() {
    const uint64_t seed = MakeRandomSeed();
    ManualSeed(seed);
    return seed;
}

uint64_t CPUGeneratorImpl::InitialSeed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return initial_seed_;
}

std::vector<uint8_t> CPUGeneratorImpl::GetState() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream oss;
    oss.imbue(std::locale::classic());
    oss << kCPUStateHeader << "\n"
        << initial_seed_ << "\n"
        << has_next_normal_sample_ << "\n"
        << std::bit_cast<uint32_t>(next_normal_sample_) << "\n"
        << engine_;
    return ToBytes(oss.str());
}

void CPUGeneratorImpl::SetState(const std::vector<uint8_t> &state) {
    const std::string serialized = ToString(state);
    std::istringstream iss(serialized);
    iss.imbue(std::locale::classic());

    std::string header;
    std::mt19937 engine;
    std::getline(iss, header);
    CHECK_EQ(header, kCPUStateHeader) << "Invalid CPU generator state header";
    const uint64_t seed = ParseStateIntegerLine<uint64_t>(iss, "Invalid CPU generator seed in state");
    const uint32_t has_next_normal
        = ParseStateIntegerLine<uint32_t>(iss, "Invalid CPU generator normal cache flag in state");
    CHECK_LE(has_next_normal, 1U) << "Invalid CPU generator normal cache flag in state";
    const uint32_t next_normal_bits
        = ParseStateIntegerLine<uint32_t>(iss, "Invalid CPU generator normal cache value in state");
    CHECK(has_next_normal != 0 || next_normal_bits == 0) << "Invalid unused CPU generator normal cache value";
    const float next_normal = std::bit_cast<float>(next_normal_bits);
    CHECK(has_next_normal == 0 || std::isfinite(next_normal)) << "Invalid CPU generator normal cache value in state";
    std::ostringstream engine_state_stream;
    engine_state_stream << iss.rdbuf();
    const std::string engine_state = engine_state_stream.str();
    ValidateCPUEngineState(engine_state);
    std::istringstream engine_iss(engine_state);
    engine_iss.imbue(std::locale::classic());
    CHECK(engine_iss >> engine) << "Invalid CPU generator engine state";
    CheckNoTrailingStateData(engine_iss, "CPU");

    std::lock_guard<std::mutex> lock(mutex_);
    initial_seed_ = seed;
    engine_ = engine;
    has_next_normal_sample_ = has_next_normal != 0;
    next_normal_sample_ = next_normal;
}

Device CPUGeneratorImpl::GetDevice() const { return Device(); }

void CPUGeneratorImpl::FillUniform(float *data, size_t num_elements, float from, float to) {
    CHECK(data != nullptr || num_elements == 0);
    detail::CheckUniformBounds(from, to);
    std::lock_guard<std::mutex> lock(mutex_);

    const float range = to - from;
    for (size_t i = 0; i < num_elements; ++i) {
        const float value = from + range * UintToUnitFloat(engine_());
        // Preserve the half-open interval if the final float rounding reaches the upper bound.
        data[i] = value == to ? from : value;
    }
}

void CPUGeneratorImpl::FillNormal(float *data, size_t num_elements, float mean, float stddev) {
    CHECK(data != nullptr || num_elements == 0);
    CHECK_GE(stddev, 0.0f);
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t i = 0; i < num_elements; ++i) {
        float normal;
        if (has_next_normal_sample_) {
            normal = next_normal_sample_;
            has_next_normal_sample_ = false;
            next_normal_sample_ = 0.0f;
        } else {
            const uint32_t first_random = engine_();
            const uint32_t second_random = engine_();
            const auto [first, second] = BoxMullerPair(first_random, second_random);
            normal = first;
            has_next_normal_sample_ = true;
            next_normal_sample_ = second;
        }
        data[i] = mean + stddev * normal;
    }
}

CUDAGeneratorImpl::CUDAGeneratorImpl(int8_t device_index, uint64_t seed)
    : device_(Device::DeviceType::kCUDA, device_index) {
    CHECK_GE(device_index, 0);
    ManualSeed(seed);
}

void CUDAGeneratorImpl::ManualSeed(uint64_t seed) {
    std::lock_guard<std::mutex> lock(mutex_);
    initial_seed_ = seed;
    offset_ = 0;
}

uint64_t CUDAGeneratorImpl::Seed() {
    const uint64_t seed = MakeRandomSeed();
    ManualSeed(seed);
    return seed;
}

uint64_t CUDAGeneratorImpl::InitialSeed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return initial_seed_;
}

std::vector<uint8_t> CUDAGeneratorImpl::GetState() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream oss;
    oss.imbue(std::locale::classic());
    oss << kCUDAStateHeader << "\n" << initial_seed_ << "\n" << offset_ << "\n";
    return ToBytes(oss.str());
}

void CUDAGeneratorImpl::SetState(const std::vector<uint8_t> &state) {
    const std::string serialized = ToString(state);
    std::istringstream iss(serialized);
    iss.imbue(std::locale::classic());

    std::string header;
    std::getline(iss, header);
    CHECK_EQ(header, kCUDAStateHeader) << "Invalid CUDA generator state header";
    const uint64_t seed = ParseStateIntegerLine<uint64_t>(iss, "Invalid CUDA generator seed in state");
    const uint64_t offset = ParseStateIntegerLine<uint64_t>(iss, "Invalid CUDA generator offset in state");
    CheckNoTrailingStateData(iss, "CUDA");

    std::lock_guard<std::mutex> lock(mutex_);
    initial_seed_ = seed;
    offset_ = offset;
}

Device CUDAGeneratorImpl::GetDevice() const { return device_; }

std::pair<uint64_t, uint64_t> CUDAGeneratorImpl::ReserveRandomOffset(uint64_t increment) {
    std::lock_guard<std::mutex> lock(mutex_);
    CHECK_LE(increment, std::numeric_limits<uint64_t>::max() - offset_) << "CUDA generator offset overflow";
    const uint64_t offset = offset_;
    // offset_ addresses individual 32-bit words in one global Philox stream.
    // It is not cuRAND's per-thread offset, so arbitrary exact increments are safe.
    offset_ += increment;
    return {initial_seed_, offset};
}

std::shared_ptr<Generator> MakeCPUGenerator(uint64_t seed) {
    return std::shared_ptr<Generator>(new Generator(std::make_shared<CPUGeneratorImpl>(seed)));
}

std::shared_ptr<Generator> MakeCUDAGenerator(int8_t device_index, uint64_t seed) {
    return std::shared_ptr<Generator>(new Generator(std::make_shared<CUDAGeneratorImpl>(device_index, seed)));
}

std::shared_ptr<Generator> GetDefaultCPUGenerator() {
    static auto generator = MakeCPUGenerator(ProcessDefaultSeed());
    return generator;
}

std::shared_ptr<Generator> GetDefaultCUDAGenerator(int8_t device_index) {
    CHECK_GE(device_index, 0);
    std::lock_guard<std::mutex> lock(DefaultGeneratorMutex());
    auto &generators = DefaultCUDAGenerators();
    auto it = generators.find(device_index);
    if (it != generators.end()) {
        return it->second;
    }
    auto generator = MakeCUDAGenerator(device_index, DefaultGeneratorSeed());
    generators.emplace(device_index, generator);
    return generator;
}

std::shared_ptr<Generator> GetDefaultGenerator(const Device &device) {
    if (device.IsCPU()) {
        return GetDefaultCPUGenerator();
    }
    if (device.IsCUDA()) {
        return GetDefaultCUDAGenerator(device.index());
    }
    LOG(FATAL) << "Unsupported default Generator device: " << device;
    return nullptr;
}

void ManualSeed(uint64_t seed) { ManualSeedAll(seed); }

void ManualSeedAll(uint64_t seed) {
    std::lock_guard<std::mutex> lock(DefaultGeneratorMutex());
    GetDefaultCPUGenerator()->ManualSeed(seed);
    DefaultGeneratorSeed() = seed;
    for (auto &[_, generator] : DefaultCUDAGenerators()) { generator->ManualSeed(seed); }
}

} // namespace infini_train
