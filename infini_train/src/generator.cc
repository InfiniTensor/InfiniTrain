#include "infini_train/include/generator.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <limits>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

#include "glog/logging.h"

namespace infini_train {
namespace {
constexpr char kCPUStateHeader[] = "InfiniTrain CPUGeneratorImpl v1";
constexpr char kCUDAStateHeader[] = "InfiniTrain CUDAGeneratorImpl v3";
constexpr float kTwoPi = 6.283185307179586476925286766559f;

struct Philox4x32State {
    uint32_t c0;
    uint32_t c1;
    uint32_t c2;
    uint32_t c3;
};

uint32_t MulHi(uint32_t a, uint32_t b) { return static_cast<uint32_t>((static_cast<uint64_t>(a) * b) >> 32); }

Philox4x32State PhiloxRound(Philox4x32State counter, uint32_t key0, uint32_t key1) {
    constexpr uint32_t kPhiloxM0 = 0xD2511F53;
    constexpr uint32_t kPhiloxM1 = 0xCD9E8D57;
    const uint32_t lo0 = counter.c0 * kPhiloxM0;
    const uint32_t hi0 = MulHi(counter.c0, kPhiloxM0);
    const uint32_t lo1 = counter.c2 * kPhiloxM1;
    const uint32_t hi1 = MulHi(counter.c2, kPhiloxM1);
    return {hi1 ^ counter.c1 ^ key0, lo1, hi0 ^ counter.c3 ^ key1, lo0};
}

Philox4x32State Philox(uint64_t seed, uint64_t subsequence) {
    constexpr uint32_t kPhiloxW0 = 0x9E3779B9;
    constexpr uint32_t kPhiloxW1 = 0xBB67AE85;
    Philox4x32State counter{static_cast<uint32_t>(subsequence), static_cast<uint32_t>(subsequence >> 32), 0, 0};
    uint32_t key0 = static_cast<uint32_t>(seed);
    uint32_t key1 = static_cast<uint32_t>(seed >> 32);
    for (int round = 0; round < 10; ++round) {
        counter = PhiloxRound(counter, key0, key1);
        key0 += kPhiloxW0;
        key1 += kPhiloxW1;
    }
    return counter;
}

uint32_t PhiloxRandomUint(uint64_t seed, uint64_t offset) {
    const auto values = Philox(seed, offset / 4);
    switch (offset % 4) {
    case 0:
        return values.c0;
    case 1:
        return values.c1;
    case 2:
        return values.c2;
    default:
        return values.c3;
    }
}

float UintToUniform(uint32_t value) { return static_cast<float>(value >> 8) * 0x1.0p-24f; }

void CheckUniformBounds(float from, float to) {
    CHECK_LE(from, to);
    CHECK(std::isfinite(from)) << "Uniform lower bound must be finite";
    CHECK(std::isfinite(to)) << "Uniform upper bound must be finite";
    const double range = static_cast<double>(to) - static_cast<double>(from);
    CHECK_LE(range, static_cast<double>(std::numeric_limits<float>::max()))
        << "Uniform bounds range exceeds float maximum";
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
    std::pair<uint64_t, uint64_t> ReserveRandomOffset(uint64_t increment) override;

    void FillUniform(std::vector<float> &buffer, float from, float to) override;
    void FillNormal(std::vector<float> &buffer, float mean, float std) override;

private:
    mutable std::mutex mutex_;
    uint64_t initial_seed_ = 0;
    std::mt19937 engine_;
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
    std::pair<uint64_t, uint64_t> ReserveRandomOffset(uint64_t increment) override;

    void FillUniform(std::vector<float> &buffer, float from, float to) override;
    void FillNormal(std::vector<float> &buffer, float mean, float std) override;

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

std::pair<uint64_t, uint64_t> Generator::ReserveRandomOffset(uint64_t increment) {
    return impl_->ReserveRandomOffset(increment);
}

void Generator::FillUniform(std::vector<float> &buffer, float from, float to) { impl_->FillUniform(buffer, from, to); }

void Generator::FillNormal(std::vector<float> &buffer, float mean, float std) { impl_->FillNormal(buffer, mean, std); }

CPUGeneratorImpl::CPUGeneratorImpl(uint64_t seed) { ManualSeed(seed); }

void CPUGeneratorImpl::ManualSeed(uint64_t seed) {
    std::lock_guard<std::mutex> lock(mutex_);
    initial_seed_ = seed;
    SeedEngine(engine_, seed);
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
    oss << kCPUStateHeader << "\n" << initial_seed_ << "\n" << engine_;
    return ToBytes(oss.str());
}

void CPUGeneratorImpl::SetState(const std::vector<uint8_t> &state) {
    const std::string serialized = ToString(state);
    std::istringstream iss(serialized);

    std::string header;
    std::mt19937 engine;
    std::getline(iss, header);
    CHECK_EQ(header, kCPUStateHeader) << "Invalid CPU generator state header";
    const uint64_t seed = ParseStateIntegerLine<uint64_t>(iss, "Invalid CPU generator seed in state");
    std::ostringstream engine_state_stream;
    engine_state_stream << iss.rdbuf();
    const std::string engine_state = engine_state_stream.str();
    ValidateCPUEngineState(engine_state);
    std::istringstream engine_iss(engine_state);
    CHECK(engine_iss >> engine) << "Invalid CPU generator engine state";
    CheckNoTrailingStateData(engine_iss, "CPU");

    std::lock_guard<std::mutex> lock(mutex_);
    initial_seed_ = seed;
    engine_ = engine;
}

Device CPUGeneratorImpl::GetDevice() const { return Device(); }

std::pair<uint64_t, uint64_t> CPUGeneratorImpl::ReserveRandomOffset(uint64_t) {
    LOG(FATAL) << "CPU generator does not expose a CUDA random offset";
    return {0, 0};
}

void CPUGeneratorImpl::FillUniform(std::vector<float> &buffer, float from, float to) {
    CheckUniformBounds(from, to);
    std::lock_guard<std::mutex> lock(mutex_);
    std::uniform_real_distribution<float> dis(from, to);
    std::generate(buffer.begin(), buffer.end(), [&]() { return dis(engine_); });
}

void CPUGeneratorImpl::FillNormal(std::vector<float> &buffer, float mean, float std) {
    CHECK_GE(std, 0.0f);
    std::lock_guard<std::mutex> lock(mutex_);
    if (std == 0.0f) {
        std::normal_distribution<float> dis(0.0f, 1.0f);
        std::generate(buffer.begin(), buffer.end(), [&]() {
            (void)dis(engine_);
            return mean;
        });
        return;
    }
    std::normal_distribution<float> dis(mean, std);
    std::generate(buffer.begin(), buffer.end(), [&]() { return dis(engine_); });
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
    oss << kCUDAStateHeader << "\n"
        << initial_seed_ << "\n"
        << static_cast<int>(device_.index()) << "\n"
        << offset_ << "\n";
    return ToBytes(oss.str());
}

void CUDAGeneratorImpl::SetState(const std::vector<uint8_t> &state) {
    const std::string serialized = ToString(state);
    std::istringstream iss(serialized);

    std::string header;
    std::getline(iss, header);
    CHECK_EQ(header, kCUDAStateHeader) << "Invalid CUDA generator state header";
    const uint64_t seed = ParseStateIntegerLine<uint64_t>(iss, "Invalid CUDA generator seed in state");
    const int device_index = ParseStateIntegerLine<int>(iss, "Invalid CUDA generator device index in state");
    CHECK_GE(device_index, 0) << "Invalid CUDA generator device index in state";
    CHECK_LE(device_index, std::numeric_limits<int8_t>::max()) << "Invalid CUDA generator device index in state";
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
    offset_ += increment;
    return {initial_seed_, offset};
}

void CUDAGeneratorImpl::FillUniform(std::vector<float> &buffer, float from, float to) {
    CheckUniformBounds(from, to);
    std::lock_guard<std::mutex> lock(mutex_);
    CHECK_LE(buffer.size(), std::numeric_limits<uint64_t>::max() - offset_) << "CUDA generator offset overflow";
    for (size_t i = 0; i < buffer.size(); ++i) {
        const float uniform = UintToUniform(PhiloxRandomUint(initial_seed_, offset_ + i));
        buffer[i] = from + (to - from) * uniform;
    }
    offset_ += buffer.size();
}

void CUDAGeneratorImpl::FillNormal(std::vector<float> &buffer, float mean, float stddev) {
    CHECK_GE(stddev, 0.0f);
    std::lock_guard<std::mutex> lock(mutex_);
    CHECK_LE(buffer.size(), std::numeric_limits<uint64_t>::max() / 2) << "CUDA generator offset overflow";
    CHECK_LE(buffer.size() * 2, std::numeric_limits<uint64_t>::max() - offset_) << "CUDA generator offset overflow";
    for (size_t i = 0; i < buffer.size(); ++i) {
        const uint64_t element_offset = offset_ + i * 2;
        float uniform1 = UintToUniform(PhiloxRandomUint(initial_seed_, element_offset));
        const float uniform2 = UintToUniform(PhiloxRandomUint(initial_seed_, element_offset + 1));
        uniform1 = std::max(uniform1, 0x1.0p-24f);
        const float normal = std::sqrt(-2.0f * std::log(uniform1)) * std::cos(kTwoPi * uniform2);
        buffer[i] = mean + stddev * normal;
    }
    offset_ += buffer.size() * 2;
}

std::shared_ptr<Generator> MakeCPUGenerator(uint64_t seed) {
    return std::make_shared<Generator>(std::make_shared<CPUGeneratorImpl>(seed));
}

std::shared_ptr<Generator> MakeCUDAGenerator(int8_t device_index, uint64_t seed) {
    return std::make_shared<Generator>(std::make_shared<CUDAGeneratorImpl>(device_index, seed));
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
    GetDefaultCPUGenerator()->ManualSeed(seed);

    std::lock_guard<std::mutex> lock(DefaultGeneratorMutex());
    DefaultGeneratorSeed() = seed;
    for (auto &[_, generator] : DefaultCUDAGenerators()) { generator->ManualSeed(seed); }
}

} // namespace infini_train
