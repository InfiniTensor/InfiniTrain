#include "infini_train/include/core/cpu_generator.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/device.h"

namespace infini_train::core {
namespace {
// Backend magic prefixing every serialized CPU state. Used by SetState() to
// reject states produced by a different backend.
constexpr uint64_t kCPUStateMagic = 0x494E464350554731ULL; // "INFCPUG1"
// Explicit format version for forward-compatible state evolution.
constexpr uint64_t kCPUStateVersion = 1;

using detail::AppendU64;
using detail::ReadU64;
} // namespace

CPUGeneratorImpl::CPUGeneratorImpl(uint64_t seed) : seed_(seed), engine_(seed) {}

void CPUGeneratorImpl::SetCurrentSeed(uint64_t seed) {
    seed_ = seed;
    engine_.seed(seed);
}

uint64_t CPUGeneratorImpl::CurrentSeed() const { return seed_; }

uint64_t CPUGeneratorImpl::Seed() {
    std::random_device rd;
    uint64_t seed = (static_cast<uint64_t>(rd()) << 32) | static_cast<uint64_t>(rd());
    SetCurrentSeed(seed);
    return seed;
}

void CPUGeneratorImpl::SetOffset(uint64_t offset) {
    CHECK_EQ(offset, 0u) << "CPUGeneratorImpl does not support non-zero offset";
}

uint64_t CPUGeneratorImpl::GetOffset() const { return 0; }

std::vector<uint8_t> CPUGeneratorImpl::GetState() const {
    // Layout: [magic u64][version u64][seed u64][engine-text-length u64][engine text bytes].
    std::ostringstream oss;
    oss << engine_;
    const std::string engine_text = oss.str();

    std::vector<uint8_t> state;
    state.reserve(32 + engine_text.size());
    AppendU64(state, kCPUStateMagic);
    AppendU64(state, kCPUStateVersion);
    AppendU64(state, seed_);
    AppendU64(state, static_cast<uint64_t>(engine_text.size()));
    state.insert(state.end(), engine_text.begin(), engine_text.end());
    return state;
}

void CPUGeneratorImpl::SetState(const std::vector<uint8_t> &state) {
    CHECK_GE(state.size(), static_cast<size_t>(32)) << "Invalid CPU generator state: too short";
    const uint64_t magic = ReadU64(state.data());
    CHECK_EQ(magic, kCPUStateMagic) << "Invalid CPU generator state: backend magic mismatch "
                                       "(state may come from a non-CPU generator)";
    const uint64_t version = ReadU64(state.data() + 8);
    CHECK_EQ(version, kCPUStateVersion) << "Invalid CPU generator state: unsupported version " << version
                                        << " (expected " << kCPUStateVersion << ")";
    const uint64_t seed = ReadU64(state.data() + 16);
    const uint64_t text_len = ReadU64(state.data() + 24);
    CHECK_EQ(state.size(), static_cast<size_t>(32) + text_len) << "Invalid CPU generator state: length mismatch";

    const std::string engine_text(reinterpret_cast<const char *>(state.data() + 32), text_len);
    std::istringstream iss(engine_text);
    iss >> engine_;
    CHECK(!iss.fail()) << "Invalid CPU generator state: engine deserialization failed";
    seed_ = seed;
}

Device CPUGeneratorImpl::GetDevice() const { return Device(Device::DeviceType::kCPU, 0); }

std::shared_ptr<GeneratorImpl> CPUGeneratorImpl::Clone() const {
    auto cloned = std::make_shared<CPUGeneratorImpl>(seed_);
    cloned->engine_ = engine_;
    return cloned;
}

std::mt19937_64 &CPUGeneratorImpl::Engine() { return engine_; }

} // namespace infini_train::core
