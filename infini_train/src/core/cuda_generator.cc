#ifdef USE_CUDA

#include "infini_train/include/core/cuda_generator.h"

#include <cstdio>
#include <cstdint>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/device.h"

namespace infini_train::core {
namespace {
constexpr uint64_t kCUDAStateMagic = 0x494E464355444731ULL; // "INFCUDG1"
// Explicit format version for forward-compatible state evolution.
constexpr uint64_t kCUDAStateVersion = 1;

using detail::AppendU64;
using detail::ReadU64;
} // namespace

CUDAGeneratorImpl::CUDAGeneratorImpl(int8_t device_index, uint64_t seed)
    : device_index_(device_index), seed_(seed), philox_offset_(0) {}

void CUDAGeneratorImpl::SetCurrentSeed(uint64_t seed) {
    seed_ = seed;
    philox_offset_ = 0;
}

uint64_t CUDAGeneratorImpl::CurrentSeed() const { return seed_; }

uint64_t CUDAGeneratorImpl::Seed() {
    std::random_device rd;
    // Limit to 53 bits to ensure unique representation when cast to double.
    // This matches PyTorch's getNonDeterministicRandom(is_cuda=true) behavior.
    uint64_t seed = ((static_cast<uint64_t>(rd()) << 32) | static_cast<uint64_t>(rd())) & 0x1FFFFFFFFFFFFF;
    SetCurrentSeed(seed);
    return seed;
}

void CUDAGeneratorImpl::SetOffset(uint64_t offset) { philox_offset_ = offset; }

uint64_t CUDAGeneratorImpl::GetOffset() const { return philox_offset_; }

std::vector<uint8_t> CUDAGeneratorImpl::GetState() const {
    // Layout: [magic u64][version u64][seed u64][philox_offset u64][device_index u64].
    // device_index_ (int8_t) is widened to uint64_t for uniform 8-byte field
    // alignment. The truncation in SetState is safe because device indices
    // are always non-negative and well within [0, 127].
    std::vector<uint8_t> state;
    state.reserve(40);
    AppendU64(state, kCUDAStateMagic);
    AppendU64(state, kCUDAStateVersion);
    AppendU64(state, seed_);
    AppendU64(state, philox_offset_);
    AppendU64(state, static_cast<uint64_t>(device_index_));
    return state;
}

void CUDAGeneratorImpl::SetState(const std::vector<uint8_t> &state) {
    CHECK_EQ(state.size(), static_cast<size_t>(40)) << "Invalid CUDA generator state: unexpected length";
    const uint64_t magic = ReadU64(state.data());
    CHECK_EQ(magic, kCUDAStateMagic) << "Invalid CUDA generator state: backend magic mismatch "
                                        "(state may come from a non-CUDA generator)";
    const uint64_t version = ReadU64(state.data() + 8);
    CHECK_EQ(version, kCUDAStateVersion) << "Invalid CUDA generator state: unsupported version "
                                         << version << " (expected " << kCUDAStateVersion << ")";
    // Truncation back to int8_t is safe: device indices are always in [0, 127].
    // The device_index check enforces that a state produced by a different CUDA
    // device cannot be loaded into this generator, preventing unintended
    // cross-device state aliasing (stricter than PyTorch, which omits this check).
    const int8_t state_device_index = static_cast<int8_t>(ReadU64(state.data() + 32));
    CHECK_EQ(state_device_index, device_index_)
        << "Invalid CUDA generator state: device index mismatch, expected " << static_cast<int>(device_index_)
        << " but found " << static_cast<int>(state_device_index);
    seed_ = ReadU64(state.data() + 16);
    philox_offset_ = ReadU64(state.data() + 24);
}

Device CUDAGeneratorImpl::GetDevice() const { return Device(Device::DeviceType::kCUDA, device_index_); }

std::shared_ptr<GeneratorImpl> CUDAGeneratorImpl::Clone() const {
    auto cloned = std::make_shared<CUDAGeneratorImpl>(device_index_, seed_);
    cloned->philox_offset_ = philox_offset_;
    return cloned;
}

std::pair<uint64_t, uint64_t> CUDAGeneratorImpl::PhiloxEngineInputs(uint64_t increment) {
    const uint64_t offset = philox_offset_;
    philox_offset_ += increment;
    return {seed_, offset};
}

} // namespace infini_train::core

#endif // USE_CUDA
