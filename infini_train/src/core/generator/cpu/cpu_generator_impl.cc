#include "infini_train/include/core/generator/cpu_generator_impl.h"

#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/core/generator/generator_impl.h"
#include "infini_train/include/device.h"

namespace infini_train::core {

namespace {

inline void WriteBytes(std::vector<uint8_t> &dst, const void *src, size_t n) {
    const auto *p = static_cast<const uint8_t *>(src);
    dst.insert(dst.end(), p, p + n);
}

inline size_t ReadBytes(const std::vector<uint8_t> &src, size_t offset, void *dst, size_t n) {
    if (offset + n > src.size()) {
        throw std::runtime_error("CPUGeneratorImpl::SetState: buffer underflow");
    }
    std::memcpy(dst, src.data() + offset, n);
    return offset + n;
}

} // namespace

CPUGeneratorImpl::CPUGeneratorImpl(int8_t device_index) : GeneratorImpl(Device(Device::DeviceType::kCPU, 0)) {
    CHECK_EQ(device_index, 0) << "CPU Generator only supports device index 0";
}

void CPUGeneratorImpl::SetCurrentSeed(uint64_t seed) {
    initial_seed_ = seed;
    seed_ = seed;
    engine_.seed(seed);
}

void CPUGeneratorImpl::Reseed(uint64_t seed) {
    seed_ = seed;
    engine_.seed(seed);
}

std::vector<uint8_t> CPUGeneratorImpl::GetState() const {
    std::ostringstream oss;
    oss << engine_;
    const std::string engine_blob = oss.str();
    const uint64_t engine_size = engine_blob.size();

    std::vector<uint8_t> payload;
    payload.reserve(8 + engine_size + 8 + 8);
    WriteBytes(payload, &engine_size, 8);
    WriteBytes(payload, engine_blob.data(), engine_size);
    WriteBytes(payload, &seed_, 8);
    WriteBytes(payload, &initial_seed_, 8);

    const uint64_t payload_size = payload.size();
    std::vector<uint8_t> out;
    out.reserve(16 + payload_size);
    const uint32_t magic = kMagic;
    const uint32_t version = kVersion;
    WriteBytes(out, &magic, 4);
    WriteBytes(out, &version, 4);
    WriteBytes(out, &payload_size, 8);
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
}

void CPUGeneratorImpl::SetState(const std::vector<uint8_t> &state) {
    if (state.size() < 16) {
        throw std::runtime_error("CPUGeneratorImpl::SetState: header truncated");
    }
    uint32_t magic = 0;
    uint32_t version = 0;
    uint64_t payload_size = 0;
    size_t off = 0;
    off = ReadBytes(state, off, &magic, 4);
    off = ReadBytes(state, off, &version, 4);
    off = ReadBytes(state, off, &payload_size, 8);
    if (magic != kMagic) {
        throw std::runtime_error("CPUGeneratorImpl::SetState: magic mismatch");
    }
    if (version != kVersion) {
        throw std::runtime_error("CPUGeneratorImpl::SetState: version mismatch");
    }
    if (off + payload_size != state.size()) {
        throw std::runtime_error("CPUGeneratorImpl::SetState: payload size mismatch");
    }

    uint64_t engine_size = 0;
    off = ReadBytes(state, off, &engine_size, 8);
    if (off + engine_size > state.size()) {
        throw std::runtime_error("CPUGeneratorImpl::SetState: engine blob truncated");
    }
    std::string engine_blob(reinterpret_cast<const char *>(state.data() + off), engine_size);
    off += engine_size;
    std::istringstream iss(engine_blob);
    iss >> engine_;
    if (!iss) {
        throw std::runtime_error("CPUGeneratorImpl::SetState: engine deserialization failed");
    }

    uint64_t new_seed = 0;
    uint64_t new_initial = 0;
    off = ReadBytes(state, off, &new_seed, 8);
    off = ReadBytes(state, off, &new_initial, 8);
    seed_ = new_seed;
    initial_seed_ = new_initial;
}

namespace {

std::shared_ptr<GeneratorImpl> CpuFactory(int8_t device_index) { return std::make_shared<CPUGeneratorImpl>(device_index); }

} // namespace

} // namespace infini_train::core

INFINI_TRAIN_REGISTER_GENERATOR_IMPL(kCPU, ::infini_train::core::CpuFactory)
