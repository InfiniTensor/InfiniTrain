#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/core/generator/cpu_generator_impl.h"

using infini_train::core::CPUGeneratorImpl;

namespace {

constexpr uint32_t kCpuMagic = 0x47555043; // 'CPUG'
constexpr uint32_t kCpuVersion = 1;

std::vector<uint8_t> MakeHeader(uint32_t magic, uint32_t version, uint64_t payload_size) {
    std::vector<uint8_t> buf(16);
    std::memcpy(buf.data() + 0, &magic, 4);
    std::memcpy(buf.data() + 4, &version, 4);
    std::memcpy(buf.data() + 8, &payload_size, 8);
    return buf;
}

} // namespace

TEST(CpuStateValidation, TruncatedHeader) {
    CPUGeneratorImpl impl(0);
    impl.SetCurrentSeed(1);
    std::vector<uint8_t> too_short(15, 0);
    EXPECT_THROW(impl.SetState(too_short), std::runtime_error);
}

TEST(CpuStateValidation, BadMagic) {
    CPUGeneratorImpl impl(0);
    impl.SetCurrentSeed(1);
    auto buf = MakeHeader(0xDEADBEEF, kCpuVersion, 0);
    EXPECT_THROW(impl.SetState(buf), std::runtime_error);
}

TEST(CpuStateValidation, BadVersion) {
    CPUGeneratorImpl impl(0);
    impl.SetCurrentSeed(1);
    auto buf = MakeHeader(kCpuMagic, kCpuVersion + 1, 0);
    EXPECT_THROW(impl.SetState(buf), std::runtime_error);
}

TEST(CpuStateValidation, PayloadSizeMismatch) {
    CPUGeneratorImpl impl(0);
    impl.SetCurrentSeed(1);
    auto buf = MakeHeader(kCpuMagic, kCpuVersion, 100);
    buf.insert(buf.end(), {0xAA, 0xBB, 0xCC, 0xDD});
    EXPECT_THROW(impl.SetState(buf), std::runtime_error);
}
