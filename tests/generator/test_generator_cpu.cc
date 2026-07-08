// CPU Generator unit tests.

#include "gtest/gtest.h"
#include "infini_train/include/core/cpu_generator.h"
#include "infini_train/include/core/generator.h"
#include "infini_train/include/device.h"
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

using namespace infini_train;

namespace {

core::Generator MakeCPUGenerator(uint64_t seed) {
    return core::Generator(std::make_shared<core::CPUGeneratorImpl>(seed));
}

std::vector<uint64_t> DrawCPU(core::Generator &generator, int64_t count) {
    auto *impl = generator.Get<core::CPUGeneratorImpl>();
    std::lock_guard<std::mutex> lock(generator.Mutex());
    std::vector<uint64_t> values(count);
    auto &engine = impl->Engine();
    for (auto &value : values) {
        value = engine();
    }
    return values;
}

} // namespace

TEST(CPUGeneratorTest, SeedSetAndGet) {
    auto gen = MakeCPUGenerator(123);
    EXPECT_EQ(gen.CurrentSeed(), 123u);

    gen.SetCurrentSeed(456);
    EXPECT_EQ(gen.CurrentSeed(), 456u);

    gen.ManualSeed(789);
    EXPECT_EQ(gen.CurrentSeed(), 789u);

    const uint64_t drawn = gen.Seed();
    EXPECT_EQ(gen.CurrentSeed(), drawn);
}

TEST(CPUGeneratorTest, DeviceType) {
    auto gen = MakeCPUGenerator(1);
    EXPECT_TRUE(gen.GetDevice().IsCPU());
    EXPECT_EQ(gen.GetDevice().type(), Device::DeviceType::kCPU);
}

TEST(CPUGeneratorTest, CloneIsIndependentWithSameState) {
    auto gen = MakeCPUGenerator(42);
    auto prefix = DrawCPU(gen, 4);
    EXPECT_EQ(prefix.size(), 4u);

    auto cloned = gen.Clone();
    EXPECT_NE(gen.UnsafeGetImpl(), cloned.UnsafeGetImpl());
    EXPECT_EQ(DrawCPU(gen, 8), DrawCPU(cloned, 8));
}

TEST(CPUGeneratorTest, SameSeedSameSequence) {
    auto g1 = MakeCPUGenerator(2024);
    auto g2 = MakeCPUGenerator(2024);
    EXPECT_EQ(DrawCPU(g1, 64), DrawCPU(g2, 64));
}

TEST(CPUGeneratorTest, DifferentSeedDifferentSequence) {
    auto g1 = MakeCPUGenerator(1);
    auto g2 = MakeCPUGenerator(2);
    EXPECT_NE(DrawCPU(g1, 64), DrawCPU(g2, 64));
}

TEST(CPUGeneratorTest, GetSetStateRestoresCPUSequence) {
    auto gen = MakeCPUGenerator(99);
    const auto state = gen.GetState();

    const auto first = DrawCPU(gen, 32);

    gen.SetState(state);
    const auto restored = DrawCPU(gen, 32);

    EXPECT_EQ(first, restored);
}

TEST(CPUGeneratorTest, StateRoundTripPreservesSeed) {
    auto gen = MakeCPUGenerator(0xABCDEF);
    const auto state = gen.GetState();

    auto other = MakeCPUGenerator(1);
    other.SetState(state);
    EXPECT_EQ(other.CurrentSeed(), 0xABCDEFu);
}

TEST(CPUGeneratorTest, RejectsMalformedCPUState) {
    auto gen = MakeCPUGenerator(1);
    EXPECT_DEATH(gen.SetState(std::vector<uint8_t>{1, 2, 3}), "Invalid CPU generator state");
}

TEST(CPUGeneratorTest, RejectsForeignBackendState) {
    auto gen = MakeCPUGenerator(1);
    std::vector<uint8_t> foreign(32, 0);
    EXPECT_DEATH(gen.SetState(foreign), "backend magic mismatch");
}

TEST(CPUGeneratorTest, RejectsUnsupportedCPUStateVersion) {
    auto gen = MakeCPUGenerator(42);
    auto state = gen.GetState();

    // Corrupt version field (bytes 8-15) to 99.
    state[8] = 99;
    for (int i = 9; i < 16; ++i) {
        state[i] = 0;
    }

    EXPECT_DEATH(gen.SetState(state), "unsupported version");
}

TEST(CPUGeneratorTest, CPUOnlySupportsZeroOffset) {
    auto gen = MakeCPUGenerator(7);
    EXPECT_EQ(gen.GetOffset(), 0u);
    gen.SetOffset(0);
    EXPECT_EQ(gen.GetOffset(), 0u);
    EXPECT_DEATH(gen.SetOffset(1), "non-zero offset");
}
