// CUDA Generator unit tests.

#include "infini_train/include/core/generator.h"
#include "infini_train/include/device.h"
#include "gtest/gtest.h"
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#ifdef USE_CUDA
#include "infini_train/include/core/cuda_generator.h"
#endif

using namespace infini_train;

#ifdef USE_CUDA

TEST(CUDAGeneratorTest, OffsetAdvancesAndRestores) {
    core::Generator gen(std::make_shared<core::CUDAGeneratorImpl>(0, 123));
    auto *impl = gen.Get<core::CUDAGeneratorImpl>();

    auto first = impl->PhiloxEngineInputs(10);
    EXPECT_EQ(first.first, 123u);
    EXPECT_EQ(first.second, 0u);
    EXPECT_EQ(gen.GetOffset(), 10u);

    const auto state = gen.GetState();
    auto second = impl->PhiloxEngineInputs(5);
    EXPECT_EQ(second.second, 10u);

    gen.SetState(state);
    EXPECT_EQ(gen.GetOffset(), 10u);
}

TEST(CUDAGeneratorTest, SetCurrentSeedResetsOffset) {
    core::Generator gen(std::make_shared<core::CUDAGeneratorImpl>(0, 123));
    auto *impl = gen.Get<core::CUDAGeneratorImpl>();

    {
        std::lock_guard<std::mutex> lock(impl->mutex_);
        impl->PhiloxEngineInputs(32);
    }

    EXPECT_EQ(gen.GetOffset(), 32u);

    gen.SetCurrentSeed(456);
    EXPECT_EQ(gen.CurrentSeed(), 456u);
    EXPECT_EQ(gen.GetOffset(), 0u);
}

TEST(CUDAGeneratorTest, RejectsDifferentDeviceState) {
    core::Generator gen0(std::make_shared<core::CUDAGeneratorImpl>(0, 123));
    core::Generator gen1(std::make_shared<core::CUDAGeneratorImpl>(1, 456));

    const auto state_from_device_one = gen1.GetState();
    EXPECT_DEATH(gen0.SetState(state_from_device_one), "device index mismatch");
}

TEST(CUDAGeneratorTest, RejectsUnsupportedCUDAStateVersion) {
    core::Generator gen(std::make_shared<core::CUDAGeneratorImpl>(0, 42));
    auto state = gen.GetState();

    // Corrupt version field (bytes 8-15) to 99.
    state[8] = 99;
    for (int i = 9; i < 16; ++i) { state[i] = 0; }

    EXPECT_DEATH(gen.SetState(state), "unsupported version");
}

TEST(CUDAGeneratorTest, CloneIsIndependent) {
    core::Generator gen(std::make_shared<core::CUDAGeneratorImpl>(0, 42));
    auto *impl = gen.Get<core::CUDAGeneratorImpl>();
    {
        std::lock_guard<std::mutex> lock(impl->mutex_);
        impl->PhiloxEngineInputs(10);
    }

    auto cloned = gen.Clone();
    EXPECT_NE(gen.UnsafeGetImpl(), cloned.UnsafeGetImpl());
    EXPECT_EQ(gen.GetOffset(), cloned.GetOffset());
    EXPECT_EQ(gen.CurrentSeed(), cloned.CurrentSeed());

    // Advance original — clone must not follow.
    {
        std::lock_guard<std::mutex> lock(impl->mutex_);
        impl->PhiloxEngineInputs(5);
    }
    EXPECT_EQ(gen.GetOffset(), 15u);
    EXPECT_EQ(cloned.GetOffset(), 10u);
}

TEST(CUDAGeneratorTest, SetStateRoundTripPreservesSeed) {
    core::Generator gen(std::make_shared<core::CUDAGeneratorImpl>(0, 0xCAFE));
    auto *impl = gen.Get<core::CUDAGeneratorImpl>();
    {
        std::lock_guard<std::mutex> lock(impl->mutex_);
        impl->PhiloxEngineInputs(100);
    }
    const auto state = gen.GetState();

    // Mutate, then restore.
    gen.SetCurrentSeed(999);
    EXPECT_EQ(gen.CurrentSeed(), 999u);
    EXPECT_EQ(gen.GetOffset(), 0u);

    gen.SetState(state);
    EXPECT_EQ(gen.CurrentSeed(), 0xCAFEu);
    EXPECT_EQ(gen.GetOffset(), 100u);
}

TEST(CUDAGeneratorTest, ZeroIncrementDoesNotAdvanceOffset) {
    core::Generator gen(std::make_shared<core::CUDAGeneratorImpl>(0, 1));

    auto *impl = gen.Get<core::CUDAGeneratorImpl>();

    {
        std::lock_guard<std::mutex> lock(impl->mutex_);
        auto state = impl->PhiloxEngineInputs(0);

        EXPECT_EQ(state.first, 1u);
        EXPECT_EQ(state.second, 0u);
    }

    EXPECT_EQ(gen.GetOffset(), 0u);
}

TEST(CUDAGeneratorTest, MultipleSmallIncrements) {
    core::Generator gen(std::make_shared<core::CUDAGeneratorImpl>(0, 1));

    auto *impl = gen.Get<core::CUDAGeneratorImpl>();

    {
        std::lock_guard<std::mutex> lock(impl->mutex_);

        impl->PhiloxEngineInputs(3);
        impl->PhiloxEngineInputs(5);
        impl->PhiloxEngineInputs(7);
    }

    EXPECT_EQ(gen.GetOffset(), 15u);
}

TEST(CUDAGeneratorTest, LargeIncrement) {
    core::Generator gen(std::make_shared<core::CUDAGeneratorImpl>(0, 1));

    auto *impl = gen.Get<core::CUDAGeneratorImpl>();

    constexpr uint64_t kIncrement = 1ull << 20;

    {
        std::lock_guard<std::mutex> lock(impl->mutex_);
        impl->PhiloxEngineInputs(kIncrement);
    }

    EXPECT_EQ(gen.GetOffset(), kIncrement);
}

TEST(CUDAGeneratorTest, SetOffsetWorks) {
    core::Generator gen(std::make_shared<core::CUDAGeneratorImpl>(0, 1));

    gen.SetOffset(1234);

    EXPECT_EQ(gen.GetOffset(), 1234u);
}

#endif // USE_CUDA
