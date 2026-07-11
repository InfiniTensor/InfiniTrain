#include "infini_train/include/core/cpu_generator.h"
#include "infini_train/include/core/generator.h"
#include "infini_train/include/device.h"
#include "gtest/gtest.h"
#include <mutex>
#include <thread>
#include <vector>

using namespace infini_train;

namespace {

core::Generator MakeCPUGenerator(uint64_t seed) {
    return core::Generator(std::make_shared<core::CPUGeneratorImpl>(seed));
}

std::vector<uint64_t> DrawCPU(core::Generator &gen, int64_t n) {
    auto *impl = gen.Get<core::CPUGeneratorImpl>();
    std::lock_guard<std::mutex> lock(gen.Mutex());
    std::vector<uint64_t> values(static_cast<size_t>(n));
    auto &engine = impl->Engine();
    for (auto &v : values) { v = engine(); }
    return values;
}

} // namespace

TEST(GeneratorThreadSafety, MultipleThreadsDraw) {
    auto gen = MakeCPUGenerator(42);
    constexpr int kNumThreads = 4;
    constexpr int kDrawsPerThread = 1000;

    auto worker = [&gen]() {
        for (int i = 0; i < kDrawsPerThread; ++i) {
            auto *impl = gen.Get<core::CPUGeneratorImpl>();
            std::lock_guard<std::mutex> lock(gen.Mutex());
            impl->Engine()();
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < kNumThreads; ++i) { threads.emplace_back(worker); }
    for (auto &t : threads) { t.join(); }

    EXPECT_TRUE(gen.Defined());

    auto state = gen.GetState();
    auto values1 = DrawCPU(gen, 32);
    gen.SetState(state);
    auto values2 = DrawCPU(gen, 32);
    EXPECT_EQ(values1, values2);
}

TEST(GeneratorThreadSafety, StateConcurrent) {
    auto gen = MakeCPUGenerator(123);
    constexpr int kNumThreads = 4;

    auto worker = [&gen]() {
        for (int i = 0; i < 50; ++i) {
            auto state = gen.GetState();
            gen.SetState(state);

            auto *impl = gen.Get<core::CPUGeneratorImpl>();
            std::lock_guard<std::mutex> lock(gen.Mutex());
            impl->Engine()();
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < kNumThreads; ++i) { threads.emplace_back(worker); }
    for (auto &t : threads) { t.join(); }
    EXPECT_TRUE(gen.Defined());

    auto state = gen.GetState();
    auto values1 = DrawCPU(gen, 32);
    gen.SetState(state);
    auto values2 = DrawCPU(gen, 32);
    EXPECT_EQ(values1, values2);
}

TEST(GeneratorThreadSafety, ManualSeedConcurrent) {
    auto gen = MakeCPUGenerator(777);
    constexpr int kNumThreads = 4;

    auto worker = [&gen](uint64_t thread_seed) {
        for (int i = 0; i < 50; ++i) {
            gen.ManualSeed(thread_seed + static_cast<uint64_t>(i));
            auto s = gen.CurrentSeed();
            (void)s;
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < kNumThreads; ++i) { threads.emplace_back(worker, static_cast<uint64_t>(i) * 1000); }
    for (auto &t : threads) { t.join(); }
    EXPECT_TRUE(gen.Defined());

    gen.ManualSeed(42);
    auto values1 = DrawCPU(gen, 32);

    auto gen2 = MakeCPUGenerator(42);
    auto values2 = DrawCPU(gen2, 32);
    EXPECT_EQ(values1, values2);
}

TEST(GeneratorThreadSafety, DefaultGeneratorConcurrent) {
    constexpr int kNumThreads = 4;

    auto worker = [](uint64_t thread_seed) {
        for (int i = 0; i < 50; ++i) {
            core::ManualSeed(thread_seed + static_cast<uint64_t>(i));
            auto gen = core::detail::DefaultCPUGenerator();
            auto state = gen.GetState();

            auto *impl = gen.Get<core::CPUGeneratorImpl>();
            std::lock_guard<std::mutex> lock(gen.Mutex());
            impl->Engine()();
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < kNumThreads; ++i) { threads.emplace_back(worker, static_cast<uint64_t>(i) * 1000); }
    for (auto &t : threads) { t.join(); }
    EXPECT_TRUE(core::detail::DefaultCPUGenerator().Defined());

    core::ManualSeed(99);
    auto gen = core::detail::DefaultCPUGenerator();
    auto state = gen.GetState();
    auto values1 = DrawCPU(gen, 32);
    gen.SetState(state);
    auto values2 = DrawCPU(gen, 32);
    EXPECT_EQ(values1, values2);
}

TEST(GeneratorThreadSafety, ConcurrentSeedAndDrawConsistency) {
    constexpr int kNumThreads = 8;

    auto gen = MakeCPUGenerator(0);

    auto worker = [&gen](int id) {
        if (id % 2 == 0) {
            for (int i = 0; i < 100; ++i) {
                gen.ManualSeed(static_cast<uint64_t>(id) * 1000 + static_cast<uint64_t>(i));
            }
        } else {
            for (int i = 0; i < 100; ++i) {
                auto *impl = gen.Get<core::CPUGeneratorImpl>();
                std::lock_guard<std::mutex> lock(gen.Mutex());
                (void)impl->Engine()();
            }
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < kNumThreads; ++i) { threads.emplace_back(worker, i); }
    for (auto &t : threads) { t.join(); }
    EXPECT_TRUE(gen.Defined());

    gen.ManualSeed(12345);
    auto state = gen.GetState();
    auto values1 = DrawCPU(gen, 64);
    gen.SetState(state);
    auto values2 = DrawCPU(gen, 64);
    EXPECT_EQ(values1, values2);
}
