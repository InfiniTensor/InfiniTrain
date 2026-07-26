#include <cstdint>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/generator.h"
#include "infini_train/include/nn/functional.h"
#include "tests/generator/generator_test_utils.h"

namespace infini_train::test {
namespace {

uint64_t CUDAOffset(const std::vector<uint8_t> &state) {
    std::istringstream stream(std::string(state.begin(), state.end()));
    std::string line;
    for (int i = 0; i < 3; ++i) { CHECK(std::getline(stream, line)); }
    return std::stoull(line);
}

} // namespace

TEST_P(GeneratorDeviceTest, CUDAUsesStablePhiloxStreamAndTracksConsumption) {
    ONLY_CUDA();
    // Philox4x32-10 known-answer vector for an all-zero counter and key.
    const auto unit_float = [](uint32_t word) { return static_cast<float>(word >> 8) * 0x1.0p-24f; };
    const std::vector<float> expected_first_counter = {
        unit_float(0x6627e8d5U),
        unit_float(0xe169c58dU),
        unit_float(0xbc57ac4cU),
        unit_float(0x9b00dbd8U),
    };
    EXPECT_EQ(CopyToCPUData(nn::function::Rand({4}, GetDevice(), MakeCUDAGenerator(0, 0))), expected_first_counter);

    auto generator = MakeCUDAGenerator(0, 123);
    EXPECT_EQ(CUDAOffset(generator->GetState()), 0U);
    (void)nn::function::Rand({13}, GetDevice(), generator);
    EXPECT_EQ(CUDAOffset(generator->GetState()), 13U);
    (void)nn::function::Randn({7}, GetDevice(), generator);
    EXPECT_EQ(CUDAOffset(generator->GetState()), 27U);
}

TEST_P(GeneratorDeviceTest, CUDAExplicitGeneratorMayTargetAnotherDeviceIndex) {
    ONLY_CUDA();
    REQUIRE_MIN_DEVICES(2);
    const Device cuda1(Device::DeviceType::kCUDA, 1);
    auto expected_generator = MakeCUDAGenerator(1, 123);
    auto cross_index_generator = MakeCUDAGenerator(0, 123);

    EXPECT_EQ(CopyToCPUData(nn::function::Rand({32}, cuda1, expected_generator)),
              CopyToCPUData(nn::function::Rand({32}, cuda1, cross_index_generator)));

    const auto state = cross_index_generator->GetState();
    auto restored_on_cuda1 = MakeCUDAGenerator(1, 999);
    restored_on_cuda1->SetState(state);
    EXPECT_EQ(CopyToCPUData(nn::function::Rand({32}, cuda1, cross_index_generator)),
              CopyToCPUData(nn::function::Rand({32}, cuda1, restored_on_cuda1)));
}

TEST_P(GeneratorDeviceTest, CUDADefaultGeneratorsUseGlobalSeedAndRemainIndependent) {
    ONLY_CUDA();
    REQUIRE_MIN_DEVICES(2);
    const Device cuda0(Device::DeviceType::kCUDA, 0);
    const Device cuda1(Device::DeviceType::kCUDA, 1);

    ManualSeedAll(4242);
    const auto cuda1_expected = CopyToCPUData(nn::function::Rand({32}, cuda1));
    EXPECT_EQ(cuda1_expected, CopyToCPUData(nn::function::Rand({32}, cuda1, MakeCUDAGenerator(cuda1.index(), 4242))));
    ManualSeedAll(4242);
    (void)nn::function::Rand({32}, cuda0);
    EXPECT_EQ(cuda1_expected, CopyToCPUData(nn::function::Rand({32}, cuda1)));
}

TEST_P(GeneratorDeviceTest, ConcurrentManualSeedAllKeepsDefaultBackendsConsistent) {
    ONLY_CUDA();
    constexpr int kNumThreads = 8;
    constexpr int kSeedsPerThread = 128;
    auto cpu_generator = GetDefaultCPUGenerator();
    auto cuda_generator = GetDefaultCUDAGenerator(GetDevice().index());
    std::vector<std::thread> workers;

    for (int thread_rank = 0; thread_rank < kNumThreads; ++thread_rank) {
        workers.emplace_back([thread_rank]() {
            for (int i = 0; i < kSeedsPerThread; ++i) {
                const uint64_t seed = (static_cast<uint64_t>(thread_rank + 1) << 32) | static_cast<uint64_t>(i);
                ManualSeedAll(seed);
            }
        });
    }
    for (auto &worker : workers) { worker.join(); }

    EXPECT_EQ(cpu_generator->InitialSeed(), cuda_generator->InitialSeed());
    ManualSeedAll(2026);
}

} // namespace infini_train::test
