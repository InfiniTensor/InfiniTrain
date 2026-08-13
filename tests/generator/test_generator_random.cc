#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <thread>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/generator.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/tensor.h"
#include "tests/generator/generator_test_utils.h"

namespace infini_train::test {

TEST_P(GeneratorDeviceTest, FanCalculationUses64BitShapeProducts) {
    constexpr int64_t kSpatialExtent = 65536;
    auto tensor = std::make_shared<Tensor>(
        std::vector<int64_t>{0, 3, kSpatialExtent, kSpatialExtent}, DataType::kFLOAT32, GetDevice());

    const auto [fan_in, fan_out] = nn::init::CalculateFanInAndFanOut(tensor);
    EXPECT_EQ(fan_in, int64_t{3} * kSpatialExtent * kSpatialExtent);
    EXPECT_EQ(fan_out, 0);
}

TEST_P(GeneratorDeviceTest, UniformAndNormalRespectExplicitGenerator) {
    const auto device = GetDevice();
    auto first_uniform = std::make_shared<Tensor>(std::vector<int64_t>{32}, DataType::kFLOAT32, device);
    auto second_uniform = std::make_shared<Tensor>(std::vector<int64_t>{32}, DataType::kFLOAT32, device);
    nn::init::Uniform(first_uniform, -2.0f, 3.0f, MakeGenerator(device, 123));
    nn::init::Uniform(second_uniform, -2.0f, 3.0f, MakeGenerator(device, 123));
    EXPECT_EQ(CopyToCPUData(first_uniform), CopyToCPUData(second_uniform));

    auto first_normal = std::make_shared<Tensor>(std::vector<int64_t>{32}, DataType::kFLOAT32, device);
    auto second_normal = std::make_shared<Tensor>(std::vector<int64_t>{32}, DataType::kFLOAT32, device);
    nn::init::Normal(first_normal, 1.5f, 0.5f, MakeGenerator(device, 456));
    nn::init::Normal(second_normal, 1.5f, 0.5f, MakeGenerator(device, 456));
    EXPECT_EQ(CopyToCPUData(first_normal), CopyToCPUData(second_normal));
}

TEST_P(GeneratorDeviceTest, GlobalSeedMakesKaimingUniformReproducible) {
    const auto device = GetDevice();
    auto first = std::make_shared<Tensor>(std::vector<int64_t>{16, 32}, DataType::kFLOAT32, device);
    auto second = std::make_shared<Tensor>(std::vector<int64_t>{16, 32}, DataType::kFLOAT32, device);

    ManualSeedAll(314159);
    nn::init::KaimingUniform(first, std::sqrt(5.0f), nn::init::KaimingMode::kFanIn,
                             nn::init::NonLinearityType::kLeakyReLU);
    infini_train::ManualSeed(314159);
    nn::init::KaimingUniform(second, std::sqrt(5.0f), nn::init::KaimingMode::kFanIn,
                             nn::init::NonLinearityType::kLeakyReLU);
    EXPECT_EQ(CopyToCPUData(first), CopyToCPUData(second));
}

TEST_P(GeneratorDeviceTest, ZeroElementCallsDoNotAdvanceState) {
    const auto device = GetDevice();
    auto generator = MakeGenerator(device, 123);
    const auto before = generator->GetState();
    (void)nn::function::Rand({0}, device, generator);
    (void)nn::function::Randn({0}, device, generator);
    EXPECT_EQ(before, generator->GetState());
}

TEST_P(GeneratorDeviceTest, ValidatesDistributionParameters) {
    const auto device = GetDevice();
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{4}, DataType::kFLOAT32, device);
    auto generator = MakeGenerator(device, 123);

    EXPECT_DEATH(nn::init::Uniform(tensor, 2.0f, 1.0f, generator), "Check failed");
    EXPECT_DEATH(nn::init::Uniform(tensor, 0.0f, std::numeric_limits<float>::infinity(), generator),
                 "Uniform upper bound must be finite");
    EXPECT_DEATH(
        nn::init::Uniform(tensor, std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max(), generator),
        "Uniform bounds range exceeds float maximum");
    EXPECT_DEATH(nn::init::Normal(tensor, 0.0f, -1.0f, generator), "Check failed");

    auto wrong_backend = device.IsCPU() ? MakeCUDAGenerator(0, 123) : MakeCPUGenerator(123);
    EXPECT_DEATH(nn::init::Uniform(tensor, 0.0f, 1.0f, wrong_backend), "Generator backend must match");
}

TEST_P(GeneratorDeviceTest, EqualUniformBoundsAndZeroStdReturnConstants) {
    const auto device = GetDevice();
    auto uniform = std::make_shared<Tensor>(std::vector<int64_t>{17}, DataType::kFLOAT32, device);
    auto normal = std::make_shared<Tensor>(std::vector<int64_t>{17}, DataType::kFLOAT32, device);
    auto uniform_generator = MakeGenerator(device, 123);
    auto normal_generator = MakeGenerator(device, 123);
    const auto uniform_state = uniform_generator->GetState();
    const auto normal_state = normal_generator->GetState();

    nn::init::Uniform(uniform, 2.5f, 2.5f, uniform_generator);
    nn::init::Normal(normal, -3.0f, 0.0f, normal_generator);
    EXPECT_EQ(CopyToCPUData(uniform), std::vector<float>(17, 2.5f));
    EXPECT_EQ(CopyToCPUData(normal), std::vector<float>(17, -3.0f));
    EXPECT_NE(uniform_state, uniform_generator->GetState());
    EXPECT_NE(normal_state, normal_generator->GetState());
}

TEST_P(GeneratorDeviceTest, UniformNeverReturnsExclusiveUpperBoundAfterRounding) {
    const auto device = GetDevice();
    // These seeds hit the float-rounding edge in the current CPU and CUDA engines.
    const uint64_t seed = device.IsCPU() ? 179040 : 16390493;
    const int64_t size = device.IsCPU() ? 28 : 1;
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{size}, DataType::kFLOAT32, device);

    nn::init::Uniform(tensor, 1.0f, 2.0f, MakeGenerator(device, seed));
    const auto values = CopyToCPUData(tensor);
    EXPECT_TRUE(std::all_of(values.begin(), values.end(), [](float value) { return value >= 1.0f && value < 2.0f; }));
}

TEST_P(GeneratorDeviceTest, ConcurrentCallsConsumeDisjointStreamSegments) {
    constexpr int kNumThreads = 4;
    constexpr int64_t kChunkSize = 257;
    const auto device = GetDevice();

    for (const bool normal : {false, true}) {
        SCOPED_TRACE(normal ? "Randn" : "Rand");
        auto generator = MakeGenerator(device, 123);
        const auto random_values = [&](int64_t size, const auto &current_generator) {
            auto tensor = normal ? nn::function::Randn({size}, device, current_generator)
                                 : nn::function::Rand({size}, device, current_generator);
            return CopyToCPUData(tensor);
        };
        std::vector<std::vector<float>> chunks(kNumThreads);
        std::vector<std::thread> workers;

        for (int i = 0; i < kNumThreads; ++i) {
            workers.emplace_back([&, i]() { chunks[i] = random_values(kChunkSize, generator); });
        }
        for (auto &worker : workers) { worker.join(); }

        std::vector<float> concurrent_values;
        concurrent_values.reserve(kNumThreads * kChunkSize);
        for (const auto &chunk : chunks) {
            concurrent_values.insert(concurrent_values.end(), chunk.begin(), chunk.end());
        }
        auto serial_generator = MakeGenerator(device, 123);
        auto serial_values = random_values(kNumThreads * kChunkSize, serial_generator);
        std::sort(concurrent_values.begin(), concurrent_values.end());
        std::sort(serial_values.begin(), serial_values.end());
        EXPECT_EQ(concurrent_values, serial_values);
        EXPECT_EQ(generator->GetState(), serial_generator->GetState());
    }
}

TEST_P(GeneratorDeviceTest, RandomValuesHaveSaneRangeAndMoments) {
    const auto device = GetDevice();
    const auto uniform = CopyToCPUData(nn::function::Rand({65537}, device, MakeGenerator(device, 123)));
    EXPECT_TRUE(std::all_of(uniform.begin(), uniform.end(), [](float value) { return value >= 0.0f && value < 1.0f; }));

    const auto normal = CopyToCPUData(nn::function::Randn({65537}, device, MakeGenerator(device, 456)));
    const double mean = std::accumulate(normal.begin(), normal.end(), 0.0) / normal.size();
    const double squared_sum = std::inner_product(normal.begin(), normal.end(), normal.begin(), 0.0);
    const double variance = squared_sum / normal.size() - mean * mean;
    EXPECT_NEAR(mean, 0.0, 0.03);
    EXPECT_NEAR(variance, 1.0, 0.06);
}

} // namespace infini_train::test
