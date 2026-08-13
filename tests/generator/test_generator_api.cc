#include <memory>

#include "gtest/gtest.h"

#include "infini_train/include/generator.h"
#include "infini_train/include/nn/functional.h"
#include "tests/generator/generator_test_utils.h"

namespace infini_train::test {

TEST_P(GeneratorDeviceTest, ExposesSeedStateAndDevice) {
    const auto device = GetDevice();
    auto generator = MakeGenerator(device, 123);

    EXPECT_EQ(generator->InitialSeed(), 123U);
    EXPECT_EQ(generator->GetDevice(), device);
    EXPECT_FALSE(generator->GetState().empty());

    (void)nn::function::Randn({1}, device, generator);
    generator->ManualSeed(456);
    EXPECT_EQ(generator->InitialSeed(), 456U);
    EXPECT_EQ(CopyToCPUData(nn::function::Randn({7}, device, generator)),
              CopyToCPUData(nn::function::Randn({7}, device, MakeGenerator(device, 456))));
    const auto random_seed = generator->Seed();
    EXPECT_EQ(random_seed, generator->InitialSeed());
}

TEST_P(GeneratorDeviceTest, SameSeedReplaysRandAndRandn) {
    const auto device = GetDevice();
    auto first = MakeGenerator(device, 123);
    auto second = MakeGenerator(device, 123);
    auto different = MakeGenerator(device, 456);

    EXPECT_EQ(CopyToCPUData(nn::function::Rand({3, 5}, device, first)),
              CopyToCPUData(nn::function::Rand({3, 5}, device, second)));
    const auto first_normal = CopyToCPUData(nn::function::Randn({3, 5}, device, first));
    EXPECT_EQ(first_normal, CopyToCPUData(nn::function::Randn({3, 5}, device, second)));
    EXPECT_NE(first_normal, CopyToCPUData(nn::function::Randn({3, 5}, device, different)));
}

TEST_P(GeneratorDeviceTest, DefaultGeneratorAdvancesAndManualSeedReplays) {
    const auto device = GetDevice();
    ManualSeedAll(2026);
    const auto first = CopyToCPUData(nn::function::Rand({8}, device));
    const auto first_normal = CopyToCPUData(nn::function::Randn({8}, device));
    const auto second = CopyToCPUData(nn::function::Rand({8}, device));
    EXPECT_NE(first, second);

    ManualSeedAll(2026);
    EXPECT_EQ(first, CopyToCPUData(nn::function::Rand({8}, device)));
    EXPECT_EQ(first_normal, CopyToCPUData(nn::function::Randn({8}, device)));
    EXPECT_EQ(second, CopyToCPUData(nn::function::Rand({8}, device)));
}

TEST_P(GeneratorDeviceTest, ExplicitGeneratorDoesNotAdvanceDefaultGenerator) {
    const auto device = GetDevice();
    ManualSeedAll(777);
    const auto expected = CopyToCPUData(nn::function::Rand({8}, device));

    ManualSeedAll(777);
    (void)nn::function::Rand({8}, device, MakeGenerator(device, 123));
    EXPECT_EQ(expected, CopyToCPUData(nn::function::Rand({8}, device)));
}

TEST_P(GeneratorDeviceTest, DefaultGeneratorDispatchesByDevice) {
    const auto device = GetDevice();
    const auto generator = GetDefaultGenerator(device);
    EXPECT_EQ(generator->GetDevice(), device);
    EXPECT_EQ(generator, GetDefaultGenerator(device));
    const auto backend_generator = device.IsCPU() ? GetDefaultCPUGenerator() : GetDefaultCUDAGenerator(device.index());
    EXPECT_EQ(generator, backend_generator);
}

INFINI_TRAIN_REGISTER_TEST(GeneratorDeviceTest);

} // namespace infini_train::test
