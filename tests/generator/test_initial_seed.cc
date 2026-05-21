#include "gtest/gtest.h"

#include "infini_train/include/device.h"
#include "infini_train/include/generator.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class GeneratorInitialSeedTest : public infini_train::test::InfiniTrainTest {};

TEST_P(GeneratorInitialSeedTest, SeedDoesNotChangeInitialSeed) {
    Generator gen(GetDevice());
    gen.ManualSeed(1);
    EXPECT_EQ(gen.InitialSeed(), 1u);
    const uint64_t new_seed = gen.Seed();
    EXPECT_NE(new_seed, 1u);
    EXPECT_EQ(gen.InitialSeed(), 1u);
}

INFINI_TRAIN_REGISTER_TEST(GeneratorInitialSeedTest);
