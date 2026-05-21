#include <thread>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/device.h"
#include "infini_train/include/generator.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class GeneratorDefaultTest : public infini_train::test::InfiniTrainTest {};

TEST_P(GeneratorDefaultTest, DefaultGeneratorIsStable) {
    auto &a = default_generator(GetDevice());
    auto &b = default_generator(GetDevice());
    EXPECT_EQ(&a, &b);
    EXPECT_EQ(a.impl().get(), b.impl().get());
}

TEST_P(GeneratorDefaultTest, ManualSeedTouchesDefault) {
    manual_seed(31415);
    auto &gen = default_generator(GetDevice());
    EXPECT_EQ(gen.InitialSeed(), 31415u);
}

TEST_P(GeneratorDefaultTest, ManualSeedBeforeFirstAccessRemembersSeed) {
    manual_seed(271828);
    auto &gen = default_generator(GetDevice());
    EXPECT_EQ(gen.InitialSeed(), 271828u);
}

INFINI_TRAIN_REGISTER_TEST(GeneratorDefaultTest);
