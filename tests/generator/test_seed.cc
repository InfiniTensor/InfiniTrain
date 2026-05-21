#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/device.h"
#include "infini_train/include/generator.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class GeneratorSeedTest : public infini_train::test::InfiniTrainTest {};

TEST_P(GeneratorSeedTest, ManualSeedReseedsState) {
    Generator gen(GetDevice());
    gen.ManualSeed(123);
    auto s1 = gen.GetState();
    gen.ManualSeed(123);
    auto s2 = gen.GetState();
    EXPECT_EQ(s1, s2);
}

TEST_P(GeneratorSeedTest, DifferentSeedsDifferState) {
    Generator a(GetDevice()), b(GetDevice());
    a.ManualSeed(1);
    b.ManualSeed(2);
    EXPECT_NE(a.GetState(), b.GetState());
}

INFINI_TRAIN_REGISTER_TEST(GeneratorSeedTest);
