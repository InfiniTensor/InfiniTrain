#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/device.h"
#include "infini_train/include/generator.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class GeneratorStateTest : public infini_train::test::InfiniTrainTest {};

TEST_P(GeneratorStateTest, GetSetStateRoundtrip) {
    Generator gen(GetDevice());
    gen.ManualSeed(2026);

    const auto baseline = gen.GetState();
    gen.ManualSeed(99);
    EXPECT_NE(gen.GetState(), baseline);
    gen.SetState(baseline);
    EXPECT_EQ(gen.GetState(), baseline);
    EXPECT_EQ(gen.InitialSeed(), 2026u);
}

INFINI_TRAIN_REGISTER_TEST(GeneratorStateTest);
