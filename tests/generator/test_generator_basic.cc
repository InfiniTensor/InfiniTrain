#include <memory>

#include "gtest/gtest.h"

#include "infini_train/include/device.h"
#include "infini_train/include/generator.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class GeneratorBasicTest : public infini_train::test::InfiniTrainTest {};

TEST_P(GeneratorBasicTest, ConstructionAttachesDevice) {
    Generator gen(GetDevice());
    EXPECT_EQ(gen.device(), GetDevice());
    EXPECT_NE(gen.impl(), nullptr);
}

TEST_P(GeneratorBasicTest, ManualSeedRoundtripsCurrentAndInitial) {
    Generator gen(GetDevice());
    gen.ManualSeed(2026);
    EXPECT_EQ(gen.InitialSeed(), 2026u);
}

TEST_P(GeneratorBasicTest, CopyShareImpl) {
    Generator a(GetDevice());
    a.ManualSeed(1);
    Generator b = a;
    EXPECT_EQ(a.impl().get(), b.impl().get());
    b.ManualSeed(2);
    EXPECT_EQ(a.InitialSeed(), 2u);
}

INFINI_TRAIN_REGISTER_TEST(GeneratorBasicTest);
