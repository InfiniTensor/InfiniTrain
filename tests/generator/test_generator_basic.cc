#include "gtest/gtest.h"

#include "infini_train/include/device.h"
#include "tests/common/test_utils.h"

using namespace infini_train;

class GeneratorBasicTest : public infini_train::test::InfiniTrainTest {};

TEST_P(GeneratorBasicTest, Placeholder) {
    // Will be replaced in Task 4 with real coverage.
    EXPECT_TRUE(true);
}

INFINI_TRAIN_REGISTER_TEST(GeneratorBasicTest);
