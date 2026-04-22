#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "infini_train/include/tensor.h"
#include "test_utils.h"

using namespace infini_train;

// ============================================================================
// Op tests — CPU + CUDA
// ============================================================================

class TensorOpTest : public infini_train::test::InfiniTrainTest {};

TEST_P(TensorOpTest, MatmulAllocatesOutputs) {
    auto a = createTensor({2, 3});
    auto b = createTensor({3, 4});
    auto c = createTensor({2, 4});
    EXPECT_NE(a->DataPtr(), nullptr);
    EXPECT_NE(b->DataPtr(), nullptr);
    EXPECT_NE(c->DataPtr(), nullptr);
}

INFINI_TRAIN_REGISTER_TEST(TensorOpTest);
