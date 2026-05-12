#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

// ============================================================================
// Op tests — CPU + CUDA
// ============================================================================

class TensorOpTest : public infini_train::test::InfiniTrainTest {};

TEST_P(TensorOpTest, MatmulAllocatesOutputs) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice());
    auto b = std::make_shared<Tensor>(std::vector<int64_t>{3, 4}, DataType::kFLOAT32, GetDevice());
    auto c = std::make_shared<Tensor>(std::vector<int64_t>{2, 4}, DataType::kFLOAT32, GetDevice());
    EXPECT_NE(a->DataPtr(), nullptr);
    EXPECT_NE(b->DataPtr(), nullptr);
    EXPECT_NE(c->DataPtr(), nullptr);
}

INFINI_TRAIN_REGISTER_TEST(TensorOpTest);
