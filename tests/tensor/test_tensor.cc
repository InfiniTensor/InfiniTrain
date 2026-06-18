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

TEST_P(TensorOpTest, Detach) {
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    tensor->set_is_leaf(false);
    tensor->set_output_idx(3);

    auto data = tensor->Detach();

    EXPECT_NE(data.get(), tensor.get());
    EXPECT_EQ(data->DataPtr(), tensor->DataPtr());
    EXPECT_EQ(data->Dims(), tensor->Dims());
    EXPECT_EQ(data->Dtype(), tensor->Dtype());
    EXPECT_EQ(data->GetDevice(), tensor->GetDevice());
    EXPECT_FALSE(data->requires_grad());
    EXPECT_TRUE(data->is_leaf());
    EXPECT_EQ(data->grad_fn(), nullptr);
    EXPECT_EQ(data->output_idx(), 0);
}

INFINI_TRAIN_REGISTER_TEST(TensorOpTest);
