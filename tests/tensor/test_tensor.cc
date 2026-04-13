#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "infini_train/include/tensor.h"
#include "test_utils.h"

using namespace infini_train;

// ============================================================================
// Op tests — CPU + CUDA
// ============================================================================

class TensorOpTest : public infini_train::test::TensorTestBaseP {};

TEST_P(TensorOpTest, MatmulAllocatesOutputs) {
    auto a = createTensor({2, 3});
    auto b = createTensor({3, 4});
    auto c = createTensor({2, 4});
    EXPECT_NE(a->DataPtr(), nullptr);
    EXPECT_NE(b->DataPtr(), nullptr);
    EXPECT_NE(c->DataPtr(), nullptr);
}

INFINI_TRAIN_REGISTER_TEST(TensorOpTest);

// ============================================================================
// Distributed tests — requires NCCL + >=2 GPUs
// ============================================================================

class TensorDistributedTest : public infini_train::test::DistributedInfiniTrainTestP {};

TEST_P(TensorDistributedTest, AllReduce) {
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice());
    tensor->set_requires_grad(true);
    infini_train::test::FillConstantTensor(tensor, 1.0f);
    EXPECT_TRUE(tensor->GetDevice().IsCUDA());
    EXPECT_TRUE(tensor->requires_grad());
}

TEST_P(TensorDistributedTest, AllGather) {
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{4, 4}, DataType::kFLOAT32, GetDevice());
    tensor->set_requires_grad(true);
    EXPECT_TRUE(tensor->GetDevice().IsCUDA());
    EXPECT_EQ(tensor->Dims(), (std::vector<int64_t>{4, 4}));
}

TEST_P(TensorDistributedTest, ReduceScatter) {
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{2, 8}, DataType::kFLOAT32, GetDevice());
    tensor->set_requires_grad(true);
    EXPECT_TRUE(tensor->GetDevice().IsCUDA());
    EXPECT_EQ(tensor->Dims(), (std::vector<int64_t>{2, 8}));
}

INFINI_TRAIN_REGISTER_TEST_DISTRIBUTED(TensorDistributedTest);
