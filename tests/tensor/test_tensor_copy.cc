#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class TensorCopyTest : public infini_train::test::InfiniTrainTest {};

TEST_P(TensorCopyTest, CopiesBetweenSameShape) {
    auto source = std::make_shared<Tensor>(std::vector<int64_t>{4, 5, 6}, DataType::kFLOAT32, GetDevice());
    auto target = std::make_shared<Tensor>(std::vector<int64_t>{4, 5, 6}, DataType::kFLOAT32, GetDevice());
    source->Fill(0.0f);
    target->CopyFrom(source);
    EXPECT_EQ(source->Dims(), target->Dims());
}

TEST_P(TensorCopyTest, CopiesPreservesDataType) {
    auto source = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice());
    auto target = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice());
    EXPECT_EQ(source->Dtype(), target->Dtype());
    target->CopyFrom(source);
    EXPECT_EQ(target->Dtype(), DataType::kFLOAT32);
}

INFINI_TRAIN_REGISTER_TEST(TensorCopyTest);
