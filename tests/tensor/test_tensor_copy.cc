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

TEST_P(TensorCopyTest, NoOpConversionsPreserveParameterMetadataAndViews) {
    auto storage = std::make_shared<Tensor>(std::vector<int64_t>{12}, DataType::kFLOAT32, GetDevice());
    auto grad_storage = std::make_shared<Tensor>(std::vector<int64_t>{12}, DataType::kFLOAT32, GetDevice());
    auto parameter = std::make_shared<Tensor>(*storage, 4 * sizeof(float), std::vector<int64_t>{4});
    auto grad = std::make_shared<Tensor>(*grad_storage, 4 * sizeof(float), std::vector<int64_t>{4});
    parameter->RequiresGrad();
    parameter->set_sequence_parallel(true);
    parameter->set_grad(grad);

    for (auto converted : {parameter->To(GetDevice()), parameter->To(parameter->Dtype())}) {
        EXPECT_TRUE(converted.requires_grad());
        EXPECT_TRUE(converted.sequence_parallel());
        EXPECT_EQ(converted.DataPtr(), parameter->DataPtr());
        EXPECT_EQ(converted.Dims(), parameter->Dims());
        ASSERT_NE(converted.grad(), nullptr);
        EXPECT_EQ(converted.grad()->DataPtr(), grad->DataPtr());
        EXPECT_EQ(converted.grad()->Dims(), grad->Dims());
    }
}

TEST_P(TensorCopyTest, NoOpConversionsPreserveFrozenTensorMetadata) {
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{4}, DataType::kFLOAT32, GetDevice());
    for (auto converted : {tensor->To(GetDevice()), tensor->To(tensor->Dtype())}) {
        EXPECT_FALSE(converted.requires_grad());
        EXPECT_FALSE(converted.sequence_parallel());
        EXPECT_EQ(converted.grad(), nullptr);
        EXPECT_EQ(converted.DataPtr(), tensor->DataPtr());
    }
}

INFINI_TRAIN_REGISTER_TEST(TensorCopyTest);
