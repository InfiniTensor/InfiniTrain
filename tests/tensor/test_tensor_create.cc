#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "infini_train/include/tensor.h"
#include "test_utils.h"

using namespace infini_train;

class TensorCreateTest : public infini_train::test::TensorTestBaseP {};

TEST_P(TensorCreateTest, CreatesTensorWithShapeAndType) {
    auto tensor = createTensor({2, 3});
    EXPECT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->Dims(), (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(tensor->Dtype(), DataType::kFLOAT32);
}

TEST_P(TensorCreateTest, TracksRequiresGrad) {
    auto tensor = createTensor({2, 3});
    EXPECT_FALSE(tensor->requires_grad());
    tensor->set_requires_grad(true);
    EXPECT_TRUE(tensor->requires_grad());
}

TEST_P(TensorCreateTest, ProvidesDataPointer) {
    auto tensor = createTensor({2, 3});
    EXPECT_NE(tensor->DataPtr(), nullptr);
}

TEST_P(TensorCreateTest, SupportsMultipleShapes) {
    for (const auto &shape : std::vector<std::vector<int64_t>>{{2, 3}, {4, 5, 6}, {10}, {1, 1, 1, 1}}) {
        auto tensor = createTensor(shape);
        EXPECT_EQ(tensor->Dims(), shape);
    }
}

TEST_P(TensorCreateTest, SupportsMultipleDtypes) {
    for (auto dtype : {DataType::kFLOAT32, DataType::kBFLOAT16}) {
        auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, dtype, GetDevice());
        EXPECT_EQ(tensor->Dtype(), dtype);
    }
}

INFINI_TRAIN_REGISTER_TEST(TensorCreateTest);
