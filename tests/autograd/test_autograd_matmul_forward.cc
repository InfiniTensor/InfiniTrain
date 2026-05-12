#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autograd/matmul.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class AutogradMatmulForwardTest : public infini_train::test::InfiniTrainTest {};

TEST_P(AutogradMatmulForwardTest, MatmulForward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto b = std::make_shared<Tensor>(std::vector<int64_t>{3, 4}, DataType::kFLOAT32, GetDevice(), true);
    b->Fill(1.0f);
    auto matmul_fn = std::make_shared<autograd::Matmul>();
    auto result = matmul_fn->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{2, 4}));
}

TEST_P(AutogradMatmulForwardTest, MatmulDifferentShapes) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{3, 4}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto b = std::make_shared<Tensor>(std::vector<int64_t>{4, 2}, DataType::kFLOAT32, GetDevice(), true);
    b->Fill(1.0f);
    auto matmul_fn = std::make_shared<autograd::Matmul>();
    auto result = matmul_fn->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{3, 2}));
}

TEST_P(AutogradMatmulForwardTest, MatmulBatch) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3, 4}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto b = std::make_shared<Tensor>(std::vector<int64_t>{2, 4, 5}, DataType::kFLOAT32, GetDevice(), true);
    b->Fill(1.0f);
    auto matmul_fn = std::make_shared<autograd::Matmul>();
    auto result = matmul_fn->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{2, 3, 5}));
}

TEST_P(AutogradMatmulForwardTest, MatmulSquare) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{3, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto b = std::make_shared<Tensor>(std::vector<int64_t>{3, 3}, DataType::kFLOAT32, GetDevice(), true);
    b->Fill(1.0f);
    auto matmul_fn = std::make_shared<autograd::Matmul>();
    auto result = matmul_fn->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{3, 3}));
}

INFINI_TRAIN_REGISTER_TEST(AutogradMatmulForwardTest);
