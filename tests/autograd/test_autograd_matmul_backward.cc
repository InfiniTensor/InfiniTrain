#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autograd/matmul.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class AutogradMatmulBackwardTest : public infini_train::test::InfiniTrainTest {};

TEST_P(AutogradMatmulBackwardTest, MatmulBackward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto b = std::make_shared<Tensor>(std::vector<int64_t>{3, 4}, DataType::kFLOAT32, GetDevice(), true);
    b->Fill(1.0f);
    auto matmul_fn = std::make_shared<autograd::Matmul>();
    auto result = matmul_fn->Apply({a, b});
    auto grad = std::make_shared<Tensor>(std::vector<int64_t>{2, 4}, DataType::kFLOAT32, GetDevice(), true);
    grad->Fill(1.0f);
    auto grad_inputs = matmul_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 2);
}

TEST_P(AutogradMatmulBackwardTest, MatmulBackwardSquare) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{3, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(2.0f);
    auto b = std::make_shared<Tensor>(std::vector<int64_t>{3, 3}, DataType::kFLOAT32, GetDevice(), true);
    b->Fill(3.0f);
    auto matmul_fn = std::make_shared<autograd::Matmul>();
    auto result = matmul_fn->Apply({a, b});
    auto grad = std::make_shared<Tensor>(std::vector<int64_t>{3, 3}, DataType::kFLOAT32, GetDevice(), true);
    grad->Fill(1.0f);
    auto grad_inputs = matmul_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 2);
}

TEST_P(AutogradMatmulBackwardTest, MatmulBackwardDifferentShapes) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{3, 4}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.5f);
    auto b = std::make_shared<Tensor>(std::vector<int64_t>{4, 2}, DataType::kFLOAT32, GetDevice(), true);
    b->Fill(2.5f);
    auto matmul_fn = std::make_shared<autograd::Matmul>();
    auto result = matmul_fn->Apply({a, b});
    auto grad = std::make_shared<Tensor>(std::vector<int64_t>{3, 2}, DataType::kFLOAT32, GetDevice(), true);
    grad->Fill(1.0f);
    auto grad_inputs = matmul_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 2);
}

INFINI_TRAIN_REGISTER_TEST(AutogradMatmulBackwardTest);
