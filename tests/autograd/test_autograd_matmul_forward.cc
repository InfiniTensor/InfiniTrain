#include <gtest/gtest.h>

#include <vector>

#include "infini_train/include/autograd/matmul.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"
#include "test_utils.h"

using namespace infini_train;

class AutogradMatmulForwardTest : public infini_train::test::AutogradTestBase {};

TEST_P(AutogradMatmulForwardTest, MatmulForward) {
    auto a = createTensor({2, 3}, 1.0f);
    auto b = createTensor({3, 4}, 1.0f);
    auto matmul_fn = std::make_shared<autograd::Matmul>();
    auto result = matmul_fn->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{2, 4}));
}

TEST_P(AutogradMatmulForwardTest, MatmulDifferentShapes) {
    auto a = createTensor({3, 4}, 1.0f);
    auto b = createTensor({4, 2}, 1.0f);
    auto matmul_fn = std::make_shared<autograd::Matmul>();
    auto result = matmul_fn->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{3, 2}));
}

TEST_P(AutogradMatmulForwardTest, MatmulBatch) {
    auto a = createTensor({2, 3, 4}, 1.0f);
    auto b = createTensor({2, 4, 5}, 1.0f);
    auto matmul_fn = std::make_shared<autograd::Matmul>();
    auto result = matmul_fn->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{2, 3, 5}));
}

TEST_P(AutogradMatmulForwardTest, MatmulSquare) {
    auto a = createTensor({3, 3}, 1.0f);
    auto b = createTensor({3, 3}, 1.0f);
    auto matmul_fn = std::make_shared<autograd::Matmul>();
    auto result = matmul_fn->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{3, 3}));
}

INFINI_TRAIN_REGISTER_TEST(AutogradMatmulForwardTest);
