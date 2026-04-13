#include <gtest/gtest.h>

#include <vector>

#include "infini_train/include/tensor.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/autograd/matmul.h"
#include "test_utils.h"

using namespace infini_train;

class AutogradMatmulBackwardTest : public infini_train::test::AutogradTestBaseP {};

TEST_P(AutogradMatmulBackwardTest, MatmulBackward) {
    auto a = createTensor({2, 3}, 1.0f);
    auto b = createTensor({3, 4}, 1.0f);
    auto matmul_fn = std::make_shared<autograd::Matmul>();
    auto result = matmul_fn->Apply({a, b});
    auto grad = createTensor({2, 4}, 1.0f);
    auto grad_inputs = matmul_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 2);
}

TEST_P(AutogradMatmulBackwardTest, MatmulBackwardSquare) {
    auto a = createTensor({3, 3}, 2.0f);
    auto b = createTensor({3, 3}, 3.0f);
    auto matmul_fn = std::make_shared<autograd::Matmul>();
    auto result = matmul_fn->Apply({a, b});
    auto grad = createTensor({3, 3}, 1.0f);
    auto grad_inputs = matmul_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 2);
}

TEST_P(AutogradMatmulBackwardTest, MatmulBackwardDifferentShapes) {
    auto a = createTensor({3, 4}, 1.5f);
    auto b = createTensor({4, 2}, 2.5f);
    auto matmul_fn = std::make_shared<autograd::Matmul>();
    auto result = matmul_fn->Apply({a, b});
    auto grad = createTensor({3, 2}, 1.0f);
    auto grad_inputs = matmul_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 2);
}

INFINI_TRAIN_REGISTER_TEST(AutogradMatmulBackwardTest);
