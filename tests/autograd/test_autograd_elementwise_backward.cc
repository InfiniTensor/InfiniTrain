#include <gtest/gtest.h>

#include <vector>
#include <cmath>

#include "infini_train/include/tensor.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/autograd/elementwise.h"
#include "test_utils.h"

using namespace infini_train;

class AutogradElementwiseBackwardTest : public infini_train::test::AutogradTestBaseP {};

TEST_P(AutogradElementwiseBackwardTest, AddBackward) {
    auto a = createTensor({2, 3}, 1.0f);
    auto b = createTensor({2, 3}, 2.0f);
    auto add_fn = std::make_shared<autograd::Add>();
    auto result = add_fn->Apply({a, b});
    auto grad = createTensor({2, 3}, 1.0f);
    auto grad_inputs = add_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 2);
}

TEST_P(AutogradElementwiseBackwardTest, SubBackward) {
    auto a = createTensor({2, 3}, 5.0f);
    auto b = createTensor({2, 3}, 3.0f);
    auto sub_fn = std::make_shared<autograd::Sub>();
    auto result = sub_fn->Apply({a, b});
    auto grad = createTensor({2, 3}, 1.0f);
    auto grad_inputs = sub_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 2);
}

TEST_P(AutogradElementwiseBackwardTest, MulBackward) {
    auto a = createTensor({2, 3}, 2.0f);
    auto b = createTensor({2, 3}, 3.0f);
    auto mul_fn = std::make_shared<autograd::Mul>();
    auto result = mul_fn->Apply({a, b});
    auto grad = createTensor({2, 3}, 1.0f);
    auto grad_inputs = mul_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 2);
}

TEST_P(AutogradElementwiseBackwardTest, DivBackward) {
    auto a = createTensor({2, 3}, 6.0f);
    auto b = createTensor({2, 3}, 2.0f);
    auto div_fn = std::make_shared<autograd::Div>();
    auto result = div_fn->Apply({a, b});
    auto grad = createTensor({2, 3}, 1.0f);
    auto grad_inputs = div_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 2);
}

TEST_P(AutogradElementwiseBackwardTest, NegBackward) {
    auto a = createTensor({2, 3}, 5.0f);
    auto neg_fn = std::make_shared<autograd::Neg>();
    auto result = neg_fn->Apply({a});
    auto grad = createTensor({2, 3}, 1.0f);
    auto grad_inputs = neg_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 1);
}

TEST_P(AutogradElementwiseBackwardTest, SinBackward) {
    auto a = createTensor({2, 3}, 0.0f);
    auto sin_fn = std::make_shared<autograd::Sin>();
    auto result = sin_fn->Apply({a});
    auto grad = createTensor({2, 3}, 1.0f);
    auto grad_inputs = sin_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 1);
}

TEST_P(AutogradElementwiseBackwardTest, CosBackward) {
    auto a = createTensor({2, 3}, 0.0f);
    auto cos_fn = std::make_shared<autograd::Cos>();
    auto result = cos_fn->Apply({a});
    auto grad = createTensor({2, 3}, 1.0f);
    auto grad_inputs = cos_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 1);
}

TEST_P(AutogradElementwiseBackwardTest, TanhBackward) {
    auto a = createTensor({2, 3}, 0.0f);
    auto tanh_fn = std::make_shared<autograd::Tanh>();
    auto result = tanh_fn->Apply({a});
    auto grad = createTensor({2, 3}, 1.0f);
    auto grad_inputs = tanh_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 1);
}

TEST_P(AutogradElementwiseBackwardTest, ExpBackward) {
    auto a = createTensor({2, 3}, 1.0f);
    auto exp_fn = std::make_shared<autograd::Exp>();
    auto result = exp_fn->Apply({a});
    auto grad = createTensor({2, 3}, 1.0f);
    auto grad_inputs = exp_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 1);
}

TEST_P(AutogradElementwiseBackwardTest, LogBackward) {
    auto a = createTensor({2, 3}, 2.0f);
    auto log_fn = std::make_shared<autograd::Log>();
    auto result = log_fn->Apply({a});
    auto grad = createTensor({2, 3}, 1.0f);
    auto grad_inputs = log_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 1);
}

TEST_P(AutogradElementwiseBackwardTest, ReciprocalBackward) {
    auto a = createTensor({2, 3}, 2.0f);
    auto reciprocal_fn = std::make_shared<autograd::Reciprocal>();
    auto result = reciprocal_fn->Apply({a});
    auto grad = createTensor({2, 3}, 1.0f);
    auto grad_inputs = reciprocal_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 1);
}

TEST_P(AutogradElementwiseBackwardTest, PowBackward) {
    auto a = createTensor({2, 3}, 2.0f);
    auto pow_fn = std::make_shared<autograd::Pow>(2.0f);
    auto result = pow_fn->Apply({a});
    auto grad = createTensor({2, 3}, 1.0f);
    auto grad_inputs = pow_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 1);
}

TEST_P(AutogradElementwiseBackwardTest, RsqrtBackward) {
    auto a = createTensor({2, 3}, 4.0f);
    auto rsqrt_fn = std::make_shared<autograd::Rsqrt>();
    auto result = rsqrt_fn->Apply({a});
    auto grad = createTensor({2, 3}, 1.0f);
    auto grad_inputs = rsqrt_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 1);
}

INFINI_TRAIN_REGISTER_TEST(AutogradElementwiseBackwardTest);
