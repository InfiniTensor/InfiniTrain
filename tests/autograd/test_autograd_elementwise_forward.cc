#include <gtest/gtest.h>

#include <vector>
#include <cmath>

#include "infini_train/include/tensor.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/autograd/elementwise.h"
#include "infini_train/include/autograd/activations.h"
#include "test_utils.h"

using namespace infini_train;

class AutogradElementwiseForwardTest : public infini_train::test::AutogradTestBaseP {};

TEST_P(AutogradElementwiseForwardTest, AddForward) {
    auto a = createTensor({2, 3}, 1.0f);
    auto b = createTensor({2, 3}, 2.0f);
    auto add_fn = std::make_shared<autograd::Add>();
    auto result = add_fn->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{2, 3}));
}

TEST_P(AutogradElementwiseForwardTest, SubForward) {
    auto a = createTensor({2, 3}, 5.0f);
    auto b = createTensor({2, 3}, 3.0f);
    auto sub_fn = std::make_shared<autograd::Sub>();
    auto result = sub_fn->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradElementwiseForwardTest, MulForward) {
    auto a = createTensor({2, 3}, 2.0f);
    auto b = createTensor({2, 3}, 3.0f);
    auto mul_fn = std::make_shared<autograd::Mul>();
    auto result = mul_fn->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradElementwiseForwardTest, DivForward) {
    auto a = createTensor({2, 3}, 6.0f);
    auto b = createTensor({2, 3}, 2.0f);
    auto div_fn = std::make_shared<autograd::Div>();
    auto result = div_fn->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradElementwiseForwardTest, NegForward) {
    auto a = createTensor({2, 3}, 5.0f);
    auto neg_fn = std::make_shared<autograd::Neg>();
    auto result = neg_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradElementwiseForwardTest, SinForward) {
    auto a = createTensor({2, 3}, 0.0f);
    auto sin_fn = std::make_shared<autograd::Sin>();
    auto result = sin_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradElementwiseForwardTest, CosForward) {
    auto a = createTensor({2, 3}, 0.0f);
    auto cos_fn = std::make_shared<autograd::Cos>();
    auto result = cos_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradElementwiseForwardTest, TanhForward) {
    auto a = createTensor({2, 3}, 0.0f);
    auto tanh_fn = std::make_shared<autograd::Tanh>();
    auto result = tanh_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradElementwiseForwardTest, ExpForward) {
    auto a = createTensor({2, 3}, 1.0f);
    auto exp_fn = std::make_shared<autograd::Exp>();
    auto result = exp_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradElementwiseForwardTest, LogForward) {
    auto a = createTensor({2, 3}, 2.0f);
    auto log_fn = std::make_shared<autograd::Log>();
    auto result = log_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradElementwiseForwardTest, ReciprocalForward) {
    auto a = createTensor({2, 3}, 2.0f);
    auto reciprocal_fn = std::make_shared<autograd::Reciprocal>();
    auto result = reciprocal_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradElementwiseForwardTest, PowForward) {
    auto a = createTensor({2, 3}, 2.0f);
    auto pow_fn = std::make_shared<autograd::Pow>(2.0f);
    auto result = pow_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradElementwiseForwardTest, RsqrtForward) {
    auto a = createTensor({2, 3}, 4.0f);
    auto rsqrt_fn = std::make_shared<autograd::Rsqrt>();
    auto result = rsqrt_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradElementwiseForwardTest, SigmoidForward) {
    auto a = createTensor({2, 3}, 0.0f);
    auto sigmoid_fn = std::make_shared<autograd::Sigmoid>();
    auto result = sigmoid_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradElementwiseForwardTest, AddScalarForward) {
    auto a = createTensor({2, 3}, 1.0f);
    auto add_scalar_fn = std::make_shared<autograd::AddScalar>(2.0f);
    auto result = add_scalar_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradElementwiseForwardTest, MulScalarForward) {
    auto a = createTensor({2, 3}, 2.0f);
    auto mul_scalar_fn = std::make_shared<autograd::MulScalar>(3.0f);
    auto result = mul_scalar_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradElementwiseForwardTest, LtForward) {
    auto a = createTensor({2, 3}, 5.0f);
    auto b = createTensor({2, 3}, 3.0f);
    auto lt_fn = std::make_shared<autograd::Lt>();
    auto result = lt_fn->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradElementwiseForwardTest, LeForward) {
    auto a = createTensor({2, 3}, 3.0f);
    auto b = createTensor({2, 3}, 3.0f);
    auto le_fn = std::make_shared<autograd::Le>();
    auto result = le_fn->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradElementwiseForwardTest, GtForward) {
    auto a = createTensor({2, 3}, 5.0f);
    auto b = createTensor({2, 3}, 3.0f);
    auto gt_fn = std::make_shared<autograd::Gt>();
    auto result = gt_fn->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradElementwiseForwardTest, GeForward) {
    auto a = createTensor({2, 3}, 3.0f);
    auto b = createTensor({2, 3}, 3.0f);
    auto ge_fn = std::make_shared<autograd::Ge>();
    auto result = ge_fn->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradElementwiseForwardTest, EqualsForward) {
    auto a = createTensor({2, 3}, 3.0f);
    auto b = createTensor({2, 3}, 3.0f);
    auto eq_fn = std::make_shared<autograd::Equals>();
    auto result = eq_fn->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradElementwiseForwardTest, AndForward) {
    auto a = createTensor({2, 3}, 1.0f);
    auto b = createTensor({2, 3}, 1.0f);
    auto and_fn = std::make_shared<autograd::And>();
    auto result = and_fn->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradElementwiseForwardTest, OrForward) {
    auto a = createTensor({2, 3}, 0.0f);
    auto b = createTensor({2, 3}, 1.0f);
    auto or_fn = std::make_shared<autograd::Or>();
    auto result = or_fn->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
}

INFINI_TRAIN_REGISTER_TEST(AutogradElementwiseForwardTest);
