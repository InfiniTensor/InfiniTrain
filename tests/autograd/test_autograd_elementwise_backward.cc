#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autograd/elementwise.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class AutogradElementwiseBackwardTest : public infini_train::test::InfiniTrainTest {};

TEST_P(AutogradElementwiseBackwardTest, AddBackward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto b = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    b->Fill(2.0f);
    auto add_fn = std::make_shared<autograd::Add>();
    auto result = add_fn->Apply({a, b});
    auto grad = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    grad->Fill(1.0f);
    auto grad_inputs = add_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 2);
}

TEST_P(AutogradElementwiseBackwardTest, SubBackward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(5.0f);
    auto b = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    b->Fill(3.0f);
    auto sub_fn = std::make_shared<autograd::Sub>();
    auto result = sub_fn->Apply({a, b});
    auto grad = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    grad->Fill(1.0f);
    auto grad_inputs = sub_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 2);
}

TEST_P(AutogradElementwiseBackwardTest, MulBackward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(2.0f);
    auto b = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    b->Fill(3.0f);
    auto mul_fn = std::make_shared<autograd::Mul>();
    auto result = mul_fn->Apply({a, b});
    auto grad = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    grad->Fill(1.0f);
    auto grad_inputs = mul_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 2);
}

TEST_P(AutogradElementwiseBackwardTest, Float32MulBroadcastBackwardAcrossLogicalWarps) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 64}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto b = std::make_shared<Tensor>(std::vector<int64_t>{2, 1}, DataType::kFLOAT32, GetDevice(), true);
    b->Fill(2.0f);
    auto mul_fn = std::make_shared<autograd::Mul>();
    auto result = mul_fn->Apply({a, b});
    auto grad = std::make_shared<Tensor>(std::vector<int64_t>{2, 64}, DataType::kFLOAT32, GetDevice(), true);
    grad->Fill(1.0f);

    auto grad_inputs = mul_fn->Backward({grad});
    ASSERT_EQ(grad_inputs.size(), 2);

    test::ExpectTensorFloatEqual(grad_inputs[0], 2.0f);
    test::ExpectTensorFloatEqual(grad_inputs[1], std::vector<float>{64.0f, 64.0f});
}

TEST_P(AutogradElementwiseBackwardTest, Float32MulBroadcastBackwardPartialLogicalWarps) {
    // A single row covers 31/33-element tails; multiple rows also exercise nonzero B offsets
    // and logical warps that straddle rows with different B offsets.
    for (int64_t rows : {1, 3}) {
        for (int64_t cols : {31, 33}) {
            SCOPED_TRACE(::testing::Message() << "rows=" << rows << ", cols=" << cols);
            const std::vector<int64_t> a_dims{rows, cols};
            const std::vector<int64_t> b_dims{rows, 1};
            std::vector<float> a_values(rows * cols), b_values(rows), grad_values(rows * cols);
            std::vector<float> expected_grad_a(rows * cols), expected_grad_b(rows, 0.0f);
            for (int64_t row = 0; row < rows; ++row) {
                b_values[row] = static_cast<float>(row + 2);
                for (int64_t col = 0; col < cols; ++col) {
                    const int64_t idx = row * cols + col;
                    a_values[idx] = static_cast<float>(idx + 1);
                    grad_values[idx] = static_cast<float>(col % 3 + 1);
                    expected_grad_a[idx] = grad_values[idx] * b_values[row];
                    expected_grad_b[row] += grad_values[idx] * a_values[idx];
                }
            }

            auto a = std::make_shared<Tensor>(a_values.data(), a_dims, DataType::kFLOAT32, GetDevice());
            auto b = std::make_shared<Tensor>(b_values.data(), b_dims, DataType::kFLOAT32, GetDevice());
            auto mul_fn = std::make_shared<autograd::Mul>();
            auto result = mul_fn->Apply({a, b});
            auto grad = std::make_shared<Tensor>(grad_values.data(), a_dims, DataType::kFLOAT32, GetDevice());
            auto grad_inputs = mul_fn->Backward({grad});
            ASSERT_EQ(grad_inputs.size(), 2);
            EXPECT_EQ(grad_inputs[0]->Dims(), a_dims);
            EXPECT_EQ(grad_inputs[1]->Dims(), b_dims);
            test::ExpectTensorFloatEqual(grad_inputs[0], expected_grad_a);
            test::ExpectTensorFloatEqual(grad_inputs[1], expected_grad_b);
        }
    }
}

TEST_P(AutogradElementwiseBackwardTest, BFloat16MulBroadcastBackwardLargeBlock) {
    ONLY_CUDA();
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{512, 8192}, DataType::kBFLOAT16, GetDevice(), true);
    a->Fill(2.0f);
    auto b = std::make_shared<Tensor>(std::vector<int64_t>{8192}, DataType::kBFLOAT16, GetDevice(), true);
    b->Fill(3.0f);
    auto mul_fn = std::make_shared<autograd::Mul>();
    auto result = mul_fn->Apply({a, b});
    auto grad = std::make_shared<Tensor>(std::vector<int64_t>{512, 8192}, DataType::kBFLOAT16, GetDevice(), true);
    grad->Fill(1.0f);
    auto grad_inputs = mul_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 2);
    EXPECT_EQ(grad_inputs[0]->Dims(), (std::vector<int64_t>{512, 8192}));
    EXPECT_EQ(grad_inputs[1]->Dims(), (std::vector<int64_t>{8192}));
}

TEST_P(AutogradElementwiseBackwardTest, DivBackward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(6.0f);
    auto b = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    b->Fill(2.0f);
    auto div_fn = std::make_shared<autograd::Div>();
    auto result = div_fn->Apply({a, b});
    auto grad = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    grad->Fill(1.0f);
    auto grad_inputs = div_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 2);
}

TEST_P(AutogradElementwiseBackwardTest, NegBackward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(5.0f);
    auto neg_fn = std::make_shared<autograd::Neg>();
    auto result = neg_fn->Apply({a});
    auto grad = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    grad->Fill(1.0f);
    auto grad_inputs = neg_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 1);
}

TEST_P(AutogradElementwiseBackwardTest, SinBackward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(0.0f);
    auto sin_fn = std::make_shared<autograd::Sin>();
    auto result = sin_fn->Apply({a});
    auto grad = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    grad->Fill(1.0f);
    auto grad_inputs = sin_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 1);
}

TEST_P(AutogradElementwiseBackwardTest, CosBackward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(0.0f);
    auto cos_fn = std::make_shared<autograd::Cos>();
    auto result = cos_fn->Apply({a});
    auto grad = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    grad->Fill(1.0f);
    auto grad_inputs = cos_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 1);
}

TEST_P(AutogradElementwiseBackwardTest, TanhBackward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(0.0f);
    auto tanh_fn = std::make_shared<autograd::Tanh>();
    auto result = tanh_fn->Apply({a});
    auto grad = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    grad->Fill(1.0f);
    auto grad_inputs = tanh_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 1);
}

TEST_P(AutogradElementwiseBackwardTest, ExpBackward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto exp_fn = std::make_shared<autograd::Exp>();
    auto result = exp_fn->Apply({a});
    auto grad = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    grad->Fill(1.0f);
    auto grad_inputs = exp_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 1);
}

TEST_P(AutogradElementwiseBackwardTest, LogBackward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(2.0f);
    auto log_fn = std::make_shared<autograd::Log>();
    auto result = log_fn->Apply({a});
    auto grad = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    grad->Fill(1.0f);
    auto grad_inputs = log_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 1);
}

TEST_P(AutogradElementwiseBackwardTest, ReciprocalBackward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(2.0f);
    auto reciprocal_fn = std::make_shared<autograd::Reciprocal>();
    auto result = reciprocal_fn->Apply({a});
    auto grad = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    grad->Fill(1.0f);
    auto grad_inputs = reciprocal_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 1);
}

TEST_P(AutogradElementwiseBackwardTest, PowBackward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(2.0f);
    auto pow_fn = std::make_shared<autograd::Pow>(2.0f);
    auto result = pow_fn->Apply({a});
    auto grad = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    grad->Fill(1.0f);
    auto grad_inputs = pow_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 1);
}

TEST_P(AutogradElementwiseBackwardTest, RsqrtBackward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(4.0f);
    auto rsqrt_fn = std::make_shared<autograd::Rsqrt>();
    auto result = rsqrt_fn->Apply({a});
    auto grad = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    grad->Fill(1.0f);
    auto grad_inputs = rsqrt_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 1);
}

INFINI_TRAIN_REGISTER_TEST(AutogradElementwiseBackwardTest);
