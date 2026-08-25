#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autograd/elementwise.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

namespace {

void ExpectExpGradient(const std::shared_ptr<Tensor> &actual, const std::vector<float> &input_values,
                       const std::vector<float> &grad_values, Device expected_device) {
    ASSERT_NE(actual, nullptr);
    ASSERT_EQ(input_values.size(), grad_values.size());
    ASSERT_EQ(actual->NumElements(), input_values.size());
    EXPECT_EQ(actual->Dtype(), DataType::kFLOAT32);
    EXPECT_EQ(actual->GetDevice(), expected_device);

    auto actual_cpu = actual->To(Device());
    core::GetDeviceGuardImpl(actual->GetDevice().type())->SynchronizeDevice(actual->GetDevice());
    const auto *actual_data = static_cast<const float *>(actual_cpu.DataPtr());
    for (size_t idx = 0; idx < input_values.size(); ++idx) {
        const float expected = grad_values[idx] * std::exp(input_values[idx]);
        EXPECT_NEAR(actual_data[idx], expected, 1e-5f) << "Mismatch at index " << idx;
    }
}

} // namespace

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
    ASSERT_EQ(grad_inputs.size(), 2);
    EXPECT_NE(grad_inputs[0].get(), grad_inputs[1].get());
    EXPECT_NE(grad_inputs[0]->DataPtr(), grad_inputs[1]->DataPtr());
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

TEST_P(AutogradElementwiseBackwardTest, BFloat16MulBroadcastBackwardLargeBlock) {
    SKIP_CPU();
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
    const std::vector<float> input_values = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f};
    const std::vector<float> grad_values = {0.25f, -0.5f, 1.0f, 1.5f, -2.0f, 0.125f};
    auto a = std::make_shared<Tensor>(input_values.data(), std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice());
    a->RequiresGrad();
    auto exp_fn = std::make_shared<autograd::Exp>();
    auto result = exp_fn->Apply({a});
    ASSERT_EQ(result.size(), 1);
    auto grad
        = std::make_shared<Tensor>(grad_values.data(), std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice());
    auto grad_inputs = exp_fn->Backward({grad});
    ASSERT_EQ(grad_inputs.size(), 1);
    ExpectExpGradient(grad_inputs[0], input_values, grad_values, GetDevice());
}

TEST_P(AutogradElementwiseBackwardTest, ExpBackwardAccumulatesIntoLeaf) {
    const std::vector<float> input_values = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f};
    const std::vector<float> grad_values = {0.25f, -0.5f, 1.0f, 1.5f, -2.0f, 0.125f};
    auto input
        = std::make_shared<Tensor>(input_values.data(), std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice());
    input->RequiresGrad();
    auto output = input->Exp();
    auto grad
        = std::make_shared<Tensor>(grad_values.data(), std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice());

    output->Backward(grad);

    ExpectExpGradient(input->grad(), input_values, grad_values, GetDevice());
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
