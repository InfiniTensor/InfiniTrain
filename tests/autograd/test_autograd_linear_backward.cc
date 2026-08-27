#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autograd/linear.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class AutogradLinearBackwardTest : public infini_train::test::InfiniTrainTest {};

TEST_P(AutogradLinearBackwardTest, LinearBackward) {
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    input->Fill(1.0f);
    auto weight = std::make_shared<Tensor>(std::vector<int64_t>{4, 3}, DataType::kFLOAT32, GetDevice(), true);
    weight->Fill(1.0f);
    auto bias = std::make_shared<Tensor>(std::vector<int64_t>{4}, DataType::kFLOAT32, GetDevice(), true);
    bias->Fill(0.0f);
    auto linear_fn = std::make_shared<autograd::Linear>();
    auto result = linear_fn->Apply({input, weight, bias});
    const float grad_values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    auto grad = std::make_shared<Tensor>(grad_values, std::vector<int64_t>{2, 4}, DataType::kFLOAT32, GetDevice());
    auto grad_inputs = linear_fn->Backward({grad});
    ASSERT_EQ(grad_inputs.size(), 3);
    ASSERT_NE(grad_inputs[2], nullptr);

    auto bias_grad_cpu = grad_inputs[2]->To(Device());
    core::GetDeviceGuardImpl(GetDevice().type())->SynchronizeDevice(GetDevice());
    const auto *bias_grad = static_cast<const float *>(bias_grad_cpu.DataPtr());
    const float expected_bias_grad[] = {6.0f, 8.0f, 10.0f, 12.0f};
    for (int idx = 0; idx < 4; ++idx) { EXPECT_FLOAT_EQ(bias_grad[idx], expected_bias_grad[idx]); }
}

TEST_P(AutogradLinearBackwardTest, LinearBackwardNoBias) {
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    input->Fill(1.0f);
    auto weight = std::make_shared<Tensor>(std::vector<int64_t>{4, 3}, DataType::kFLOAT32, GetDevice(), true);
    weight->Fill(1.0f);
    auto linear_fn = std::make_shared<autograd::Linear>();
    auto result = linear_fn->Apply({input, weight});
    auto grad = std::make_shared<Tensor>(std::vector<int64_t>{2, 4}, DataType::kFLOAT32, GetDevice(), true);
    grad->Fill(1.0f);
    auto grad_inputs = linear_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 2);
}

INFINI_TRAIN_REGISTER_TEST(AutogradLinearBackwardTest);
