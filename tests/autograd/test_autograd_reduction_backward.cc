#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autograd/reduction.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class AutogradReductionBackwardTest : public infini_train::test::InfiniTrainTest {};

namespace {
void ExpectValues(const std::shared_ptr<Tensor> &tensor, const std::vector<float> &expected) {
    auto cpu = std::make_shared<Tensor>(tensor->Dims(), tensor->Dtype(), Device());
    cpu->CopyFrom(tensor);
    core::GetDeviceGuardImpl(tensor->GetDevice().type())->SynchronizeDevice(tensor->GetDevice());

    ASSERT_EQ(cpu->NumElements(), expected.size());
    const float *actual = static_cast<const float *>(cpu->DataPtr());
    for (size_t i = 0; i < expected.size(); ++i) { EXPECT_FLOAT_EQ(actual[i], expected[i]); }
}
} // namespace

TEST_P(AutogradReductionBackwardTest, SumBackward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto sum_fn = std::make_shared<autograd::Sum>(1, false);
    auto result = sum_fn->Apply({a});
    auto grad = std::make_shared<Tensor>(std::vector<int64_t>{2}, DataType::kFLOAT32, GetDevice(), true);
    grad->Fill(1.0f);
    auto grad_inputs = sum_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 1);
    ExpectValues(grad_inputs[0], {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
}

TEST_P(AutogradReductionBackwardTest, MeanBackward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto mean_fn = std::make_shared<autograd::Mean>(1, false);
    auto result = mean_fn->Apply({a});
    auto grad = std::make_shared<Tensor>(std::vector<int64_t>{2}, DataType::kFLOAT32, GetDevice(), true);
    grad->Fill(1.0f);
    auto grad_inputs = mean_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 1);
    ExpectValues(grad_inputs[0], {1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f});
}

TEST_P(AutogradReductionBackwardTest, MaxBackward) {
    std::vector<float> values{1.0f, 3.0f, 2.0f, 4.0f, 0.0f, -1.0f};
    auto a = std::make_shared<Tensor>(values.data(), std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice());
    auto max_fn = std::make_shared<autograd::Max>(1, false);
    auto result = max_fn->Apply({a});
    auto grad = std::make_shared<Tensor>(std::vector<int64_t>{2}, DataType::kFLOAT32, GetDevice(), true);
    grad->Fill(1.0f);
    auto grad_inputs = max_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 1);
    ExpectValues(grad_inputs[0], {0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f});
}

TEST_P(AutogradReductionBackwardTest, MinBackward) {
    std::vector<float> values{1.0f, 3.0f, 2.0f, 4.0f, 0.0f, -1.0f};
    auto a = std::make_shared<Tensor>(values.data(), std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice());
    auto min_fn = std::make_shared<autograd::Min>(1, false);
    auto result = min_fn->Apply({a});
    auto grad = std::make_shared<Tensor>(std::vector<int64_t>{2}, DataType::kFLOAT32, GetDevice(), true);
    grad->Fill(1.0f);
    auto grad_inputs = min_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 1);
    ExpectValues(grad_inputs[0], {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f});
}

TEST_P(AutogradReductionBackwardTest, SumBackwardKeepDim) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto sum_fn = std::make_shared<autograd::Sum>(1, true);
    auto result = sum_fn->Apply({a});
    auto grad = std::make_shared<Tensor>(std::vector<int64_t>{2, 1}, DataType::kFLOAT32, GetDevice(), true);
    grad->Fill(1.0f);
    auto grad_inputs = sum_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 1);
    ExpectValues(grad_inputs[0], {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
}

TEST_P(AutogradReductionBackwardTest, MeanBackwardKeepDim) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto mean_fn = std::make_shared<autograd::Mean>(1, true);
    auto result = mean_fn->Apply({a});
    auto grad = std::make_shared<Tensor>(std::vector<int64_t>{2, 1}, DataType::kFLOAT32, GetDevice(), true);
    grad->Fill(1.0f);
    auto grad_inputs = mean_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 1);
    ExpectValues(grad_inputs[0], {1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f});
}

INFINI_TRAIN_REGISTER_TEST(AutogradReductionBackwardTest);
