#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autograd/linear.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class AutogradLinearForwardTest : public infini_train::test::InfiniTrainTest {};

namespace {
std::shared_ptr<Tensor> CopyToCPU(const std::shared_ptr<Tensor> &tensor) {
    auto cpu = std::make_shared<Tensor>(tensor->Dims(), tensor->Dtype(), Device());
    cpu->CopyFrom(tensor);
    core::GetDeviceGuardImpl(tensor->GetDevice().type())->SynchronizeDevice(tensor->GetDevice());
    return cpu;
}
} // namespace

TEST_P(AutogradLinearForwardTest, LinearForward) {
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    input->Fill(1.0f);
    auto weight = std::make_shared<Tensor>(std::vector<int64_t>{4, 3}, DataType::kFLOAT32, GetDevice(), true);
    weight->Fill(1.0f);
    auto bias = std::make_shared<Tensor>(std::vector<int64_t>{4}, DataType::kFLOAT32, GetDevice(), true);
    bias->Fill(2.0f);
    auto linear_fn = std::make_shared<autograd::Linear>();
    auto result = linear_fn->Apply({input, weight, bias});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{2, 4}));

    auto output = CopyToCPU(result[0]);
    const float *actual = static_cast<const float *>(output->DataPtr());
    for (int64_t i = 0; i < output->NumElements(); ++i) { EXPECT_FLOAT_EQ(actual[i], 5.0f); }
}

TEST_P(AutogradLinearForwardTest, LinearNoBias) {
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    input->Fill(1.0f);
    auto weight = std::make_shared<Tensor>(std::vector<int64_t>{4, 3}, DataType::kFLOAT32, GetDevice(), true);
    weight->Fill(1.0f);
    auto linear_fn = std::make_shared<autograd::Linear>();
    auto result = linear_fn->Apply({input, weight});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{2, 4}));

    auto output = CopyToCPU(result[0]);
    const float *actual = static_cast<const float *>(output->DataPtr());
    for (int64_t i = 0; i < output->NumElements(); ++i) { EXPECT_FLOAT_EQ(actual[i], 3.0f); }
}

TEST_P(AutogradLinearForwardTest, LinearBatch) {
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{32, 128}, DataType::kFLOAT32, GetDevice(), true);
    input->Fill(1.0f);
    auto weight = std::make_shared<Tensor>(std::vector<int64_t>{64, 128}, DataType::kFLOAT32, GetDevice(), true);
    weight->Fill(1.0f);
    auto bias = std::make_shared<Tensor>(std::vector<int64_t>{64}, DataType::kFLOAT32, GetDevice(), true);
    bias->Fill(0.0f);
    auto linear_fn = std::make_shared<autograd::Linear>();
    auto result = linear_fn->Apply({input, weight, bias});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{32, 64}));
}

INFINI_TRAIN_REGISTER_TEST(AutogradLinearForwardTest);
