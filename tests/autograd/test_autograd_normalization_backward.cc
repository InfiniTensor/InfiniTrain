#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autograd/normalization.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class AutogradNormalizationBackwardTest : public infini_train::test::InfiniTrainTest {};

namespace {
void ExpectValues(const std::shared_ptr<Tensor> &tensor, float expected) {
    auto cpu = std::make_shared<Tensor>(tensor->Dims(), tensor->Dtype(), Device());
    cpu->CopyFrom(tensor);
    core::GetDeviceGuardImpl(tensor->GetDevice().type())->SynchronizeDevice(tensor->GetDevice());

    const float *actual = static_cast<const float *>(cpu->DataPtr());
    for (int64_t i = 0; i < cpu->NumElements(); ++i) { EXPECT_NEAR(actual[i], expected, 1e-5f); }
}
} // namespace

TEST_P(AutogradNormalizationBackwardTest, LayerNormBackward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3, 4}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto weight = std::make_shared<Tensor>(std::vector<int64_t>{4}, DataType::kFLOAT32, GetDevice(), true);
    weight->Fill(1.0f);
    auto bias = std::make_shared<Tensor>(std::vector<int64_t>{4}, DataType::kFLOAT32, GetDevice(), true);
    bias->Fill(0.0f);
    auto layernorm_fn = std::make_shared<autograd::LayerNorm>(1e-5f);
    auto result = layernorm_fn->Apply({a, weight, bias});
    auto grad = std::make_shared<Tensor>(std::vector<int64_t>{2, 3, 4}, DataType::kFLOAT32, GetDevice(), true);
    grad->Fill(1.0f);
    auto grad_inputs = layernorm_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 3);
    ExpectValues(grad_inputs[0], 0.0f);
    ExpectValues(grad_inputs[1], 0.0f);
    ExpectValues(grad_inputs[2], 6.0f);
}

TEST_P(AutogradNormalizationBackwardTest, LayerNormBackwardZeroBias) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3, 4}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto weight = std::make_shared<Tensor>(std::vector<int64_t>{4}, DataType::kFLOAT32, GetDevice(), true);
    weight->Fill(1.0f);
    auto bias = std::make_shared<Tensor>(std::vector<int64_t>{4}, DataType::kFLOAT32, GetDevice(), true);
    bias->Fill(0.0f);
    auto layernorm_fn = std::make_shared<autograd::LayerNorm>(1e-5f);
    auto result = layernorm_fn->Apply({a, weight, bias});
    auto grad = std::make_shared<Tensor>(std::vector<int64_t>{2, 3, 4}, DataType::kFLOAT32, GetDevice(), true);
    grad->Fill(1.0f);
    auto grad_inputs = layernorm_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 3);
}

INFINI_TRAIN_REGISTER_TEST(AutogradNormalizationBackwardTest);
