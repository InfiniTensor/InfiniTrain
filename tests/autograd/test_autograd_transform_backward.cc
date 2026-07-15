#include <numeric>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autograd/transform.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class AutogradTransformBackwardTest : public infini_train::test::InfiniTrainTest {};

namespace {
std::shared_ptr<Tensor> CopyToCPU(const std::shared_ptr<Tensor> &tensor) {
    auto cpu = std::make_shared<Tensor>(tensor->Dims(), tensor->Dtype(), Device());
    cpu->CopyFrom(tensor);
    core::GetDeviceGuardImpl(tensor->GetDevice().type())->SynchronizeDevice(tensor->GetDevice());
    return cpu;
}

void ExpectValues(const std::shared_ptr<Tensor> &tensor, const std::vector<float> &expected) {
    auto cpu = CopyToCPU(tensor);
    ASSERT_EQ(cpu->NumElements(), expected.size());
    const float *actual = static_cast<const float *>(cpu->DataPtr());
    for (size_t i = 0; i < expected.size(); ++i) { EXPECT_FLOAT_EQ(actual[i], expected[i]); }
}
} // namespace

TEST_P(AutogradTransformBackwardTest, TransposeBackward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto transpose_fn = std::make_shared<autograd::Transpose>(0, 1);
    auto result = transpose_fn->Apply({a});
    auto grad = std::make_shared<Tensor>(std::vector<int64_t>{3, 2}, DataType::kFLOAT32, GetDevice(), true);
    grad->Fill(1.0f);
    auto grad_inputs = transpose_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 1);
}

TEST_P(AutogradTransformBackwardTest, SplitBackwardValues) {
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 5}, DataType::kFLOAT32, GetDevice(), true);
    auto split = std::make_shared<autograd::Split>(2, 1);
    auto outputs = split->Apply({input});
    ASSERT_EQ(outputs.size(), 3);

    std::vector<std::shared_ptr<Tensor>> grad_outputs;
    for (size_t i = 0; i < outputs.size(); ++i) {
        auto grad = std::make_shared<Tensor>(outputs[i]->Dims(), DataType::kFLOAT32, GetDevice());
        grad->Fill(static_cast<float>(i + 1));
        grad_outputs.push_back(grad);
    }

    auto grad_inputs = split->Backward(grad_outputs);
    ASSERT_EQ(grad_inputs.size(), 1);
    ExpectValues(grad_inputs[0], {1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 1.0f, 1.0f, 2.0f, 2.0f, 3.0f});
}

TEST_P(AutogradTransformBackwardTest, StackBackwardValues) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    auto b = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    auto stack = std::make_shared<autograd::Stack>(1);
    auto outputs = stack->Apply({a, b});
    ASSERT_EQ(outputs.size(), 1);

    std::vector<float> grad_values(12);
    std::iota(grad_values.begin(), grad_values.end(), 0.0f);
    auto grad_output
        = std::make_shared<Tensor>(grad_values.data(), outputs[0]->Dims(), DataType::kFLOAT32, GetDevice());
    auto grad_inputs = stack->Backward({grad_output});
    ASSERT_EQ(grad_inputs.size(), 2);
    ExpectValues(grad_inputs[0], {0.0f, 1.0f, 2.0f, 6.0f, 7.0f, 8.0f});
    ExpectValues(grad_inputs[1], {3.0f, 4.0f, 5.0f, 9.0f, 10.0f, 11.0f});
}

TEST_P(AutogradTransformBackwardTest, ConcatBackwardValues) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 2}, DataType::kFLOAT32, GetDevice(), true);
    auto b = std::make_shared<Tensor>(std::vector<int64_t>{2, 1}, DataType::kFLOAT32, GetDevice(), true);
    auto concat = std::make_shared<autograd::Concat>(1);
    auto outputs = concat->Apply({a, b});
    ASSERT_EQ(outputs.size(), 1);

    std::vector<float> grad_values(6);
    std::iota(grad_values.begin(), grad_values.end(), 0.0f);
    auto grad_output
        = std::make_shared<Tensor>(grad_values.data(), outputs[0]->Dims(), DataType::kFLOAT32, GetDevice());
    auto grad_inputs = concat->Backward({grad_output});
    ASSERT_EQ(grad_inputs.size(), 2);
    ExpectValues(grad_inputs[0], {0.0f, 1.0f, 3.0f, 4.0f});
    ExpectValues(grad_inputs[1], {2.0f, 5.0f});
}

INFINI_TRAIN_REGISTER_TEST(AutogradTransformBackwardTest);
