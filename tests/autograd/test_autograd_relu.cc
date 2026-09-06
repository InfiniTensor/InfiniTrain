#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autograd/activations.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

namespace {
std::shared_ptr<Tensor> MakeTensor(const std::vector<int64_t> &dims, const std::vector<float> &values,
                                   const Device &device, bool requires_grad = false) {
    auto cpu_tensor = std::make_shared<Tensor>(dims, DataType::kFLOAT32, Device());
    std::copy(values.begin(), values.end(), static_cast<float *>(cpu_tensor->DataPtr()));
    auto tensor = std::make_shared<Tensor>(cpu_tensor->To(device));
    return requires_grad ? tensor->RequiresGrad() : tensor;
}
} // namespace

class AutogradReluTest : public infini_train::test::InfiniTrainTest {};

TEST_P(AutogradReluTest, ReluForward) {
    auto input = MakeTensor({2, 3}, {-1.0f, 0.0f, 2.5f, -3.0f, 1.5f, -0.25f}, GetDevice());
    auto relu_fn = std::make_shared<autograd::Relu>();
    auto result = relu_fn->Apply({input});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{2, 3}));
    test::ExpectTensorFloatEqual(result[0], {0.0f, 0.0f, 2.5f, 0.0f, 1.5f, 0.0f});
}

TEST_P(AutogradReluTest, ReluBackward) {
    auto output = MakeTensor({2, 3}, {0.0f, 0.0f, 2.5f, 0.0f, 1.5f, 0.0f}, GetDevice());
    auto grad_output = MakeTensor({2, 3}, {1.0f, -2.0f, 0.5f, -1.0f, 3.0f, 7.0f}, GetDevice());

    auto relu_fn = std::make_shared<autograd::Relu>();
    auto input = MakeTensor({2, 3}, {-1.0f, 0.0f, 2.5f, -3.0f, 1.5f, -0.25f}, GetDevice());
    relu_fn->Apply({input});

    const auto grads = relu_fn->Backward({grad_output});
    EXPECT_EQ(grads.size(), 1);
    test::ExpectTensorFloatEqual(grads[0], {0.0f, 0.0f, 0.5f, 0.0f, 3.0f, 0.0f});
}

INFINI_TRAIN_REGISTER_TEST(AutogradReluTest);
