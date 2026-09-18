#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autograd/activations.h"
#include "infini_train/include/nn/modules/activations.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class AutogradReLUTest : public infini_train::test::InfiniTrainTest {};

TEST_P(AutogradReLUTest, ReLUForward) {
    std::vector<float> input_values{-2.0f, -0.0f, 0.0f, 0.5f, 1.0f, 3.0f};
    auto a = std::make_shared<Tensor>(input_values.data(), std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice());
    auto relu_fn = std::make_shared<autograd::ReLU>();
    auto result = relu_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{2, 3}));
    test::ExpectTensorFloatEqual(result[0], {0.0f, 0.0f, 0.0f, 0.5f, 1.0f, 3.0f});
}

TEST_P(AutogradReLUTest, ReLUBackward) {
    std::vector<float> input_values{-2.0f, -0.0f, 0.0f, 0.5f, 1.0f, 3.0f};
    auto a = std::make_shared<Tensor>(input_values.data(), std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice());
    auto relu_fn = std::make_shared<autograd::ReLU>();
    auto result = relu_fn->Apply({a});
    std::vector<float> grad_values{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto grad
        = std::make_shared<Tensor>(grad_values.data(), std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice());
    auto grad_inputs = relu_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 1);
    test::ExpectTensorFloatEqual(grad_inputs[0], {0.0f, 0.0f, 0.0f, 4.0f, 5.0f, 6.0f});
}

TEST_P(AutogradReLUTest, ReLUModuleForward) {
    std::vector<float> input_values{-1.0f, 0.0f, 2.0f};
    auto a = std::make_shared<Tensor>(input_values.data(), std::vector<int64_t>{3}, DataType::kFLOAT32, GetDevice());
    nn::ReLU relu;
    auto result = relu.Forward({a});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{3}));
    test::ExpectTensorFloatEqual(result[0], {0.0f, 0.0f, 2.0f});
}

INFINI_TRAIN_REGISTER_TEST(AutogradReLUTest);
