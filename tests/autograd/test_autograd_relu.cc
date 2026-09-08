#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autograd/activations.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class AutogradReLUTest : public test::InfiniTrainTest {};

TEST_P(AutogradReLUTest, ForwardAndBackward) {
    const std::vector<float> values{-2.0f, -0.0f, 1.5f, 3.0f};
    auto input = std::make_shared<Tensor>(values.data(), std::vector<int64_t>{2, 2}, DataType::kFLOAT32, GetDevice());
    input->RequiresGrad();

    auto output = std::make_shared<autograd::ReLU>()->Apply({input})[0];
    test::ExpectTensorFloatEqual(output, std::vector<float>{0.0f, 0.0f, 1.5f, 3.0f});

    auto grad_output = std::make_shared<Tensor>(output->Dims(), DataType::kFLOAT32, GetDevice());
    grad_output->Fill(1.0f);
    output->Backward(grad_output);
    test::ExpectTensorFloatEqual(input->grad(), std::vector<float>{0.0f, 0.0f, 1.0f, 1.0f});
}

INFINI_TRAIN_REGISTER_TEST(AutogradReLUTest);
