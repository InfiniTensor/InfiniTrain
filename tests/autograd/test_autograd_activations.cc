#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autograd/activations.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class AutogradActivationTest : public infini_train::test::InfiniTrainTest {};

TEST_P(AutogradActivationTest, NewGELUForwardAndBackward) {
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    input->Fill(1.0f);

    auto gelu = std::make_shared<autograd::NewGELU>();
    auto outputs = gelu->Apply({input});
    ASSERT_EQ(outputs.size(), 1);
    test::ExpectTensorNear(outputs[0], 0.84119199f, 1e-6f);

    auto grad_output = std::make_shared<Tensor>(input->Dims(), DataType::kFLOAT32, GetDevice());
    grad_output->Fill(1.0f);
    auto grad_inputs = gelu->Backward({grad_output});
    ASSERT_EQ(grad_inputs.size(), 1);
    test::ExpectTensorNear(grad_inputs[0], 1.08296408f, 1e-6f);
}

INFINI_TRAIN_REGISTER_TEST(AutogradActivationTest);
