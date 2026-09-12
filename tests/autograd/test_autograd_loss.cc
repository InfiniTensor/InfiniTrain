#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autograd/loss.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class AutogradLossTest : public infini_train::test::InfiniTrainTest {};

TEST_P(AutogradLossTest, CrossEntropyForwardAndBackwardValues) {
    std::vector<float> logits_values{1.0f, 2.0f, 3.0f, 2.0f, 1.0f, 0.0f};
    auto logits
        = std::make_shared<Tensor>(logits_values.data(), std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice());
    auto target = std::make_shared<Tensor>(std::vector<int64_t>{2}, DataType::kINT64, GetDevice());
    target->Fill(0);

    auto cross_entropy = std::make_shared<autograd::CrossEntropy>();
    auto result = cross_entropy->Apply({logits, target});
    ASSERT_EQ(result.size(), 1);
    ASSERT_TRUE(result[0]->Dims().empty());

    const float row0_sum = std::exp(1.0f) + std::exp(2.0f) + std::exp(3.0f);
    const float row1_sum = std::exp(2.0f) + std::exp(1.0f) + std::exp(0.0f);
    const float expected_loss = 0.5f * ((std::log(row0_sum) - 1.0f) + (std::log(row1_sum) - 2.0f));
    test::ExpectTensorNear(result[0], expected_loss, 1e-5f);

    auto grad_output = std::make_shared<Tensor>(std::vector<int64_t>{}, DataType::kFLOAT32, GetDevice());
    grad_output->Fill(1.0f);
    auto grad_inputs = cross_entropy->Backward({grad_output});
    ASSERT_EQ(grad_inputs.size(), 2);
    ASSERT_NE(grad_inputs[0], nullptr);
    EXPECT_EQ(grad_inputs[1], nullptr);

    std::vector<float> expected_grad(logits_values.size());
    for (int row = 0; row < 2; ++row) {
        const float sum = row == 0 ? row0_sum : row1_sum;
        for (int col = 0; col < 3; ++col) {
            const float probability = std::exp(logits_values[row * 3 + col]) / sum;
            expected_grad[row * 3 + col] = 0.5f * (probability - (col == 0 ? 1.0f : 0.0f));
        }
    }
    test::ExpectTensorNear(grad_inputs[0], expected_grad, 1e-5f);
}

INFINI_TRAIN_REGISTER_TEST(AutogradLossTest);
