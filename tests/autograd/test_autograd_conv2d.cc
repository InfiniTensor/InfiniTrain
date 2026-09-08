#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autograd/conv2d.h"
#include "infini_train/include/autograd/grad_mode.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class AutogradConv2dTest : public test::InfiniTrainTest {};

TEST_P(AutogradConv2dTest, ForwardWithoutBias) {
    const std::vector<float> input_values{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    const std::vector<float> weight_values{1.0f, 0.0f, 0.0f, -1.0f};
    auto input = std::make_shared<Tensor>(input_values.data(), std::vector<int64_t>{1, 1, 3, 3}, DataType::kFLOAT32,
                                          GetDevice());
    auto weight = std::make_shared<Tensor>(weight_values.data(), std::vector<int64_t>{1, 1, 2, 2}, DataType::kFLOAT32,
                                           GetDevice());

    auto output = std::make_shared<autograd::Conv2d>(1, 0)->Apply({input, weight})[0];
    EXPECT_EQ(output->Dims(), (std::vector<int64_t>{1, 1, 2, 2}));
    test::ExpectTensorFloatEqual(output, std::vector<float>{-4.0f, -4.0f, -4.0f, -4.0f});
}

TEST_P(AutogradConv2dTest, BackwardWithBias) {
    const std::vector<float> input_values{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    const std::vector<float> weight_values{1.0f, 0.0f, 0.0f, -1.0f};
    const std::vector<float> bias_values{2.0f};
    auto input = std::make_shared<Tensor>(input_values.data(), std::vector<int64_t>{1, 1, 3, 3}, DataType::kFLOAT32,
                                          GetDevice());
    auto weight = std::make_shared<Tensor>(weight_values.data(), std::vector<int64_t>{1, 1, 2, 2}, DataType::kFLOAT32,
                                           GetDevice());
    auto bias = std::make_shared<Tensor>(bias_values.data(), std::vector<int64_t>{1}, DataType::kFLOAT32, GetDevice());
    input->RequiresGrad();
    weight->RequiresGrad();
    bias->RequiresGrad();

    auto output = std::make_shared<autograd::Conv2d>(1, 0)->Apply({input, weight, bias})[0];
    test::ExpectTensorFloatEqual(output, std::vector<float>{-2.0f, -2.0f, -2.0f, -2.0f});
    auto grad_output = std::make_shared<Tensor>(output->Dims(), DataType::kFLOAT32, GetDevice());
    grad_output->Fill(1.0f);
    output->Backward(grad_output);

    test::ExpectTensorFloatEqual(input->grad(),
                                 std::vector<float>{1.0f, 1.0f, 0.0f, 1.0f, 0.0f, -1.0f, 0.0f, -1.0f, -1.0f});
    test::ExpectTensorFloatEqual(weight->grad(), std::vector<float>{12.0f, 16.0f, 24.0f, 28.0f});
    test::ExpectTensorFloatEqual(bias->grad(), std::vector<float>{4.0f});
}

TEST_P(AutogradConv2dTest, BackwardSupportsPaddingAndStride) {
    const std::vector<float> input_values{1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> weight_values{1.0f, 1.0f, 1.0f, 1.0f};
    auto input = std::make_shared<Tensor>(input_values.data(), std::vector<int64_t>{1, 1, 2, 2}, DataType::kFLOAT32,
                                          GetDevice());
    auto weight = std::make_shared<Tensor>(weight_values.data(), std::vector<int64_t>{1, 1, 2, 2}, DataType::kFLOAT32,
                                           GetDevice());
    input->RequiresGrad();
    weight->RequiresGrad();

    auto output = std::make_shared<autograd::Conv2d>(2, 1)->Apply({input, weight})[0];
    EXPECT_EQ(output->Dims(), (std::vector<int64_t>{1, 1, 2, 2}));
    test::ExpectTensorFloatEqual(output, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
    auto grad_output = std::make_shared<Tensor>(output->Dims(), DataType::kFLOAT32, GetDevice());
    grad_output->Fill(1.0f);
    auto direct_grad_input = Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
        {GetDevice().type(), "Conv2dBackwardInput"}, weight, grad_output, input->Dims(), 2, 1);
    test::ExpectTensorFloatEqual(direct_grad_input, std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f});
    output->Backward(grad_output);

    test::ExpectTensorFloatEqual(input->grad(), std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f});
    test::ExpectTensorFloatEqual(weight->grad(), std::vector<float>{4.0f, 3.0f, 2.0f, 1.0f});
}

TEST_P(AutogradConv2dTest, SupportsNoGradMode) {
    const std::vector<float> input_values{1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> weight_values{1.0f};
    auto input = std::make_shared<Tensor>(input_values.data(), std::vector<int64_t>{1, 1, 2, 2}, DataType::kFLOAT32,
                                          GetDevice());
    auto weight = std::make_shared<Tensor>(weight_values.data(), std::vector<int64_t>{1, 1, 1, 1}, DataType::kFLOAT32,
                                           GetDevice());
    input->RequiresGrad();
    weight->RequiresGrad();

    autograd::NoGradGuard no_grad;
    auto output = std::make_shared<autograd::Conv2d>(1, 0)->Apply({input, weight})[0];
    EXPECT_EQ(output->grad_fn(), nullptr);
    test::ExpectTensorFloatEqual(output, input_values);
}

INFINI_TRAIN_REGISTER_TEST(AutogradConv2dTest);
