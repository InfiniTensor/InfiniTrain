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

TEST_P(AutogradConv2dTest, BatchedMultiChannelForwardAndBackward) {
    // Fixed PyTorch conv2d reference with nonuniform inputs and upstream gradients.
    const std::vector<float> input_values{-9.0f, -2.0f, 5.0f, -7.0f, 0.0f,  7.0f, -5.0f, 2.0f,
                                          9.0f,  -3.0f, 4.0f, -8.0f, -1.0f, 6.0f, -6.0f, 1.0f,
                                          8.0f,  -4.0f, 3.0f, -9.0f, -2.0f, 5.0f, -7.0f, 0.0f};
    const std::vector<float> weight_values{-5.0f, -2.0f, 1.0f, 4.0f,  -4.0f, -1.0f, 2.0f,  5.0f,
                                           -3.0f, 0.0f,  3.0f, -5.0f, -2.0f, 1.0f,  4.0f,  -4.0f,
                                           -1.0f, 2.0f,  5.0f, -3.0f, 0.0f,  3.0f,  -5.0f, -2.0f};
    const std::vector<float> bias_values{1.0f, -2.0f, 3.0f};
    const std::vector<float> grad_values{
        -6.0f, -1.0f, 4.0f,  -4.0f, 1.0f,  6.0f,  -2.0f, 3.0f,  -5.0f, 0.0f,  5.0f,  -3.0f, 2.0f,  -6.0f, -1.0f,
        4.0f,  -4.0f, 1.0f,  6.0f,  -2.0f, 3.0f,  -5.0f, 0.0f,  5.0f,  -3.0f, 2.0f,  -6.0f, -1.0f, 4.0f,  -4.0f,
        1.0f,  6.0f,  -2.0f, 3.0f,  -5.0f, 0.0f,  5.0f,  -3.0f, 2.0f,  -6.0f, -1.0f, 4.0f,  -4.0f, 1.0f,  6.0f,
        -2.0f, 3.0f,  -5.0f, 0.0f,  5.0f,  -3.0f, 2.0f,  -6.0f, -1.0f, 4.0f,  -4.0f, 1.0f,  6.0f,  -2.0f, 3.0f,
        -5.0f, 0.0f,  5.0f,  -3.0f, 2.0f,  -6.0f, -1.0f, 4.0f,  -4.0f, 1.0f,  6.0f,  -2.0f};
    auto input = std::make_shared<Tensor>(input_values.data(), std::vector<int64_t>{2, 2, 2, 3}, DataType::kFLOAT32,
                                          GetDevice());
    auto weight = std::make_shared<Tensor>(weight_values.data(), std::vector<int64_t>{3, 2, 2, 2}, DataType::kFLOAT32,
                                           GetDevice());
    auto bias = std::make_shared<Tensor>(bias_values.data(), std::vector<int64_t>{3}, DataType::kFLOAT32, GetDevice());
    input->RequiresGrad();
    weight->RequiresGrad();
    bias->RequiresGrad();
    auto output = std::make_shared<autograd::Conv2d>(1, 1)->Apply({input, weight, bias})[0];
    EXPECT_EQ(output->Dims(), (std::vector<int64_t>{2, 3, 3, 4}));
    test::ExpectTensorFloatEqual(
        output, std::vector<float>{
                    -60.0f, -16.0f, 68.0f,  24.0f,  -19.0f, 75.0f,  -20.0f, -69.0f, 18.0f,  44.0f,  -21.0f, -2.0f,
                    63.0f,  -47.0f, -61.0f, 49.0f,  40.0f,  -12.0f, 22.0f,  -46.0f, -5.0f,  29.0f,  -18.0f, -7.0f,
                    40.0f,  -15.0f, -50.0f, -17.0f, -3.0f,  -14.0f, 17.0f,  73.0f,  -20.0f, 22.0f,  -7.0f,  -4.0f,
                    12.0f,  -15.0f, -45.0f, -9.0f,  29.0f,  -1.0f,  -1.0f,  35.0f,  -6.0f,  -33.0f, -3.0f,  21.0f,
                    -9.0f,  13.0f,  18.0f,  -28.0f, -24.0f, -3.0f,  12.0f,  8.0f,   3.0f,   -22.0f, -12.0f, 10.0f,
                    0.0f,   -17.0f, 100.0f, -17.0f, -3.0f,  -41.0f, 66.0f,  -11.0f, 20.0f,  -3.0f,  -13.0f, 7.0f});
    auto grad_output = std::make_shared<Tensor>(grad_values.data(), output->Dims(), DataType::kFLOAT32, GetDevice());
    output->Backward(grad_output);
    test::ExpectTensorFloatEqual(input->grad(),
                                 std::vector<float>{-57.0f, -38.0f, 33.0f, 19.0f,  38.0f, -34.0f, -87.0f, 28.0f,
                                                    52.0f,  61.0f,  33.0f, -86.0f, 42.0f, -30.0f, 2.0f,   -38.0f,
                                                    33.0f,  0.0f,   36.0f, -83.0f, 32.0f, 28.0f,  52.0f,  -41.0f});
    test::ExpectTensorFloatEqual(weight->grad(),
                                 std::vector<float>{-48.0f, 66.0f,  -21.0f, 67.0f,  52.0f,  -23.0f, -40.0f,  106.0f,
                                                    66.0f,  76.0f,  67.0f,  77.0f,  -23.0f, -7.0f,  106.0f,  -125.0f,
                                                    76.0f,  -96.0f, 77.0f,  -95.0f, -7.0f,  74.0f,  -125.0f, -44.0f});
    test::ExpectTensorFloatEqual(bias->grad(), std::vector<float>{-2.0f, 8.0f, -8.0f});
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
