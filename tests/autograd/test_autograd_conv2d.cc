#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autograd/conv2d.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"


using namespace infini_train;

namespace {
// 在CPU上按值构造张量，再搬运到设备，最后打开requires_grad
std::shared_ptr<Tensor> MakeTensor(const std::vector<int64_t> &dims, const std::vector<float> &values, Device device) {
    auto cpu_tensor = std::make_shared<Tensor>(dims, DataType::kFLOAT32);
    std::copy(values.begin(), values.end(), static_cast<float *>(cpu_tensor->DataPtr()));

    return std::make_shared<Tensor>(cpu_tensor->To(device))->RequiresGrad();

}

}


class AutogradConv2dForwardTest : public infini_train::test::InfiniTrainTest{};

// 前向：1*1*3*3， 1个2*2卷积核, stride=1, padding=0, 带bias
// input = [[1, 2, 3], [4, 5, 6], [7, 8, 9]] weight = [[1, 2], [3, 4]], bias = 1
// out =[38， 48, 68, 78】
TEST_P(AutogradConv2dForwardTest, Basic) {
    auto input = MakeTensor({1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9}, GetDevice());
    auto weight = MakeTensor({1, 1, 2, 2}, {1, 2, 3, 4}, GetDevice());
    auto bias = MakeTensor({1}, {1.0f}, GetDevice());

    auto conv_fn = std::make_shared<autograd::Conv2d>(1, 0); //stride = 1, padding = 0
    auto result = conv_fn->Apply({input, weight, bias});

    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{1, 1, 2, 2}));
    test::ExpectTensorNear(result[0], std::vector<float>{38, 48, 68, 78}, 1e-5f);
}


TEST_P(AutogradConv2dForwardTest, MultiChannelWithPadding) {
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{1, 2, 3, 3}, DataType::kFLOAT32, GetDevice(), true);
    input->Fill(1.0f);
    auto weight = std::make_shared<Tensor>(std::vector<int64_t>{1, 2, 3, 3}, DataType::kFLOAT32, GetDevice(), true);
    weight->Fill(1.0f);

    auto conv_fn = std::make_shared<autograd::Conv2d>(1, 1); // stride = 1, padding = 1
    auto result = conv_fn->Apply({input, weight});

    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{1, 1, 3, 3}));
    test::ExpectTensorNear(result[0], std::vector<float>{8, 12, 8, 12, 18, 12, 8, 12, 8}, 1e-5f);
}


TEST_P(AutogradConv2dForwardTest, Stride2) {
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 4, 4}, DataType::kFLOAT32, GetDevice(), true);
    input->Fill(1.0f);
    auto weight = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 2, 2}, DataType::kFLOAT32, GetDevice(), true);
    weight->Fill(1.0f);

    auto conv_fn = std::make_shared<autograd::Conv2d>(2, 0); // stride=2, padding = 0
    auto result = conv_fn->Apply({input, weight});

    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{1, 1, 2, 2}));
    test::ExpectTensorNear(result[0], 4.0f, 1e-5f);
}

INFINI_TRAIN_REGISTER_TEST(AutogradConv2dForwardTest);

class AutogradConv2dBackwardTest : public infini_train::test::InfiniTrainTest {};

// 反向
TEST_P(AutogradConv2dBackwardTest, WithBias) {
    auto input = MakeTensor({1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9}, GetDevice());
    auto weight = MakeTensor({1, 1, 2, 2}, {1, 2, 3, 4}, GetDevice());
    auto bias = MakeTensor({1}, {0.0f}, GetDevice());

    auto conv_fn = std::make_shared<autograd::Conv2d>(1, 0); //stride = 1, padding = 0
    auto result = conv_fn->Apply({input, weight, bias});
    EXPECT_EQ(result.size(), 1);

    auto grad_output = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 2, 2}, DataType::kFLOAT32, GetDevice(), true);
    grad_output->Fill(1.0f);
    auto grads = conv_fn->Backward({grad_output});

    ASSERT_EQ(grads.size(), 3);
    EXPECT_EQ(grads[0]->Dims(), (std::vector<int64_t>{1, 1, 3, 3}));
    test::ExpectTensorNear(grads[0], std::vector<float>{1, 3, 2, 4, 10, 6, 3, 7, 4}, 1e-5f);
    EXPECT_EQ(grads[1]->Dims(), (std::vector<int64_t>{1, 1, 2, 2}));
    test::ExpectTensorNear(grads[1], std::vector<float>{12, 16, 24, 28}, 1e-5f); 

    EXPECT_EQ(grads[2]->Dims(), (std::vector<int64_t>{1}));
    test::ExpectTensorNear(grads[2], 4.0f, 1e-5f);

}



TEST_P(AutogradConv2dBackwardTest, NoBias) {
    auto input = MakeTensor({1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9}, GetDevice());
    auto weight = MakeTensor({1, 1, 2, 2}, {1, 2, 3, 4}, GetDevice());

    auto conv_fn = std::make_shared<autograd::Conv2d>(1, 0); // stride = 1, padding = 0
    auto result = conv_fn->Apply({input, weight});
    EXPECT_EQ(result.size(), 1);

    auto grad_output = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 2, 2}, DataType::kFLOAT32, GetDevice(), true);
    grad_output->Fill(1.0f);
    auto grads = conv_fn->Backward({grad_output});

    ASSERT_EQ(grads.size(), 2);
    test::ExpectTensorNear(grads[0], std::vector<float>{1, 3, 2, 4, 10, 6, 3, 7, 4}, 1e-5f);
    test::ExpectTensorNear(grads[1], std::vector<float>{12, 16, 24, 28}, 1e-5f);
}

INFINI_TRAIN_REGISTER_TEST(AutogradConv2dBackwardTest);

