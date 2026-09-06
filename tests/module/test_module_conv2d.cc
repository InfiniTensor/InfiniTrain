#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/nn/modules/activations.h"
#include "infini_train/include/nn/modules/container.h"
#include "infini_train/include/nn/modules/conv.h"
#include "infini_train/include/nn/modules/flatten.h"
#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/nn/modules/loss.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class ModuleConv2dTest : public infini_train::test::InfiniTrainTest {};

TEST_P(ModuleConv2dTest, Conv2dParameterShapes) {
    const int64_t in_channels = 2;
    const int64_t out_channels = 4;
    const int64_t kernel_size = 3;
    nn::Conv2d conv(in_channels, out_channels, kernel_size);

    const auto state_dict = conv.StateDict();
    EXPECT_EQ(state_dict.size(), 2);
    ASSERT_TRUE(state_dict.contains("weight"));
    EXPECT_EQ(state_dict.at("weight")->Dims(),
              (std::vector<int64_t>{out_channels, in_channels, kernel_size, kernel_size}));
    ASSERT_TRUE(state_dict.contains("bias"));
    EXPECT_EQ(state_dict.at("bias")->Dims(), (std::vector<int64_t>{out_channels}));
    EXPECT_TRUE(state_dict.at("weight")->requires_grad());
    EXPECT_TRUE(state_dict.at("bias")->requires_grad());
}

TEST_P(ModuleConv2dTest, Conv2dNoBiasParameterShapes) {
    nn::Conv2d conv(2, 4, 3, /*stride=*/1, /*padding=*/0, /*bias=*/false);
    const auto state_dict = conv.StateDict();
    EXPECT_EQ(state_dict.size(), 1);
    EXPECT_TRUE(state_dict.contains("weight"));
}

TEST_P(ModuleConv2dTest, Conv2dForwardShape) {
    nn::Conv2d conv(2, 4, 3, /*stride=*/1, /*padding=*/1, /*bias=*/true, GetDevice());
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 2, 5, 5}, DataType::kFLOAT32, GetDevice(), false);
    input->Fill(1.0f);

    const auto output = conv.Forward({input});
    EXPECT_EQ(output.size(), 1);
    // With padding 1, the spatial size is preserved: (5 + 2 * 1 - 3) / 1 + 1 = 5.
    EXPECT_EQ(output[0]->Dims(), (std::vector<int64_t>{2, 4, 5, 5}));
}

TEST_P(ModuleConv2dTest, FlattenModule) {
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 3, 4, 5}, DataType::kFLOAT32, GetDevice(), true);
    input->Fill(1.0f);

    nn::Flatten flatten;
    const auto output = flatten.Forward({input});
    EXPECT_EQ(output.size(), 1);
    EXPECT_EQ(output[0]->Dims(), (std::vector<int64_t>{2, 60}));
}

TEST_P(ModuleConv2dTest, ConvReluFlattenLinearChain) {
    // End-to-end wiring test: conv -> relu -> flatten -> linear -> loss trains with SGD.
    const int64_t batch_size = 2;
    const int64_t num_classes = 10;

    const Device device = GetDevice();
    nn::Conv2d conv(1, 2, 3, /*stride=*/1, /*padding=*/0, /*bias=*/true, device);
    nn::Relu relu;
    nn::Flatten flatten;
    nn::Linear linear(2 * 26 * 26, num_classes, /*bias=*/true, device);
    nn::CrossEntropyLoss loss_fn;
    loss_fn.To(device);

    auto input
        = std::make_shared<Tensor>(std::vector<int64_t>{batch_size, 1, 28, 28}, DataType::kFLOAT32, device, false);
    input->Fill(0.5f);
    auto label_cpu = std::make_shared<Tensor>(std::vector<int64_t>{batch_size}, DataType::kUINT8, Device(), false);
    static_cast<uint8_t *>(label_cpu->DataPtr())[0] = 1;
    static_cast<uint8_t *>(label_cpu->DataPtr())[1] = 2;
    auto label = std::make_shared<Tensor>(label_cpu->To(device));

    auto outputs = conv.Forward({input});
    outputs = relu.Forward(outputs);
    outputs = flatten.Forward(outputs);
    outputs = linear.Forward(outputs);
    const auto loss = loss_fn.Forward({outputs[0], label});
    ASSERT_EQ(loss.size(), 1);
    loss[0]->Backward();

    std::vector<std::shared_ptr<Tensor>> params{conv.StateDict().at("weight"), conv.StateDict().at("bias"),
                                                linear.StateDict().at("weight"), linear.StateDict().at("bias")};
    for (const auto &param : params) {
        ASSERT_NE(param->grad(), nullptr);
        EXPECT_EQ(param->grad()->Dims(), param->Dims());
    }

    optimizers::SGD optimizer(params, 0.01);
    optimizer.Step();
}

INFINI_TRAIN_REGISTER_TEST(ModuleConv2dTest);
