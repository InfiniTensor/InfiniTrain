#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/nn/modules/activations.h"
#include "infini_train/include/nn/modules/conv2d.h"
#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/nn/modules/loss.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

namespace {

std::vector<float> ToHostValues(const std::shared_ptr<Tensor> &tensor) {
    const Tensor host_tensor = tensor->To(Device());
    const auto *data = static_cast<const float *>(host_tensor.DataPtr());
    return {data, data + host_tensor.NumElements()};
}

float L1Norm(const std::shared_ptr<Tensor> &tensor) {
    const auto values = ToHostValues(tensor);
    return std::accumulate(values.begin(), values.end(), 0.0f,
                           [](float total, float value) { return total + std::abs(value); });
}

float L1Difference(const std::vector<float> &before, const std::shared_ptr<Tensor> &after) {
    const auto after_values = ToHostValues(after);
    CHECK_EQ(before.size(), after_values.size());
    float difference = 0.0f;
    for (size_t idx = 0; idx < before.size(); ++idx) { difference += std::abs(before[idx] - after_values[idx]); }
    return difference;
}

} // namespace

class AutogradCnnTrainingSmokeTest : public infini_train::test::InfiniTrainTest {};

TEST_P(AutogradCnnTrainingSmokeTest, BackwardPopulatesGradientsAndSGDUpdatesParameters) {
    constexpr int64_t kBatchSize = 4;
    constexpr int64_t kImageSize = 28;

    std::vector<float> image_values(kBatchSize * kImageSize * kImageSize);
    for (size_t idx = 0; idx < image_values.size(); ++idx) {
        image_values[idx] = static_cast<float>((idx * 17) % 256) / 255.0f;
    }
    auto images
        = std::make_shared<Tensor>(image_values.data(), std::vector<int64_t>{kBatchSize, 1, kImageSize, kImageSize},
                                   DataType::kFLOAT32, GetDevice());

    const std::vector<uint8_t> label_values{0, 1, 2, 3};
    auto host_labels = std::make_shared<Tensor>(std::vector<int64_t>{kBatchSize}, DataType::kUINT8);
    std::copy(label_values.begin(), label_values.end(), static_cast<uint8_t *>(host_labels->DataPtr()));
    auto labels = std::make_shared<Tensor>(host_labels->To(GetDevice()));

    auto conv1 = std::make_shared<nn::Conv2d>(1, 4, 3, 1, 1, true, GetDevice());
    auto relu1 = std::make_shared<nn::ReLU>();
    auto conv2 = std::make_shared<nn::Conv2d>(4, 8, 3, 2, 1, true, GetDevice());
    auto relu2 = std::make_shared<nn::ReLU>();
    auto classifier = std::make_shared<nn::Linear>(8 * 14 * 14, 10, true, GetDevice());

    auto hidden = (*conv1)({images});
    hidden = (*relu1)(hidden);
    hidden = (*conv2)(hidden);
    hidden = (*relu2)(hidden);
    const auto logits = (*classifier)({hidden[0]->Flatten(1)});

    nn::CrossEntropyLoss loss_fn;
    const auto loss = loss_fn.Forward({logits[0], labels});

    std::vector<std::shared_ptr<Tensor>> parameters = conv1->Parameters();
    const auto append_parameters = [&parameters](const std::shared_ptr<nn::Module> &module) {
        const auto module_parameters = module->Parameters();
        parameters.insert(parameters.end(), module_parameters.begin(), module_parameters.end());
    };
    append_parameters(conv2);
    append_parameters(classifier);
    ASSERT_EQ(parameters.size(), 6U);
    const auto classifier_weight_before = ToHostValues(classifier->parameter("weight"));

    loss[0]->Backward();
    for (const auto &parameter : parameters) {
        ASSERT_NE(parameter->grad(), nullptr);
        EXPECT_GT(L1Norm(parameter->grad()), 0.0f);
    }

    optimizers::SGD optimizer(parameters, 0.1f);
    optimizer.Step();
    EXPECT_GT(L1Difference(classifier_weight_before, classifier->parameter("weight")), 0.0f);
}

INFINI_TRAIN_REGISTER_TEST(AutogradCnnTrainingSmokeTest);
