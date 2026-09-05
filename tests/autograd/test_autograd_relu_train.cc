#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/nn/modules/activations.h"
#include "infini_train/include/nn/modules/conv.h"
#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/nn/modules/loss.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

// Minimal end-to-end training step with ReLU in the middle:
// Conv2d -> ReLU -> Flatten -> Linear -> CrossEntropy, one loss->Backward() through the
// autograd graph and one SGD step, on CPU and CUDA.
class AutogradReLUTrainTest : public infini_train::test::InfiniTrainTest {};

TEST_P(AutogradReLUTrainTest, ReluConvFlattenLinearCrossEntropySgdStep) {
    const Device device = GetDevice();
    const Device host = Device();
    constexpr float kLearningRate = 0.1f;

    std::vector<float> input_values;
    for (int idx = 0; idx < 2 * 1 * 8 * 8; ++idx) { input_values.push_back((idx * 37) % 101 / 50.0f - 1.0f); }
    auto input
        = std::make_shared<Tensor>(input_values.data(), std::vector<int64_t>{2, 1, 8, 8}, DataType::kFLOAT32, device);
    auto target = std::make_shared<Tensor>(std::vector<int64_t>{2}, DataType::kINT64, device);
    target->Fill(0);

    auto conv = std::make_shared<nn::Conv2d>(1, 2, 3, true, device);
    auto relu = std::make_shared<nn::ReLU>();
    auto fc = std::make_shared<nn::Linear>(2 * 6 * 6, 3, true, device);
    auto loss_fn = std::make_shared<nn::CrossEntropyLoss>();

    auto conv_out = (*conv)({input})[0];
    ASSERT_EQ(conv_out->Dims(), (std::vector<int64_t>{2, 2, 6, 6}));
    auto relu_out = (*relu)({conv_out})[0];
    ASSERT_EQ(relu_out->Dims(), (std::vector<int64_t>{2, 2, 6, 6}));

    // The pre-activations must contain negatives so the ReLU mask actually fires.
    bool saw_negative = false;
    bool saw_masked_zero = false;
    const auto conv_cpu = conv_out->To(host);
    const auto relu_cpu = relu_out->To(host);
    const float *conv_data = static_cast<const float *>(conv_cpu.DataPtr());
    const float *relu_data = static_cast<const float *>(relu_cpu.DataPtr());
    for (size_t idx = 0; idx < conv_cpu.NumElements(); ++idx) {
        ASSERT_TRUE(std::isfinite(relu_data[idx]));
        if (conv_data[idx] < 0.0f) {
            saw_negative = true;
        }
        if (relu_data[idx] == 0.0f) {
            saw_masked_zero = true;
        }
    }
    ASSERT_TRUE(saw_negative);
    ASSERT_TRUE(saw_masked_zero);

    auto flat = relu_out->Flatten(1);
    ASSERT_EQ(flat->Dims(), (std::vector<int64_t>{2, 72}));
    auto logits = (*fc)({flat})[0];
    ASSERT_EQ(logits->Dims(), (std::vector<int64_t>{2, 3}));
    auto loss = (*loss_fn)({logits, target})[0];
    ASSERT_TRUE(loss->Dims().empty());
    const auto loss_cpu = loss->To(host);
    EXPECT_TRUE(std::isfinite(*static_cast<const float *>(loss_cpu.DataPtr())));

    loss->Backward();

    std::vector<std::shared_ptr<Tensor>> params = conv->Parameters();
    for (auto &param : fc->Parameters()) { params.push_back(param); }
    ASSERT_EQ(params.size(), 4);

    // Every parameter must have received a finite, non-trivial gradient through the graph.
    for (const auto &param : params) {
        const auto &grad = param->grad();
        ASSERT_NE(grad, nullptr);
        ASSERT_EQ(grad->Dims(), param->Dims());
        const auto grad_cpu = grad->To(host);
        const auto *grad_data = static_cast<const float *>(grad_cpu.DataPtr());
        float max_abs = 0.0f;
        for (size_t idx = 0; idx < grad_cpu.NumElements(); ++idx) {
            ASSERT_TRUE(std::isfinite(grad_data[idx]));
            max_abs = std::max(max_abs, std::fabs(grad_data[idx]));
        }
        EXPECT_GT(max_abs, 0.0f);
    }

    std::vector<std::shared_ptr<Tensor>> old_values_cpu;
    for (const auto &param : params) {
        // CopyFrom (not To) so the snapshot owns its buffer: on CPU To(home device)
        // returns a view, which would alias the in-place update below.
        auto old_value = std::make_shared<Tensor>(param->Dims(), param->Dtype(), host);
        old_value->CopyFrom(*param);
        old_values_cpu.push_back(old_value);
    }

    auto optimizer = std::make_shared<optimizers::SGD>(params, kLearningRate);
    optimizer->Step();

    for (size_t p = 0; p < params.size(); ++p) {
        const auto new_cpu = params[p]->To(host);
        const auto grad_cpu = params[p]->grad()->To(host);
        const auto *old_data = static_cast<const float *>(old_values_cpu[p]->DataPtr());
        const auto *grad_data = static_cast<const float *>(grad_cpu.DataPtr());
        const auto *new_data = static_cast<const float *>(new_cpu.DataPtr());
        for (size_t idx = 0; idx < params[p]->NumElements(); ++idx) {
            EXPECT_FLOAT_EQ(new_data[idx], old_data[idx] - kLearningRate * grad_data[idx]) << "param " << p;
        }
    }
}

INFINI_TRAIN_REGISTER_TEST(AutogradReLUTrainTest);
