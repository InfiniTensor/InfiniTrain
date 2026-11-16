#include <stddef.h>

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/modules/activations.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

namespace infini_train {

namespace {

class NewGELUTest : public test::InfiniTrainTest {};

TEST_P(NewGELUTest, ForwardAndBackwardMatchUnfusedExpression) {
  for (const std::vector<int64_t>& shape :
       {std::vector<int64_t>{1}, {3, 257}, {1, 64, 3072}}) {
    auto host = std::make_shared<Tensor>(shape, DataType::kFLOAT32, Device());
    auto* values = static_cast<float*>(host->DataPtr());
    for (size_t i = 0; i < host->NumElements(); ++i) {
      values[i] = static_cast<float>(static_cast<int>(i % 101) - 50) / 8.0f;
    }
    auto input = std::make_shared<Tensor>(host->To(GetDevice()));
    auto reference = std::make_shared<Tensor>(host->To(GetDevice()));
    input->set_requires_grad(true);
    reference->set_requires_grad(true);
    nn::NewGELU activation;
    auto output = activation.Forward({input})[0];
    auto expected =
        0.5 * reference *
        (1.0 + nn::function::Tanh(
                   std::sqrt(2.0 / std::acos(-1.0)) *
                   (reference + 0.044715 * nn::function::Pow(reference, 3.0))));
    auto gradient_host =
        std::make_shared<Tensor>(shape, DataType::kFLOAT32, Device());
    auto* gradients = static_cast<float*>(gradient_host->DataPtr());
    for (size_t i = 0; i < gradient_host->NumElements(); ++i) {
      gradients[i] = static_cast<float>(static_cast<int>(i % 7) - 3) / 4.0f;
    }
    auto gradient = std::make_shared<Tensor>(gradient_host->To(GetDevice()));
    output->Backward(gradient);
    expected->Backward(gradient);
    ASSERT_NE(input->grad(), nullptr);
    ASSERT_NE(reference->grad(), nullptr);
    const auto actual_cpu = output->To(Device());
    const auto expected_cpu = expected->To(Device());
    const auto actual_gradient = input->grad()->To(Device());
    const auto expected_gradient = reference->grad()->To(Device());
    for (size_t i = 0; i < host->NumElements(); ++i) {
      EXPECT_NEAR(static_cast<const float*>(actual_cpu.DataPtr())[i],
                  static_cast<const float*>(expected_cpu.DataPtr())[i], 2e-6f);
      EXPECT_NEAR(static_cast<const float*>(actual_gradient.DataPtr())[i],
                  static_cast<const float*>(expected_gradient.DataPtr())[i],
                  2e-6f);
    }
  }
}

TEST_P(NewGELUTest, CudaRegistersFusedForwardAndBackward) {
  if (GetDevice().type() != Device::DeviceType::kCUDA) GTEST_SKIP();
  EXPECT_TRUE(
      Dispatcher::Instance().HasKernel({GetDevice().type(), "NewGELUForward"}));
  EXPECT_TRUE(Dispatcher::Instance().HasKernel(
      {GetDevice().type(), "NewGELUBackward"}));
}

INFINI_TRAIN_REGISTER_TEST(NewGELUTest);

}  // namespace

}  // namespace infini_train
