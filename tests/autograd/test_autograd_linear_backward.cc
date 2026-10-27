#include <stdint.h>

#include <initializer_list>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autograd/linear.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class AutogradLinearBackwardTest : public infini_train::test::InfiniTrainTest {
};

TEST_P(AutogradLinearBackwardTest, LinearBackward) {
  auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 3},
                                        DataType::kFLOAT32, GetDevice(), true);
  input->Fill(1.0f);
  auto weight = std::make_shared<Tensor>(std::vector<int64_t>{4, 3},
                                         DataType::kFLOAT32, GetDevice(), true);
  weight->Fill(1.0f);
  auto bias = std::make_shared<Tensor>(std::vector<int64_t>{4},
                                       DataType::kFLOAT32, GetDevice(), true);
  bias->Fill(0.0f);
  auto linear_fn = std::make_shared<autograd::Linear>();
  auto result = linear_fn->Apply({input, weight, bias});
  const float grad_values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  auto grad = std::make_shared<Tensor>(grad_values, std::vector<int64_t>{2, 4},
                                       DataType::kFLOAT32, GetDevice());
  auto grad_inputs = linear_fn->Backward({grad});
  ASSERT_EQ(grad_inputs.size(), 3);
  ASSERT_NE(grad_inputs[2], nullptr);

  auto bias_grad_cpu = grad_inputs[2]->To(Device());
  core::GetDeviceGuardImpl(GetDevice().type())->SynchronizeDevice(GetDevice());
  const auto *bias_grad = static_cast<const float *>(bias_grad_cpu.DataPtr());
  const float expected_bias_grad[] = {6.0f, 8.0f, 10.0f, 12.0f};
  for (int idx = 0; idx < 4; ++idx) {
    EXPECT_FLOAT_EQ(bias_grad[idx], expected_bias_grad[idx]);
  }
}

TEST_P(AutogradLinearBackwardTest, LinearBackwardNoBias) {
  auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 3},
                                        DataType::kFLOAT32, GetDevice(), true);
  input->Fill(1.0f);
  auto weight = std::make_shared<Tensor>(std::vector<int64_t>{4, 3},
                                         DataType::kFLOAT32, GetDevice(), true);
  weight->Fill(1.0f);
  auto linear_fn = std::make_shared<autograd::Linear>();
  auto result = linear_fn->Apply({input, weight});
  auto grad = std::make_shared<Tensor>(std::vector<int64_t>{2, 4},
                                       DataType::kFLOAT32, GetDevice(), true);
  grad->Fill(1.0f);
  auto grad_inputs = linear_fn->Backward({grad});
  EXPECT_EQ(grad_inputs.size(), 2);
}

TEST_P(AutogradLinearBackwardTest, RawLinearSupportsBothWeightLayouts) {
  const float x[] = {1, 2, 3, 4, 5, 6};
  const float w[] = {1, 0, -1, 2, 3, 1};
  const float wt[] = {1, 2, 0, 3, -1, 1};
  const float b[] = {1, -2};
  const float dy[] = {1, 2, -1, 3};
  auto input = std::make_shared<Tensor>(x, std::vector<int64_t>{2, 3},
                                        DataType::kFLOAT32, GetDevice());
  auto bias = std::make_shared<Tensor>(b, std::vector<int64_t>{2},
                                       DataType::kFLOAT32, GetDevice());
  auto grad = std::make_shared<Tensor>(dy, std::vector<int64_t>{2, 2},
                                       DataType::kFLOAT32, GetDevice());
  for (bool transpose : {true, false}) {
    SCOPED_TRACE(transpose);
    const auto weight_dims =
        transpose ? std::vector<int64_t>{2, 3} : std::vector<int64_t>{3, 2};
    auto weight = std::make_shared<Tensor>(transpose ? w : wt, weight_dims,
                                           DataType::kFLOAT32, GetDevice());
    test::ExpectTensorFloatEqual(
        autograd::linear::Forward(input, weight, transpose, bias),
        std::vector<float>{-1, 9, -1, 27});
    test::ExpectTensorFloatEqual(
        autograd::linear::Forward(input, weight, transpose),
        std::vector<float>{-2, 11, -2, 29});
    test::ExpectTensorFloatEqual(
        autograd::linear::BackwardInput(weight, grad, transpose, 3, 2,
                                        input->Dims()),
        std::vector<float>{5, 6, 1, 5, 9, 4});
    auto grad_weight =
        autograd::linear::BackwardWeight(input, grad, transpose, 3, 2);
    EXPECT_EQ(grad_weight->Dims(), weight_dims);
    test::ExpectTensorFloatEqual(
        grad_weight, transpose ? std::vector<float>{-3, -3, -3, 14, 19, 24}
                               : std::vector<float>{-3, 14, -3, 19, -3, 24});
  }
  test::ExpectTensorFloatEqual(autograd::linear::BackwardBias(grad, 2),
                               std::vector<float>{0, 5});
}

INFINI_TRAIN_REGISTER_TEST(AutogradLinearBackwardTest);
