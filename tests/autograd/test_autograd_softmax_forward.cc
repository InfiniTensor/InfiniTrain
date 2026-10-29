#include <stdint.h>

#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autograd/softmax.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class AutogradSoftmaxForwardTest : public infini_train::test::InfiniTrainTest {
};

TEST_P(AutogradSoftmaxForwardTest, SoftmaxForward) {
  auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3},
                                    DataType::kFLOAT32, GetDevice(), true);
  a->Fill(1.0f);
  auto softmax_fn = std::make_shared<autograd::Softmax>(1);
  auto result = softmax_fn->Apply({a});
  EXPECT_EQ(result.size(), 1);
  EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{2, 3}));
}

TEST_P(AutogradSoftmaxForwardTest, SoftmaxDim0) {
  auto a = std::make_shared<Tensor>(std::vector<int64_t>{4, 3},
                                    DataType::kFLOAT32, GetDevice(), true);
  a->Fill(1.0f);
  auto softmax_fn = std::make_shared<autograd::Softmax>(0);
  auto result = softmax_fn->Apply({a});
  EXPECT_EQ(result.size(), 1);
  EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{4, 3}));
}

TEST_P(AutogradSoftmaxForwardTest, SoftmaxLastDim) {
  auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3, 4},
                                    DataType::kFLOAT32, GetDevice(), true);
  a->Fill(1.0f);
  auto softmax_fn = std::make_shared<autograd::Softmax>(2);
  auto result = softmax_fn->Apply({a});
  EXPECT_EQ(result.size(), 1);
  EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{2, 3, 4}));
}

TEST_P(AutogradSoftmaxForwardTest, Bfloat16RoundsAfterNormalization) {
  if (GetDevice().type() != Device::DeviceType::kCUDA) {
    GTEST_SKIP() << "CUDA BF16 softmax precision regression";
  }
  const std::vector<float> logits = {0.0f, -1.0f};
  auto input =
      std::make_shared<Tensor>(logits.data(), std::vector<int64_t>{1, 2},
                               DataType::kFLOAT32, GetDevice());
  input = std::make_shared<Tensor>(input->To(DataType::kBFLOAT16));
  auto result = std::make_shared<autograd::Softmax>(-1)->Apply({input})[0];
  EXPECT_EQ(result->Dtype(), DataType::kBFLOAT16);
  // Correctly rounded 1 / (1 + exp(-1)) and exp(-1) / (1 + exp(-1)).
  // Rounding exp(-1) first incorrectly gives 0.267578125 for the second value.
  test::ExpectTensorFloatEqual(
      std::make_shared<Tensor>(result->To(DataType::kFLOAT32)),
      std::vector<float>{0.73046875f, 0.26953125f});
}

INFINI_TRAIN_REGISTER_TEST(AutogradSoftmaxForwardTest);
