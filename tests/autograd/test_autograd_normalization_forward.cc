#include <stdint.h>

#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autocast.h"
#include "infini_train/include/autograd/normalization.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class AutogradNormalizationForwardTest
    : public infini_train::test::InfiniTrainTest {};

TEST_P(AutogradNormalizationForwardTest, LayerNormForward) {
  auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3, 4},
                                    DataType::kFLOAT32, GetDevice(), true);
  a->Fill(1.0f);
  auto weight = std::make_shared<Tensor>(std::vector<int64_t>{4},
                                         DataType::kFLOAT32, GetDevice(), true);
  weight->Fill(1.0f);
  auto bias = std::make_shared<Tensor>(std::vector<int64_t>{4},
                                       DataType::kFLOAT32, GetDevice(), true);
  bias->Fill(0.0f);
  auto layernorm_fn = std::make_shared<autograd::LayerNorm>(1e-5f);
  auto result = layernorm_fn->Apply({a, weight, bias});
  EXPECT_EQ(result.size(), 3);
  EXPECT_FALSE(result[1]->requires_grad());
  EXPECT_EQ(result[1]->grad_fn(), nullptr);
  EXPECT_FALSE(result[2]->requires_grad());
  EXPECT_EQ(result[2]->grad_fn(), nullptr);
}

TEST_P(AutogradNormalizationForwardTest,
       LayerNormAutocastPromotesInputsToFp32) {
  auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 3, 4},
                                        DataType::kBFLOAT16, GetDevice(), true);
  input->Fill(1.0f);
  auto weight = std::make_shared<Tensor>(
      std::vector<int64_t>{4}, DataType::kBFLOAT16, GetDevice(), true);
  weight->Fill(1.0f);
  auto bias = std::make_shared<Tensor>(std::vector<int64_t>{4},
                                       DataType::kBFLOAT16, GetDevice(), true);
  bias->Fill(0.0f);

  AutocastGuard autocast_guard(GetDevice().type(), DataType::kBFLOAT16);
  auto result = std::make_shared<autograd::LayerNorm>(1e-5f)->Apply(
      {input, weight, bias});

  ASSERT_EQ(result.size(), 3);
  EXPECT_EQ(result[0]->Dtype(), DataType::kFLOAT32);
  EXPECT_EQ(result[1]->Dtype(), DataType::kFLOAT32);
  EXPECT_EQ(result[2]->Dtype(), DataType::kFLOAT32);
}

TEST_P(AutogradNormalizationForwardTest, LayerNormZeroBias) {
  auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3, 4},
                                    DataType::kFLOAT32, GetDevice(), true);
  a->Fill(1.0f);
  auto weight = std::make_shared<Tensor>(std::vector<int64_t>{4},
                                         DataType::kFLOAT32, GetDevice(), true);
  weight->Fill(1.0f);
  auto bias = std::make_shared<Tensor>(std::vector<int64_t>{4},
                                       DataType::kFLOAT32, GetDevice(), true);
  bias->Fill(0.0f);
  auto layernorm_fn = std::make_shared<autograd::LayerNorm>(1e-5f);
  auto result = layernorm_fn->Apply({a, weight, bias});
  EXPECT_EQ(result.size(), 3);
  EXPECT_FALSE(result[1]->requires_grad());
  EXPECT_EQ(result[1]->grad_fn(), nullptr);
  EXPECT_FALSE(result[2]->requires_grad());
  EXPECT_EQ(result[2]->grad_fn(), nullptr);
}

TEST_P(AutogradNormalizationForwardTest, LayerNormThreeDim) {
  auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 1, 4},
                                    DataType::kFLOAT32, GetDevice(), true);
  a->Fill(1.0f);
  auto weight = std::make_shared<Tensor>(std::vector<int64_t>{4},
                                         DataType::kFLOAT32, GetDevice(), true);
  weight->Fill(1.0f);
  auto bias = std::make_shared<Tensor>(std::vector<int64_t>{4},
                                       DataType::kFLOAT32, GetDevice(), true);
  bias->Fill(0.0f);
  auto layernorm_fn = std::make_shared<autograd::LayerNorm>(1e-5f);
  auto result = layernorm_fn->Apply({a, weight, bias});
  EXPECT_EQ(result.size(), 3);
  EXPECT_FALSE(result[1]->requires_grad());
  EXPECT_EQ(result[1]->grad_fn(), nullptr);
  EXPECT_FALSE(result[2]->requires_grad());
  EXPECT_EQ(result[2]->grad_fn(), nullptr);
  EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{2, 1, 4}));
}

INFINI_TRAIN_REGISTER_TEST(AutogradNormalizationForwardTest);
