#include <stdint.h>

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autograd/linear.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class AutogradLinearForwardTest : public infini_train::test::InfiniTrainTest {};

TEST_P(AutogradLinearForwardTest, LinearForward) {
  auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 3},
                                        DataType::kFLOAT32, GetDevice(), true);
  input->Fill(1.0f);
  auto weight = std::make_shared<Tensor>(std::vector<int64_t>{4, 3},
                                         DataType::kFLOAT32, GetDevice(), true);
  weight->Fill(1.0f);
  auto bias = std::make_shared<Tensor>(std::vector<int64_t>{4},
                                       DataType::kFLOAT32, GetDevice(), true);
  bias->Fill(2.0f);
  auto linear_fn = std::make_shared<autograd::Linear>();
  auto result = linear_fn->Apply({input, weight, bias});
  EXPECT_EQ(result.size(), 1);
  EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{2, 4}));
  test::ExpectTensorFloatEqual(result[0], 5.0f);
}

TEST_P(AutogradLinearForwardTest, LinearNoBias) {
  auto input = std::make_shared<Tensor>(std::vector<int64_t>{1, 3},
                                        DataType::kFLOAT32, GetDevice(), true);
  input->Fill(1.0f);
  auto weight = std::make_shared<Tensor>(std::vector<int64_t>{4, 3},
                                         DataType::kFLOAT32, GetDevice(), true);
  weight->Fill(1.0f);
  auto linear_fn = std::make_shared<autograd::Linear>();
  auto result = linear_fn->Apply({input, weight});
  EXPECT_EQ(result.size(), 1);
  EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{1, 4}));
  test::ExpectTensorFloatEqual(result[0], 3.0f);
}

TEST_P(AutogradLinearForwardTest, LinearBatch) {
  auto input = std::make_shared<Tensor>(std::vector<int64_t>{32, 128},
                                        DataType::kFLOAT32, GetDevice(), true);
  input->Fill(1.0f);
  auto weight = std::make_shared<Tensor>(std::vector<int64_t>{64, 128},
                                         DataType::kFLOAT32, GetDevice(), true);
  weight->Fill(1.0f);
  auto linear_fn = std::make_shared<autograd::Linear>();
  auto result = linear_fn->Apply({input, weight});
  EXPECT_EQ(result.size(), 1);
  EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{32, 64}));
  test::ExpectTensorFloatEqual(result[0], 128.0f);
}

TEST_P(AutogradLinearForwardTest, BFloat16ReductionPreservesCancellation) {
  if (GetDevice().type() != Device::DeviceType::kCUDA) {
    GTEST_SKIP() << "cuBLAS reduction regression";
  }
  constexpr int64_t kInputFeatures = 4096;
  constexpr int64_t kOutputFeatures = 1536;
  constexpr float kResidual = 1.0f / 1024.0f;
  std::vector<float> weights(kOutputFeatures * kInputFeatures, 0.0f);
  for (int64_t i = 0; i < kOutputFeatures; ++i) {
    weights[i * kInputFeatures] = 1.0f;
    weights[i * kInputFeatures + 1] = kResidual;
    weights[(i + 1) * kInputFeatures - 1] = -1.0f;
  }
  auto weight = std::make_shared<Tensor>(
      weights.data(), std::vector<int64_t>{kOutputFeatures, kInputFeatures},
      DataType::kFLOAT32, GetDevice());
  weight = std::make_shared<Tensor>(weight->To(DataType::kBFLOAT16));
  // Both operands and the exact dot product are representable in BF16. A
  // BF16 partial reduction can round away the residual before cancellation.
  for (const int64_t rows : {64, 32}) {
    auto input =
        std::make_shared<Tensor>(std::vector<int64_t>{rows, kInputFeatures},
                                 DataType::kBFLOAT16, GetDevice());
    input->Fill(1.0f);
    auto result =
        std::make_shared<autograd::Linear>()->Apply({input, weight})[0];
    ASSERT_EQ(result->Dtype(), DataType::kBFLOAT16);
    auto host_result = result->To(DataType::kFLOAT32).To(Device());
    const auto *values = static_cast<const float *>(host_result.DataPtr());
    float max_error = 0.0f;
    for (int64_t i = 0; i < rows * kOutputFeatures; ++i) {
      max_error = std::max(max_error, std::abs(values[i] - kResidual));
    }
    EXPECT_EQ(max_error, 0.0f) << "rows=" << rows;
  }
}

TEST_P(AutogradLinearForwardTest, Float32MicrobatchAccumulationAccuracy) {
  if (GetDevice().type() != Device::DeviceType::kCUDA) {
    GTEST_SKIP() << "cuBLAS accumulation regression";
  }
  constexpr int64_t kRows = 512;
  constexpr int64_t kInputFeatures = 8192;
  constexpr int64_t kOutputFeatures = 2048;
  // Full-mantissa inputs exercise accumulation rounding, unlike small integer
  // inputs whose sums are exactly representable.
  uint32_t state = 19;
  auto sample = [&]() {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return (static_cast<float>(state >> 8) / 16777216.0f - 0.5f) * 0.1f;
  };
  std::vector<float> inputs(kRows * kInputFeatures);
  std::vector<float> weights(kOutputFeatures * kInputFeatures);
  std::generate(inputs.begin(), inputs.end(), sample);
  std::generate(weights.begin(), weights.end(), sample);
  auto weight = std::make_shared<Tensor>(
      weights.data(), std::vector<int64_t>{kOutputFeatures, kInputFeatures},
      DataType::kFLOAT32, GetDevice());
  std::vector<Tensor> results;
  for (const int64_t rows : {kRows, kRows / 2}) {
    auto input = std::make_shared<Tensor>(
        inputs.data(), std::vector<int64_t>{rows, kInputFeatures},
        DataType::kFLOAT32, GetDevice());
    results.push_back(
        std::make_shared<autograd::Linear>()->Apply({input, weight})[0]->To(
            Device()));
  }
  const auto *full = static_cast<const float *>(results[0].DataPtr());
  const auto *half = static_cast<const float *>(results[1].DataPtr());
  double error = 0.0;
  double norm = 0.0;
  for (int64_t i = 0; i < kRows / 2 * kOutputFeatures; ++i) {
    const double delta = static_cast<double>(full[i]) - half[i];
    error += delta * delta;
    norm += static_cast<double>(full[i]) * full[i];
  }
  EXPECT_LT(std::sqrt(error / norm), 1e-7);
  // Also check accuracy against an independent FP64 dot product; merely
  // choosing the same inaccurate algorithm for both shapes is insufficient.
  double reference_error = 0.0;
  double reference_norm = 0.0;
  for (int64_t row = 0; row < 4; ++row) {
    for (int64_t col = 0; col < 32; ++col) {
      double reference = 0.0;
      for (int64_t inner = 0; inner < kInputFeatures; ++inner) {
        reference += static_cast<double>(inputs[row * kInputFeatures + inner]) *
                     weights[col * kInputFeatures + inner];
      }
      const double delta = full[row * kOutputFeatures + col] - reference;
      reference_error += delta * delta;
      reference_norm += reference * reference;
    }
  }
  EXPECT_LT(std::sqrt(reference_error / reference_norm), 2e-6);
}

INFINI_TRAIN_REGISTER_TEST(AutogradLinearForwardTest);
