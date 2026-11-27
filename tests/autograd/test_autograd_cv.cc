#include <stdint.h>

#include <cmath>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/modules/activations.h"
#include "infini_train/include/nn/modules/container.h"
#include "infini_train/include/nn/modules/conv.h"
#include "infini_train/include/nn/modules/normalization.h"
#include "infini_train/include/nn/modules/pooling.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

namespace {

std::shared_ptr<Tensor> MakeTensor(const std::vector<float> &data,
                                   const std::vector<int64_t> &dims,
                                   Device device, bool requires_grad = false) {
  auto cpu_t = std::make_shared<Tensor>(data.data(), dims, DataType::kFLOAT32);
  auto dev_t = std::make_shared<Tensor>(cpu_t->To(device));
  if (requires_grad) {
    dev_t->set_requires_grad(true);
  }
  return dev_t;
}

std::vector<float> ToVector(const std::shared_ptr<Tensor> &t) {
  auto cpu_t = t->To(Device());
  const float *ptr = static_cast<const float *>(cpu_t.DataPtr());
  return std::vector<float>(ptr, ptr + cpu_t.NumElements());
}

}  // namespace

class AutogradCVTest : public infini_train::test::InfiniTrainTest {};

TEST_P(AutogradCVTest, ReLUAndReLU6ForwardBackward) {
  auto x =
      MakeTensor({-2.0f, 0.0f, 0.5f, 6.0f, 8.5f}, {1, 5}, GetDevice(), true);
  auto relu = nn::ReLU();
  auto y = relu({x})[0];
  infini_train::test::ExpectTensorFloatEqual(y, {0.0f, 0.0f, 0.5f, 6.0f, 8.5f});

  auto go = MakeTensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, {1, 5}, GetDevice());
  y->Backward(go);
  infini_train::test::ExpectTensorFloatEqual(x->grad(),
                                             {0.0f, 0.0f, 3.0f, 4.0f, 5.0f});

  auto x6 =
      MakeTensor({-2.0f, 0.0f, 0.5f, 6.0f, 8.5f}, {1, 5}, GetDevice(), true);
  auto relu6 = nn::ReLU6();
  auto y6 = relu6({x6})[0];
  infini_train::test::ExpectTensorFloatEqual(y6,
                                             {0.0f, 0.0f, 0.5f, 6.0f, 6.0f});
  y6->Backward(go);
  infini_train::test::ExpectTensorFloatEqual(x6->grad(),
                                             {0.0f, 0.0f, 3.0f, 0.0f, 0.0f});
}

TEST_P(AutogradCVTest, MaxPool2dForwardBackward) {
  // Input [1, 1, 4, 4]
  std::vector<float> in_data = {
      1,  2,  3,  4,   //
      5,  6,  7,  8,   //
      9,  10, 11, 12,  //
      13, 14, 15, 16,
  };
  auto x = MakeTensor(in_data, {1, 1, 4, 4}, GetDevice(), true);
  nn::MaxPool2d pool(2, 2, 0);
  auto y = pool({x})[0];
  ASSERT_EQ(y->Dims(), (std::vector<int64_t>{1, 1, 2, 2}));
  infini_train::test::ExpectTensorFloatEqual(y, {6.0f, 8.0f, 14.0f, 16.0f});

  auto go = MakeTensor({1.0f, 2.0f, 3.0f, 4.0f}, {1, 1, 2, 2}, GetDevice());
  y->Backward(go);
  std::vector<float> expected_grad = {
      0, 0, 0, 0,  //
      0, 1, 0, 2,  //
      0, 0, 0, 0,  //
      0, 3, 0, 4,
  };
  infini_train::test::ExpectTensorFloatEqual(x->grad(), expected_grad);
}

TEST_P(AutogradCVTest, AdaptiveAvgPool2dAndFlatten) {
  std::vector<float> in_data = {
      1, 2, 3, 4,  // c=0
      5, 6, 7, 8,  // c=1
  };
  auto x = MakeTensor(in_data, {1, 2, 2, 2}, GetDevice(), true);
  nn::AdaptiveAvgPool2d gap(1);
  nn::Flatten flatten(1);
  auto pooled = gap({x})[0];
  ASSERT_EQ(pooled->Dims(), (std::vector<int64_t>{1, 2, 1, 1}));
  auto flat = flatten({pooled})[0];
  ASSERT_EQ(flat->Dims(), (std::vector<int64_t>{1, 2}));
  infini_train::test::ExpectTensorNear(flat, {2.5f, 6.5f}, 1e-5f);

  auto go = MakeTensor({4.0f, 8.0f}, {1, 2}, GetDevice());
  flat->Backward(go);
  infini_train::test::ExpectTensorNear(
      x->grad(), {1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f}, 1e-5f);
}

TEST_P(AutogradCVTest, Conv2dAllPathsMatchCPUReference) {
  struct Case {
    int64_t N, C_in, H, W, C_out, kH, kW, sH, sW, pH, pW, groups;
    bool bias;
  };
  std::vector<Case> cases = {
      // Standard 3x3 stride 1 pad 1
      {2, 3, 8, 8, 4, 3, 3, 1, 1, 1, 1, 1, true},
      // Strided 3x3 stride 2 pad 1, no bias
      {2, 4, 8, 8, 6, 3, 3, 2, 2, 1, 1, 1, false},
      // Pointwise 1x1 fast path
      {3, 4, 6, 6, 8, 1, 1, 1, 1, 0, 0, 1, true},
      // Depthwise 3x3 stride 1 & stride 2
      {2, 6, 8, 8, 6, 3, 3, 1, 1, 1, 1, 6, false},
      {2, 6, 8, 8, 6, 3, 3, 2, 2, 1, 1, 6, true},
      // Grouped conv (groups=2)
      {2, 4, 6, 6, 6, 3, 3, 1, 1, 1, 1, 2, true},
  };

  for (const auto &tc : cases) {
    std::vector<float> x_vals(tc.N * tc.C_in * tc.H * tc.W);
    for (size_t i = 0; i < x_vals.size(); ++i) {
      x_vals[i] = std::sin(static_cast<float>(i) * 0.13f);
    }
    std::vector<float> w_vals(tc.C_out * (tc.C_in / tc.groups) * tc.kH * tc.kW);
    for (size_t i = 0; i < w_vals.size(); ++i) {
      w_vals[i] = std::cos(static_cast<float>(i) * 0.17f) * 0.25f;
    }
    std::vector<float> b_vals(tc.C_out);
    for (size_t i = 0; i < b_vals.size(); ++i) {
      b_vals[i] = 0.05f * static_cast<float>(i + 1);
    }

    // CPU reference
    auto conv_cpu = std::make_shared<nn::Conv2d>(
        tc.C_in, tc.C_out, std::pair<int64_t, int64_t>{tc.kH, tc.kW},
        std::pair<int64_t, int64_t>{tc.sH, tc.sW},
        std::pair<int64_t, int64_t>{tc.pH, tc.pW},
        std::pair<int64_t, int64_t>{1, 1}, tc.groups, tc.bias, Device());
    (*conv_cpu->mutable_parameter("weight"))
        ->CopyFrom(MakeTensor(w_vals,
                              (*conv_cpu->mutable_parameter("weight"))->Dims(),
                              Device()));
    if (tc.bias) {
      (*conv_cpu->mutable_parameter("bias"))
          ->CopyFrom(MakeTensor(b_vals, {tc.C_out}, Device()));
    }
    auto x_cpu =
        MakeTensor(x_vals, {tc.N, tc.C_in, tc.H, tc.W}, Device(), true);
    auto y_cpu = (*conv_cpu)({x_cpu})[0];

    std::vector<float> go_vals(y_cpu->NumElements());
    for (size_t i = 0; i < go_vals.size(); ++i) {
      go_vals[i] = std::sin(static_cast<float>(i) * 0.23f + 0.5f);
    }
    auto go_cpu = MakeTensor(go_vals, y_cpu->Dims(), Device());
    y_cpu->Backward(go_cpu);

    // Target device
    auto conv_dev = std::make_shared<nn::Conv2d>(
        tc.C_in, tc.C_out, std::pair<int64_t, int64_t>{tc.kH, tc.kW},
        std::pair<int64_t, int64_t>{tc.sH, tc.sW},
        std::pair<int64_t, int64_t>{tc.pH, tc.pW},
        std::pair<int64_t, int64_t>{1, 1}, tc.groups, tc.bias, GetDevice());
    (*conv_dev->mutable_parameter("weight"))
        ->CopyFrom(MakeTensor(w_vals,
                              (*conv_dev->mutable_parameter("weight"))->Dims(),
                              GetDevice()));
    if (tc.bias) {
      (*conv_dev->mutable_parameter("bias"))
          ->CopyFrom(MakeTensor(b_vals, {tc.C_out}, GetDevice()));
    }
    auto x_dev =
        MakeTensor(x_vals, {tc.N, tc.C_in, tc.H, tc.W}, GetDevice(), true);
    auto y_dev = (*conv_dev)({x_dev})[0];
    auto go_dev = MakeTensor(go_vals, y_dev->Dims(), GetDevice());
    y_dev->Backward(go_dev);

    infini_train::test::ExpectTensorNear(y_dev, ToVector(y_cpu), 1e-4f);
    infini_train::test::ExpectTensorNear(x_dev->grad(), ToVector(x_cpu->grad()),
                                         1e-4f);
    infini_train::test::ExpectTensorNear(
        conv_dev->parameter("weight")->grad(),
        ToVector(conv_cpu->parameter("weight")->grad()), 1e-4f);
    if (tc.bias) {
      infini_train::test::ExpectTensorNear(
          conv_dev->parameter("bias")->grad(),
          ToVector(conv_cpu->parameter("bias")->grad()), 1e-4f);
    }
  }
}

TEST_P(AutogradCVTest, BatchNorm2dTrainEvalAndBackward) {
  std::vector<float> x_vals(2 * 3 * 4 * 4);
  for (size_t i = 0; i < x_vals.size(); ++i) {
    x_vals[i] = std::sin(static_cast<float>(i) * 0.31f) * 2.0f + 0.5f;
  }
  std::vector<float> go_vals(x_vals.size());
  for (size_t i = 0; i < go_vals.size(); ++i) {
    go_vals[i] = std::cos(static_cast<float>(i) * 0.19f);
  }

  auto bn_cpu = std::make_shared<nn::BatchNorm2d>(3, 1e-5f, 0.1f, Device());
  auto bn_dev = std::make_shared<nn::BatchNorm2d>(3, 1e-5f, 0.1f, GetDevice());

  // Parameters vs Buffers check
  EXPECT_EQ(bn_dev->Parameters().size(), 2);
  EXPECT_EQ(bn_dev->Buffers().size(), 2);

  // Train mode
  bn_cpu->Train();
  bn_dev->Train();
  EXPECT_TRUE(bn_dev->is_training());

  auto x_cpu = MakeTensor(x_vals, {2, 3, 4, 4}, Device(), true);
  auto y_cpu = (*bn_cpu)({x_cpu})[0];
  y_cpu->Backward(MakeTensor(go_vals, {2, 3, 4, 4}, Device()));

  auto x_dev = MakeTensor(x_vals, {2, 3, 4, 4}, GetDevice(), true);
  auto y_dev = (*bn_dev)({x_dev})[0];
  y_dev->Backward(MakeTensor(go_vals, {2, 3, 4, 4}, GetDevice()));

  infini_train::test::ExpectTensorNear(y_dev, ToVector(y_cpu), 1e-4f);
  infini_train::test::ExpectTensorNear(x_dev->grad(), ToVector(x_cpu->grad()),
                                       1e-4f);
  infini_train::test::ExpectTensorNear(
      bn_dev->parameter("weight")->grad(),
      ToVector(bn_cpu->parameter("weight")->grad()), 1e-4f);
  infini_train::test::ExpectTensorNear(
      bn_dev->parameter("bias")->grad(),
      ToVector(bn_cpu->parameter("bias")->grad()), 1e-4f);
  infini_train::test::ExpectTensorNear(bn_dev->buffer("running_mean"),
                                       ToVector(bn_cpu->buffer("running_mean")),
                                       1e-5f);
  infini_train::test::ExpectTensorNear(bn_dev->buffer("running_var"),
                                       ToVector(bn_cpu->buffer("running_var")),
                                       1e-5f);

  // Eval mode must not mutate running stats
  auto rm_before = ToVector(bn_dev->buffer("running_mean"));
  auto rv_before = ToVector(bn_dev->buffer("running_var"));
  bn_cpu->Eval();
  bn_dev->Eval();
  EXPECT_FALSE(bn_dev->is_training());

  auto y_eval_cpu = (*bn_cpu)({MakeTensor(x_vals, {2, 3, 4, 4}, Device())})[0];
  auto y_eval_dev =
      (*bn_dev)({MakeTensor(x_vals, {2, 3, 4, 4}, GetDevice())})[0];
  infini_train::test::ExpectTensorNear(y_eval_dev, ToVector(y_eval_cpu), 1e-4f);
  infini_train::test::ExpectTensorFloatEqual(bn_dev->buffer("running_mean"),
                                             rm_before);
  infini_train::test::ExpectTensorFloatEqual(bn_dev->buffer("running_var"),
                                             rv_before);
}

TEST_P(AutogradCVTest, SGDMomentumWeightDecayAndStateDict) {
  auto p = MakeTensor({1.0f, -2.0f, 3.0f}, {3}, GetDevice(), true);
  optimizers::SGD opt({p}, /*learning_rate=*/0.1f, /*momentum=*/0.9f,
                      /*weight_decay=*/0.01f);

  // Step 1: d_p = g + wd * p; buf = d_p; p = p - lr * buf
  auto g1 = MakeTensor({0.5f, -0.5f, 1.0f}, {3}, GetDevice());
  p->set_grad(g1);
  opt.Step();
  // p0: d_p = 0.5 + 0.01*1 = 0.51 -> p = 1.0 - 0.051 = 0.949
  // p1: d_p = -0.5 + 0.01*(-2) = -0.52 -> p = -2.0 + 0.052 = -1.948
  // p2: d_p = 1.0 + 0.01*3 = 1.03 -> p = 3.0 - 0.103 = 2.897
  infini_train::test::ExpectTensorNear(p, {0.949f, -1.948f, 2.897f}, 1e-5f);

  // Save state dict, clone param+opt, resume and compare Step 2
  auto sd = opt.StateDict();
  auto p_resume = MakeTensor(ToVector(p), {3}, GetDevice(), true);
  optimizers::SGD opt_resume({p_resume}, 0.1f, 0.9f, 0.01f);
  opt_resume.LoadStateDict(sd);

  auto g2 = MakeTensor({-0.2f, 0.4f, 0.1f}, {3}, GetDevice());
  p->set_grad(g2);
  p_resume->set_grad(MakeTensor({-0.2f, 0.4f, 0.1f}, {3}, GetDevice()));
  opt.Step();
  opt_resume.Step();
  infini_train::test::ExpectTensorFloatEqual(p_resume, ToVector(p));
}

INFINI_TRAIN_REGISTER_TEST(AutogradCVTest);
