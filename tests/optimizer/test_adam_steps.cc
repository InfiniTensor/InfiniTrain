#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>

#include "gtest/gtest.h"

#include "infini_train/include/checkpoint/checkpoint.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"
using namespace infini_train;
class AdamStepsTest : public test::InfiniTrainTest {
 protected:
  std::shared_ptr<Tensor> P(float v = 1) {
    auto p = std::make_shared<Tensor>(std::vector<int64_t>{1},
                                      DataType::kFLOAT32, GetDevice(), true);
    p->Fill(v);
    return p;
  }
  void G(std::shared_ptr<Tensor> p, float v) {
    auto g =
        std::make_shared<Tensor>(p->Dims(), DataType::kFLOAT32, GetDevice());
    g->Fill(v);
    p->set_grad(g);
  }
  float V(std::shared_ptr<Tensor> p) {
    auto t = p->To(Device());
    core::GetDeviceGuardImpl(GetDevice().type())
        ->SynchronizeDevice(GetDevice());
    return *static_cast<const float*>(t.DataPtr());
  }
  int64_t S(Optimizer& o, std::string name) {
    auto s = o.StateDict().at("adam.step." + name);
    return *static_cast<int64_t*>(s->DataPtr());
  }
};
TEST_P(AdamStepsTest, DelayedFirstGradient) {
  auto a = P(), b = P();
  optimizers::Adam o(std::vector{a, b}, .01f);
  G(a, .5f);
  o.Step();
  G(b, .5f);
  o.Step();
  EXPECT_NEAR(V(b), .99f, 2e-6);
  EXPECT_EQ(S(o, "0"), 2);
  EXPECT_EQ(S(o, "1"), 1);
}
TEST_P(AdamStepsTest, IntermittentGradientMatchesStandalone) {
  auto a = P(), b = P(), r = P();
  optimizers::Adam o(std::vector{a, b}, .01f), ref(std::vector{r}, .01f);
  for (int i = 0; i < 30; ++i) {
    o.ZeroGrad();
    ref.ZeroGrad();
    G(a, .1f);
    if (i % 3 == 0) {
      G(b, float(i + 1) / 30);
      G(r, float(i + 1) / 30);
      ref.Step();
    }
    o.Step();
  }
  EXPECT_NEAR(V(b), V(r), 1e-7);
  EXPECT_EQ(S(o, "1"), 10);
}
TEST_P(AdamStepsTest, AllMissingGradientsDoNotAgeParameter) {
  auto p = P();
  optimizers::Adam o(std::vector{p}, .01f);
  for (int i = 0; i < 50; ++i) o.Step();
  EXPECT_EQ(S(o, "0"), 0);
  EXPECT_FLOAT_EQ(V(p), 1);
  G(p, .5f);
  o.Step();
  EXPECT_NEAR(V(p), .99f, 2e-6);
  EXPECT_EQ(S(o, "0"), 1);
}
TEST_P(AdamStepsTest, ZeroGradientIsStillAnUpdate) {
  auto p = P();
  optimizers::Adam o(std::vector{p}, .01f);
  G(p, .5f);
  o.Step();
  o.ZeroGrad(false);
  o.Step();
  EXPECT_EQ(S(o, "0"), 2);
  EXPECT_LT(V(p), .99f);
}
TEST_P(AdamStepsTest, MissingGradientPreservesMoments) {
  auto p = P();
  optimizers::Adam o(std::vector{p}, .01f);
  G(p, .5f);
  o.Step();
  float m = V(o.StateDict().at("adam.m.0")),
        v = V(o.StateDict().at("adam.v.0"));
  o.ZeroGrad();
  o.Step();
  EXPECT_FLOAT_EQ(V(o.StateDict().at("adam.m.0")), m);
  EXPECT_FLOAT_EQ(V(o.StateDict().at("adam.v.0")), v);
  EXPECT_EQ(S(o, "0"), 1);
}
TEST_P(AdamStepsTest, NamedReorderAndResume) {
  auto a = P(), b = P(), c = P(), d = P();
  optimizers::Adam src(NamedParameterList{{"A", a}, {"B", b}}, .01f),
      dst(NamedParameterList{{"B", d}, {"A", c}}, .01f);
  G(a, .3f);
  src.Step();
  G(b, .2f);
  src.Step();
  c->CopyFrom(a);
  d->CopyFrom(b);
  dst.LoadStateDict(src.StateDict());
  EXPECT_EQ(S(dst, "A"), 2);
  EXPECT_EQ(S(dst, "B"), 1);
  for (int i = 0; i < 10; ++i) {
    src.ZeroGrad();
    dst.ZeroGrad();
    G(b, -.2f);
    G(d, -.2f);
    if (i % 2) {
      G(a, .1f);
      G(c, .1f);
    }
    src.Step();
    dst.Step();
  }
  EXPECT_FLOAT_EQ(V(a), V(c));
  EXPECT_FLOAT_EQ(V(b), V(d));
  EXPECT_EQ(S(src, "A"), S(dst, "A"));
  EXPECT_EQ(S(src, "B"), S(dst, "B"));
}
TEST_P(AdamStepsTest, LegacyStateAllParametersActive) {
  auto a = P(), b = P();
  optimizers::Adam src(std::vector{a}, .01f), dst(std::vector{b}, .01f);
  G(a, .5f);
  for (int i = 0; i < 5; ++i) src.Step();
  auto legacy = src.StateDict();
  legacy.erase("adam.step.0");
  b->CopyFrom(a);
  dst.LoadStateDict(legacy);
  G(b, .5f);
  src.Step();
  dst.Step();
  EXPECT_FLOAT_EQ(V(a), V(b));
  EXPECT_EQ(S(dst, "0"), 6);
}
TEST_P(AdamStepsTest, EmptyOptimizer) {
  optimizers::Adam o(std::vector<std::shared_ptr<Tensor>>{}, .01f);
  o.Step();
  auto s = o.StateDict();
  o.LoadStateDict(s);
  EXPECT_EQ(s.size(), 1);
}
TEST_P(AdamStepsTest, DiskRoundTripUnequalSteps) {
  auto src = std::make_shared<nn::Linear>(1, 1, true, GetDevice());
  auto dst = std::make_shared<nn::Linear>(1, 1, true, GetDevice());
  for (auto p : src->Parameters()) p->Fill(1.f);
  optimizers::Adam a(src->NamedParameters(), .01f),
      b(dst->NamedParameters(), .01f);
  auto w = src->parameter("weight"), bias = src->parameter("bias");
  G(w, .3f);
  a.Step();
  G(bias, .5f);
  a.Step();
  a.ZeroGrad();
  const char* artifact_root = std::getenv("INFINITRAIN_TEST_ARTIFACT_DIR");
  auto root = artifact_root ? std::filesystem::path(artifact_root)
                            : std::filesystem::temp_directory_path();
  auto dir = root / ("infinitrain_adam_checkpoint_" + std::to_string(getpid()) +
                     "_" + std::to_string(int(GetDevice().type())));
  ASSERT_FALSE(std::filesystem::exists(dir));
  TrainerState st{.global_step = 2};
  core::GetDeviceGuardImpl(GetDevice().type())->SynchronizeDevice(GetDevice());
  Checkpoint::Save(dir, *src, &a, st, nullptr);
  TrainerState loaded;
  Checkpoint::Load(dir, *dst, &b, loaded, nullptr);
  EXPECT_EQ(loaded.global_step, 2);
  EXPECT_EQ(S(b, "weight"), 2);
  EXPECT_EQ(S(b, "bias"), 1);
  for (int i = 0; i < 10; ++i) {
    a.ZeroGrad();
    b.ZeroGrad();
    G(bias, .1f);
    G(dst->parameter("bias"), .1f);
    if (i % 2) {
      G(w, -.2f);
      G(dst->parameter("weight"), -.2f);
    }
    a.Step();
    b.Step();
  }
  EXPECT_FLOAT_EQ(V(w), V(dst->parameter("weight")));
  EXPECT_FLOAT_EQ(V(bias), V(dst->parameter("bias")));
}
TEST_P(AdamStepsTest, RejectsIncompleteParameterSteps) {
  if (GetDevice().type() != Device::DeviceType::kCPU)
    GTEST_SKIP() << "Death test only on CPU";
  auto a = P(), b = P();
  optimizers::Adam o(std::vector{a, b});
  auto st = o.StateDict();
  st.erase("adam.step.1");
  EXPECT_DEATH(o.LoadStateDict(st), "Missing optimizer state");
}
TEST_P(AdamStepsTest, RejectsNegativeStep) {
  if (GetDevice().type() != Device::DeviceType::kCPU)
    GTEST_SKIP() << "Death test only on CPU";
  auto p = P();
  optimizers::Adam o(std::vector{p});
  auto st = o.StateDict();
  *static_cast<int64_t*>(st.at("adam.step.0")->DataPtr()) = -1;
  EXPECT_DEATH(o.LoadStateDict(st), "Negative Adam step");
}
INFINI_TRAIN_REGISTER_TEST(AdamStepsTest);
