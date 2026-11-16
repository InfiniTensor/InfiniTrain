#include <chrono>
#include <filesystem>

#include "gtest/gtest.h"

#include "infini_train/include/checkpoint/checkpoint.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/optimizer.h"

#include "tests/common/test_utils.h"

using namespace infini_train;
namespace {
using T = std::shared_ptr<Tensor>;
class Holder : public nn::Module {
 public:
  Holder(T p, T b) {
    parameters_["weight"] = p;
    buffers_["buffer"] = b;
  }

  T buffer() const { return buffers_.at("buffer"); }
  void AddNulls() {
    parameters_["optional"] = nullptr;
    buffers_["optional"] = nullptr;
    modules_["optional"] = nullptr;
  }
};
class SharedModel : public nn::Module {
 public:
  std::shared_ptr<Holder> left, right;
  explicit SharedModel(bool shared_child = false) {
    auto p = std::make_shared<Tensor>(std::vector<int64_t>{1},
                                      DataType::kFLOAT32, Device());
    p->Fill(1);
    p->set_requires_grad(true);
    auto b = std::make_shared<Tensor>(std::vector<int64_t>{1},
                                      DataType::kFLOAT32, Device());
    b->Fill(2);
    left = std::make_shared<Holder>(p, b);
    right = shared_child ? left : std::make_shared<Holder>(p, b);
    modules_["left"] = left;
    modules_["right"] = right;
  }

  T Weight() { return left->parameter("weight"); }
};
void Sync(Device d) {
  core::GetDeviceGuardImpl(d.type())->SynchronizeDevice(d);
}
float Value(T p) {
  auto f = std::make_shared<Tensor>(p->To(DataType::kFLOAT32));
  auto cpu = f->To(Device());
  Sync(p->GetDevice());
  return *static_cast<const float*>(cpu.DataPtr());
}
void CheckAlias(std::shared_ptr<SharedModel> m) {
  EXPECT_EQ(m->left->parameter("weight").get(),
            m->right->parameter("weight").get());
  EXPECT_EQ(m->left->buffer().get(), m->right->buffer().get());
  EXPECT_EQ(m->Parameters().size(), 1);
  EXPECT_EQ(m->NamedParameters("", true, false).size(), 2);
}
void SetGrad(T p, float value) {
  auto g = std::make_shared<Tensor>(p->Dims(), p->Dtype(), p->GetDevice());
  g->Fill(value);
  p->set_grad(g);
}
}  // namespace

class SharedMigrationTest : public test::InfiniTrainTest {};
TEST_P(SharedMigrationTest, DeviceTransferPreservesAliases) {
  auto m = std::make_shared<SharedModel>();
  m->To(GetDevice());
  Sync(GetDevice());
  CheckAlias(m);
  EXPECT_EQ(m->Weight()->GetDevice(), GetDevice());
  EXPECT_TRUE(m->Weight()->requires_grad());
  EXPECT_FLOAT_EQ(Value(m->Weight()), 1);
  EXPECT_FLOAT_EQ(Value(m->left->buffer()), 2);
  m->To(Device());
  Sync(GetDevice());
  CheckAlias(m);
  EXPECT_EQ(m->Weight()->GetDevice(), Device());
}
TEST_P(SharedMigrationTest, DtypeNoOpPreservesTensorIdentity) {
  auto m = std::make_shared<SharedModel>();
  m->To(GetDevice());
  auto p = m->Weight();
  m->To(DataType::kFLOAT32);
  CheckAlias(m);
  EXPECT_EQ(p.get(), m->Weight().get());
}
TEST_P(SharedMigrationTest, Bfloat16RoundTripPreservesAliasesAndGradient) {
  auto m = std::make_shared<SharedModel>();
  m->To(GetDevice());
  SetGrad(m->Weight(), .25f);
  m->To(DataType::kBFLOAT16);
  Sync(GetDevice());
  CheckAlias(m);
  EXPECT_EQ(m->Weight()->Dtype(), DataType::kBFLOAT16);
  EXPECT_TRUE(m->Weight()->requires_grad());
  ASSERT_NE(m->Weight()->grad(), nullptr);
  EXPECT_EQ(m->Weight()->grad()->Dtype(), DataType::kBFLOAT16);
  m->To(DataType::kFLOAT32);
  Sync(GetDevice());
  CheckAlias(m);
  EXPECT_FLOAT_EQ(Value(m->Weight()), 1);
  EXPECT_FLOAT_EQ(Value(m->Weight()->grad()), .25f);
}
TEST_P(SharedMigrationTest, SharedChildVisitedOnce) {
  auto m = std::make_shared<SharedModel>(true);
  m->To(GetDevice());
  auto p = m->Weight();
  m->To(DataType::kFLOAT32);
  EXPECT_EQ(m->Weight().get(), p.get());
  EXPECT_EQ(m->Parameters().size(), 1);
}
TEST_P(SharedMigrationTest, NullRegistrationsAreIgnored) {
  auto m = std::make_shared<SharedModel>();
  m->left->AddNulls();
  m->To(GetDevice());
  m->To(DataType::kFLOAT32);
  EXPECT_EQ(m->Weight().get(), m->right->parameter("weight").get());
}
TEST_P(SharedMigrationTest, ParentDeviceDoesNotHideMovedChild) {
  auto m = std::make_shared<SharedModel>();
  m->left->To(GetDevice());
  m->To(Device());
  Sync(GetDevice());
  EXPECT_EQ(m->left->parameter("weight")->GetDevice(), Device());
  EXPECT_EQ(m->right->parameter("weight")->GetDevice(), Device());
}
TEST_P(SharedMigrationTest, SharedBranchGradientAndSingleSGDUpdate) {
  auto m = std::make_shared<SharedModel>();
  m->To(GetDevice());
  m->To(DataType::kFLOAT32);
  auto loss = m->left->parameter("weight")->Mul(2)->Add(
      m->right->parameter("weight")->Mul(3));
  loss->Sum(0)->Backward();
  Sync(GetDevice());
  CheckAlias(m);
  EXPECT_NEAR(Value(m->Weight()->grad()), 5, 1e-6);
  optimizers::SGD opt(m->Parameters(), .1f);
  opt.Step();
  Sync(GetDevice());
  EXPECT_NEAR(Value(m->Weight()), .5f, 1e-6);
  EXPECT_NEAR(Value(m->right->parameter("weight")), .5f, 1e-6);
}
TEST_P(SharedMigrationTest, DiskCheckpointAndAdamResumePreserveAliases) {
  auto m = std::make_shared<SharedModel>();
  m->To(GetDevice());
  optimizers::Adam opt(m->NamedParameters(), .01f);
  SetGrad(m->Weight(), .5f);
  opt.Step();
  Sync(GetDevice());
  const auto base = std::getenv("INFINITRAIN_TEST_ARTIFACT_DIR");
  auto dir = (base ? std::filesystem::path(base)
                   : std::filesystem::temp_directory_path()) /
             ("shared_migration_" +
              std::to_string(
                  std::chrono::steady_clock::now().time_since_epoch().count()));
  Checkpoint::Save(dir, *m, &opt, TrainerState{.global_step = 1}, nullptr);
  auto restored = std::make_shared<SharedModel>();
  restored->To(GetDevice());
  optimizers::Adam resumed(restored->NamedParameters(), .01f);
  TrainerState state;
  Checkpoint::Load(dir, *restored, &resumed, state, nullptr);
  Sync(GetDevice());
  CheckAlias(restored);
  EXPECT_EQ(state.global_step, 1);
  EXPECT_FLOAT_EQ(Value(restored->Weight()), Value(m->Weight()));
  SetGrad(m->Weight(), -.2f);
  SetGrad(restored->Weight(), -.2f);
  opt.Step();
  resumed.Step();
  Sync(GetDevice());
  EXPECT_FLOAT_EQ(Value(restored->Weight()), Value(m->Weight()));
  CheckAlias(restored);
  // Keep this tiny, uniquely named checkpoint as a diagnostic artifact.
}
INFINI_TRAIN_REGISTER_TEST(SharedMigrationTest);
