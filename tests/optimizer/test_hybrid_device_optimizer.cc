#include <stddef.h>
#include <stdint.h>

#include <filesystem>
#include <initializer_list>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/checkpoint/checkpoint.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class HybridDeviceOptimizerTest : public infini_train::test::InfiniTrainTest {};

namespace {
std::shared_ptr<Tensor> MakeTensor(const std::vector<int64_t> &dims,
                                   DataType dtype, Device device, float value) {
  auto tensor = std::make_shared<Tensor>(dims, dtype, device);
  tensor->Fill(value);
  return tensor;
}

void SetGrad(const std::shared_ptr<Tensor> &param, float value) {
  param->set_grad(
      MakeTensor(param->Dims(), param->Dtype(), param->GetDevice(), value));
}

std::shared_ptr<Tensor> ToCpuFp32(const std::shared_ptr<Tensor> &tensor) {
  auto cpu = std::make_shared<Tensor>(tensor->To(Device()));
  if (!tensor->GetDevice().IsCPU()) {
    core::GetDeviceGuardImpl(tensor->GetDevice().type())
        ->SynchronizeDevice(tensor->GetDevice());
  }
  if (cpu->Dtype() == DataType::kFLOAT32) {
    return cpu;
  }
  return std::make_shared<Tensor>(cpu->To(DataType::kFLOAT32));
}

void ExpectTensorNear(const std::shared_ptr<Tensor> &lhs,
                      const std::shared_ptr<Tensor> &rhs, float tolerance) {
  auto lhs_cpu = ToCpuFp32(lhs);
  auto rhs_cpu = ToCpuFp32(rhs);
  ASSERT_EQ(lhs_cpu->Dims(), rhs_cpu->Dims());
  const auto *lhs_data = static_cast<const float *>(lhs_cpu->DataPtr());
  const auto *rhs_data = static_cast<const float *>(rhs_cpu->DataPtr());
  for (size_t i = 0; i < lhs_cpu->NumElements(); ++i) {
    EXPECT_NEAR(lhs_data[i], rhs_data[i], tolerance);
  }
}

std::unordered_map<std::string, std::shared_ptr<Tensor>> CloneStateDict(
    const std::unordered_map<std::string, std::shared_ptr<Tensor>>
        &state_dict) {
  std::unordered_map<std::string, std::shared_ptr<Tensor>> cloned;
  for (const auto &[key, tensor] : state_dict) {
    auto cpu =
        std::make_shared<Tensor>(tensor->Dims(), tensor->Dtype(), Device());
    cpu->CopyFrom(tensor);
    if (!tensor->GetDevice().IsCPU()) {
      core::GetDeviceGuardImpl(tensor->GetDevice().type())
          ->SynchronizeDevice(tensor->GetDevice());
    }
    cloned.emplace(key, std::move(cpu));
  }
  return cloned;
}

void ExpectFullOffloadMatchesAdam(Device device,
                                  bool overlap_cpu_optimizer_d2h_h2d) {
  auto ref = MakeTensor({8}, DataType::kFLOAT32, device, 1.0f);
  auto offloaded = MakeTensor({8}, DataType::kFLOAT32, device, 1.0f);
  SetGrad(ref, 0.25f);
  SetGrad(offloaded, 0.25f);

  optimizers::Adam ref_optimizer({ref}, 1e-3f);
  auto adam_creator = optimizers::Adam::Create(1e-3f);
  optimizers::HybridDeviceOptimizer offload_optimizer(
      {offloaded}, 1.0f, adam_creator, adam_creator,
      /*param_update_in_fp32=*/true,
      /*pin_cpu_grads=*/false,
      /*pin_cpu_params=*/false, overlap_cpu_optimizer_d2h_h2d);

  for (int step = 0; step < 3; ++step) {
    ref_optimizer.Step();
    offload_optimizer.Step();
  }

  ExpectTensorNear(ref, offloaded, 1e-6f);
}
}  // namespace

TEST_P(HybridDeviceOptimizerTest, FullOffloadMatchesAdam) {
  SKIP_CPU();
  ExpectFullOffloadMatchesAdam(GetDevice(),
                               /*overlap_cpu_optimizer_d2h_h2d=*/false);
}

TEST_P(HybridDeviceOptimizerTest, FullOffloadOverlapMatchesAdam) {
  SKIP_CPU();
  ExpectFullOffloadMatchesAdam(GetDevice(),
                               /*overlap_cpu_optimizer_d2h_h2d=*/true);
}

TEST_P(HybridDeviceOptimizerTest, OverlapBuildsPerParamCpuOptimizerStates) {
  SKIP_CPU();
  std::vector<std::shared_ptr<Tensor>> params;
  for (int i = 0; i < 2; ++i) {
    auto param = std::make_shared<Tensor>(std::vector<int64_t>{4},
                                          DataType::kFLOAT32, GetDevice());
    param->Fill(1.0f);
    auto grad = std::make_shared<Tensor>(std::vector<int64_t>{4},
                                         DataType::kFLOAT32, GetDevice());
    grad->Fill(0.25f);
    param->set_grad(grad);
    params.push_back(param);
  }

  auto adam_creator = optimizers::Adam::Create(1e-3f);
  optimizers::HybridDeviceOptimizer offload_optimizer(
      params, 1.0f, adam_creator, adam_creator,
      /*param_update_in_fp32=*/true,
      /*pin_cpu_grads=*/false,
      /*pin_cpu_params=*/false,
      /*overlap_cpu_optimizer_d2h_h2d=*/true);
  offload_optimizer.Step();
  const auto state = offload_optimizer.StateDict();
  EXPECT_TRUE(state.contains("hybrid.cpu.0.adam.t"));
  EXPECT_TRUE(state.contains("hybrid.cpu.1.adam.t"));
  EXPECT_TRUE(state.contains("hybrid.master_param.0"));
  EXPECT_TRUE(state.contains("hybrid.master_param.1"));
  EXPECT_TRUE(state.contains("hybrid.layout"));
}

TEST_P(HybridDeviceOptimizerTest, ZeroOffloadDoesNotRequireCpuOptimizer) {
  SKIP_CPU();
  auto ref = MakeTensor({8}, DataType::kFLOAT32, GetDevice(), 1.0f);
  auto actual = MakeTensor({8}, DataType::kFLOAT32, GetDevice(), 1.0f);
  SetGrad(ref, 0.125f);
  SetGrad(actual, 0.125f);

  optimizers::Adam ref_optimizer({ref}, 1e-3f);
  optimizers::HybridDeviceOptimizer actual_optimizer(
      {actual}, 0.0f, nullptr, optimizers::Adam::Create(1e-3f),
      /*param_update_in_fp32=*/false,
      /*pin_cpu_grads=*/true, /*pin_cpu_params=*/true,
      /*overlap_cpu_optimizer_d2h_h2d=*/true);
  for (int step = 0; step < 3; ++step) {
    ref_optimizer.Step();
    actual_optimizer.Step();
  }
  ExpectTensorNear(ref, actual, 1e-6f);
  EXPECT_TRUE(actual_optimizer.StateDict().contains("hybrid.device.adam.t"));
}

TEST_P(HybridDeviceOptimizerTest,
       PositiveFractionBelowOneElementStillOffloadsFirstParam) {
  SKIP_CPU();
  auto first = MakeTensor({4}, DataType::kFLOAT32, GetDevice(), 1.0f);
  auto second = MakeTensor({4}, DataType::kFLOAT32, GetDevice(), 2.0f);
  SetGrad(first, 0.125f);
  SetGrad(second, 0.25f);

  auto adam_creator = optimizers::Adam::Create(1e-3f);
  optimizers::HybridDeviceOptimizer optimizer(
      {first, second}, 0.01f, adam_creator, adam_creator,
      /*param_update_in_fp32=*/false,
      /*pin_cpu_grads=*/false, /*pin_cpu_params=*/false,
      /*overlap_cpu_optimizer_d2h_h2d=*/true);
  optimizer.Step();

  const auto state = optimizer.StateDict();
  EXPECT_TRUE(state.contains("hybrid.cpu.0.adam.t"));
  EXPECT_TRUE(state.contains("hybrid.device.adam.t"));
}

TEST_P(HybridDeviceOptimizerTest,
       PartialOffloadMixedPrecisionMatchesGpuMasterUpdate) {
  SKIP_CPU();
  std::vector<std::shared_ptr<Tensor>> ref_params;
  std::vector<std::shared_ptr<Tensor>> actual_params;
  for (int i = 0; i < 2; ++i) {
    ref_params.push_back(
        MakeTensor({16}, DataType::kBFLOAT16, GetDevice(), 1.0f + i));
    actual_params.push_back(
        MakeTensor({16}, DataType::kBFLOAT16, GetDevice(), 1.0f + i));
    SetGrad(ref_params.back(), 0.125f + 0.125f * i);
    SetGrad(actual_params.back(), 0.125f + 0.125f * i);
  }

  auto adam_creator = optimizers::Adam::Create(1e-3f);
  optimizers::HybridDeviceOptimizer ref_optimizer(
      ref_params, 0.0f, nullptr, adam_creator,
      /*param_update_in_fp32=*/true,
      /*pin_cpu_grads=*/false, /*pin_cpu_params=*/false,
      /*overlap_cpu_optimizer_d2h_h2d=*/true);
  optimizers::HybridDeviceOptimizer actual_optimizer(
      actual_params, 0.5f, adam_creator, adam_creator,
      /*param_update_in_fp32=*/true,
      /*pin_cpu_grads=*/false, /*pin_cpu_params=*/false,
      /*overlap_cpu_optimizer_d2h_h2d=*/true);
  for (int step = 0; step < 3; ++step) {
    ref_optimizer.Step();
    actual_optimizer.Step();
  }
  for (size_t i = 0; i < ref_params.size(); ++i) {
    ExpectTensorNear(ref_params[i], actual_params[i], 0.0f);
  }
}

TEST_P(HybridDeviceOptimizerTest,
       ModelOnlyCheckpointReloadMatchesFreshOptimizer) {
  SKIP_CPU();
  const auto dir =
      std::filesystem::temp_directory_path() / "test_hdo_model_only";
  for (const auto dtype : {DataType::kFLOAT32, DataType::kBFLOAT16}) {
    auto source = std::make_shared<nn::Linear>(1, 4, true, GetDevice());
    source->To(dtype);
    source->parameter("weight")->Fill(1.5f);
    source->parameter("bias")->Fill(-0.5f);
    std::filesystem::remove_all(dir);
    Checkpoint::Save(dir, *source, nullptr, TrainerState{.global_step = 7},
                     nullptr);

    for (const float fraction : {0.0f, 0.5f, 1.0f}) {
      for (const bool overlap : {false, true}) {
        SCOPED_TRACE(::testing::Message()
                     << "dtype=" << static_cast<int>(dtype)
                     << ", fraction=" << fraction << ", overlap=" << overlap);
        auto resumed = std::make_shared<nn::Linear>(1, 4, true, GetDevice());
        resumed->To(dtype);
        for (const auto &param : resumed->Parameters()) {
          param->Fill(-4.0f);
        }
        auto adam_creator = optimizers::Adam::Create(1e-3f);
        optimizers::HybridDeviceOptimizer optimizer(
            resumed->Parameters(), fraction, adam_creator, adam_creator,
            /*param_update_in_fp32=*/true, /*pin_cpu_grads=*/false,
            /*pin_cpu_params=*/false, overlap);

        // Exercise the real model-only load path with an already-created HDO.
        TrainerState loaded;
        Checkpoint::Load(dir, *resumed, nullptr, loaded, nullptr);
        ASSERT_EQ(loaded.global_step, 7);
        optimizer.ReloadModelParams();

        auto reference = std::make_shared<nn::Linear>(1, 4, true, GetDevice());
        reference->To(dtype);
        Checkpoint::Load(dir, *reference, nullptr, loaded, nullptr);
        optimizers::HybridDeviceOptimizer reference_optimizer(
            reference->Parameters(), 0.0f, nullptr, adam_creator,
            /*param_update_in_fp32=*/true, /*pin_cpu_grads=*/false,
            /*pin_cpu_params=*/false, overlap);
        const auto params = resumed->Parameters();
        const auto reference_params = reference->Parameters();
        for (size_t i = 0; i < params.size(); ++i) {
          SetGrad(params[i], 0.25f);
          SetGrad(reference_params[i], 0.25f);
        }
        for (int step = 0; step < 3; ++step) {
          optimizer.Step();
          reference_optimizer.Step();
          for (size_t i = 0; i < params.size(); ++i) {
            ExpectTensorNear(params[i], reference_params[i], 1e-6f);
          }
        }
      }
    }
  }
  std::filesystem::remove_all(dir);
}

TEST_P(HybridDeviceOptimizerTest, ReloadModelParamsPreservesOptimizerState) {
  SKIP_CPU();
  std::vector<std::shared_ptr<Tensor>> params;
  for (int i = 0; i < 2; ++i) {
    params.push_back(MakeTensor({4}, DataType::kBFLOAT16, GetDevice(), 1.0f));
    SetGrad(params.back(), 0.25f);
  }
  const auto adam_creator = optimizers::Adam::Create(1e-3f);
  optimizers::HybridDeviceOptimizer optimizer(
      params, 0.5f, adam_creator, adam_creator, /*param_update_in_fp32=*/true,
      /*pin_cpu_grads=*/false, /*pin_cpu_params=*/false,
      /*overlap_cpu_optimizer_d2h_h2d=*/true);
  optimizer.Step();
  const auto before = CloneStateDict(optimizer.StateDict());
  for (const auto &param : params) {
    param->Fill(2.0f);
  }
  optimizer.ReloadModelParams();
  const auto after = optimizer.StateDict();
  for (const auto &[key, tensor] : before) {
    if (key.starts_with("hybrid.master_param.")) {
      auto expected =
          MakeTensor(tensor->Dims(), tensor->Dtype(), Device(), 2.0f);
      ExpectTensorNear(after.at(key), expected, 0.0f);
    } else {
      ExpectTensorNear(after.at(key), tensor, 0.0f);
    }
  }
}

TEST_P(HybridDeviceOptimizerTest, StateLoadRefreshesInnerParams) {
  SKIP_CPU();
  std::vector<std::shared_ptr<Tensor>> source_params;
  std::vector<std::shared_ptr<Tensor>> resumed_params;
  for (int i = 0; i < 2; ++i) {
    source_params.push_back(
        MakeTensor({8}, DataType::kFLOAT32, GetDevice(), 1.0f + i));
    resumed_params.push_back(
        MakeTensor({8}, DataType::kFLOAT32, GetDevice(), -4.0f - i));
    SetGrad(source_params.back(), 0.25f + 0.125f * i);
  }

  auto adam_creator = optimizers::Adam::Create(1e-3f);
  optimizers::HybridDeviceOptimizer source_optimizer(
      source_params, 1.0f, adam_creator, nullptr,
      /*param_update_in_fp32=*/false,
      /*pin_cpu_grads=*/false, /*pin_cpu_params=*/false,
      /*overlap_cpu_optimizer_d2h_h2d=*/true);
  for (int step = 0; step < 3; ++step) {
    source_optimizer.Step();
  }
  const auto state = CloneStateDict(source_optimizer.StateDict());

  // Match Checkpoint::Load order: HDO exists before model tensors and optimizer
  // state are loaded.
  optimizers::HybridDeviceOptimizer resumed_optimizer(
      resumed_params, 1.0f, adam_creator, nullptr,
      /*param_update_in_fp32=*/false,
      /*pin_cpu_grads=*/false, /*pin_cpu_params=*/false,
      /*overlap_cpu_optimizer_d2h_h2d=*/true);
  for (size_t i = 0; i < source_params.size(); ++i) {
    resumed_params[i]->CopyFrom(source_params[i]);
    SetGrad(resumed_params[i], 0.25f + 0.125f * i);
  }
  resumed_optimizer.LoadStateDict(state);

  source_optimizer.Step();
  resumed_optimizer.Step();
  for (size_t i = 0; i < source_params.size(); ++i) {
    ExpectTensorNear(source_params[i], resumed_params[i], 1e-6f);
  }
}

TEST_P(HybridDeviceOptimizerTest, StateLoadPreservesMixedPrecisionMasterParam) {
  SKIP_CPU();
  auto source = MakeTensor({16}, DataType::kBFLOAT16, GetDevice(), 1.0f);
  auto resumed = MakeTensor({16}, DataType::kBFLOAT16, GetDevice(), -2.0f);
  SetGrad(source, 0.25f);

  auto adam_creator = optimizers::Adam::Create(1e-3f);
  optimizers::HybridDeviceOptimizer source_optimizer(
      {source}, 1.0f, adam_creator, nullptr, /*param_update_in_fp32=*/true,
      /*pin_cpu_grads=*/false, /*pin_cpu_params=*/false,
      /*overlap_cpu_optimizer_d2h_h2d=*/true);
  for (int step = 0; step < 3; ++step) {
    source_optimizer.Step();
  }
  const auto state = CloneStateDict(source_optimizer.StateDict());

  optimizers::HybridDeviceOptimizer resumed_optimizer(
      {resumed}, 1.0f, adam_creator, nullptr, /*param_update_in_fp32=*/true,
      /*pin_cpu_grads=*/false, /*pin_cpu_params=*/false,
      /*overlap_cpu_optimizer_d2h_h2d=*/true);
  resumed->CopyFrom(source);
  SetGrad(resumed, 0.25f);
  resumed_optimizer.LoadStateDict(state);

  source_optimizer.Step();
  resumed_optimizer.Step();
  ExpectTensorNear(source, resumed, 0.0f);
}

INFINI_TRAIN_REGISTER_TEST(HybridDeviceOptimizerTest);
