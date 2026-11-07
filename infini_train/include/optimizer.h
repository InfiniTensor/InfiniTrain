#pragma once

#include <stddef.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "infini_train/include/common/hook.h"
#include "infini_train/include/device.h"

namespace infini_train {
class Tensor;
namespace core {
class Event;
class Stream;
}  // namespace core

}  // namespace infini_train
namespace infini_train {
class Optimizer;
using OptimizerPostStepHook = std::function<void(Optimizer &optimizer)>;

using NamedParameter = std::pair<std::string, std::shared_ptr<Tensor>>;
using NamedParameterList = std::vector<NamedParameter>;
using OptimizerCreator = std::function<std::shared_ptr<Optimizer>(
    const std::vector<std::shared_ptr<Tensor>> &params)>;
using OptimizerCreatorNamed = std::function<std::shared_ptr<Optimizer>(
    const NamedParameterList &named_params)>;

class Optimizer {
 public:
  explicit Optimizer(const std::vector<std::shared_ptr<Tensor>> &params,
                     float learning_rate);

  Optimizer(const NamedParameterList &named_params, float learning_rate);

  virtual void ZeroGrad(bool set_to_none = true);

  virtual ~Optimizer() = default;

  void Step();

  std::shared_ptr<HookHandle> RegisterPostStepHook(OptimizerPostStepHook hook);

  virtual std::unordered_map<std::string, std::shared_ptr<Tensor>> StateDict()
      const;

  NamedParameterList NamedParameters() const;

  virtual void LoadStateDict(
      const std::unordered_map<std::string, std::shared_ptr<Tensor>>
          &state_dict) {}

  virtual void set_learning_rate(float lr);

  virtual float learning_rate() const;

  float initial_learning_rate() const;

  bool initial_lr_set() const;

  void set_initial_learning_rate(float lr);

 protected:
  virtual void StepImpl() = 0;

  std::vector<std::shared_ptr<Tensor>> params_;
  std::vector<std::string> parameter_names_;
  float learning_rate_ = 0.0f;
  float initial_learning_rate_ = 0.0f;
  bool initial_lr_set_ = false;
  std::vector<OptimizerPostStepHook> optimizer_step_post_hooks_;
};

namespace optimizers {
inline constexpr std::string_view kAdamOptimizerPrefix = "adam.";
inline constexpr std::string_view kAdamFirstMomentPrefix = "adam.m.";
inline constexpr std::string_view kAdamSecondMomentPrefix = "adam.v.";
inline constexpr std::string_view kAdamStepKey = "adam.t";

class SGD : public Optimizer {
 public:
  SGD(const std::vector<std::shared_ptr<Tensor>> &params, float learning_rate);
  SGD(const NamedParameterList &named_params, float learning_rate);

  static OptimizerCreator Create(float learning_rate);
  static OptimizerCreatorNamed CreateNamed(float learning_rate);

 protected:
  void StepImpl() override;
};

class Adam : public Optimizer {
 public:
  Adam(const std::vector<std::shared_ptr<Tensor>> &params,
       float learning_rate = 1e-3, float beta1 = 0.9, float beta2 = 0.999,
       float eps = 1e-8);
  Adam(const NamedParameterList &named_params, float learning_rate = 1e-3,
       float beta1 = 0.9, float beta2 = 0.999, float eps = 1e-8);

  std::unordered_map<std::string, std::shared_ptr<Tensor>> StateDict()
      const override;

  void LoadStateDict(
      const std::unordered_map<std::string, std::shared_ptr<Tensor>>
          &state_dict) override;
  static OptimizerCreator Create(float learning_rate = 1e-3, float beta1 = 0.9,
                                 float beta2 = 0.999, float eps = 1e-8);
  static OptimizerCreatorNamed CreateNamed(float learning_rate = 1e-3,
                                           float beta1 = 0.9,
                                           float beta2 = 0.999,
                                           float eps = 1e-8);

 protected:
  void StepImpl() override;

 private:
  int64_t t_;
  const float beta1_;
  const float beta2_;
  const float eps_;
  std::vector<std::shared_ptr<Tensor>> m_;
  std::vector<std::shared_ptr<Tensor>> v_;
};
class HybridDeviceOptimizer final : public Optimizer {
 public:
  HybridDeviceOptimizer(const std::vector<std::shared_ptr<Tensor>> &params,
                        float offload_fraction = 0.5f,
                        OptimizerCreator cpu_optimizer_creator = nullptr,
                        OptimizerCreator gpu_optimizer_creator = nullptr,
                        bool param_update_in_fp32 = false,
                        bool pin_cpu_grads = true, bool pin_cpu_params = true,
                        bool overlap_cpu_optimizer_d2h_h2d = true);
  ~HybridDeviceOptimizer() override;

  // Refresh inner parameters after loading model weights without optimizer
  // state.
  void ReloadModelParams();

  void ZeroGrad(bool set_to_none = true) override;

  std::unordered_map<std::string, std::shared_ptr<Tensor>> StateDict()
      const override;

  void LoadStateDict(
      const std::unordered_map<std::string, std::shared_ptr<Tensor>>
          &state_dict) override;

  void set_learning_rate(float lr) override;

  float learning_rate() const override;

  static OptimizerCreator Create(
      float offload_fraction = 0.5f,
      OptimizerCreator cpu_optimizer_creator = nullptr,
      OptimizerCreator gpu_optimizer_creator = nullptr,
      bool param_update_in_fp32 = false, bool pin_cpu_grads = true,
      bool pin_cpu_params = true, bool overlap_cpu_optimizer_d2h_h2d = true);

 protected:
  void StepImpl() override;

 private:
  struct ParamMapping {
    std::shared_ptr<Tensor> inner_param;
    size_t offloaded_index = static_cast<size_t>(-1);
    bool uses_master_param = false;
  };

  struct OffloadedParam {
    std::shared_ptr<Tensor> cpu_grad;
    std::shared_ptr<Tensor> cpu_grad_staging;
    std::shared_ptr<Tensor> h2d_staging;
  };

  struct CpuOptimizerMetadata {
    std::vector<size_t> original_indices;
    core::Event *d2h_done = nullptr;
    core::Event *h2d_ready = nullptr;
    core::Event *h2d_done = nullptr;
  };

  void BuildParamPartitions(float offload_fraction,
                            const OptimizerCreator &cpu_optimizer_creator,
                            const OptimizerCreator &gpu_optimizer_creator,
                            bool param_update_in_fp32,
                            bool overlap_cpu_optimizer_d2h_h2d);
  void RegisterCpuParamCopyBackHook(
      const std::shared_ptr<Optimizer> &cpu_optimizer);

 private:
  std::vector<ParamMapping> param_mappings_;
  std::vector<size_t> gpu_fp32_param_indices_;
  std::vector<OffloadedParam> offloaded_params_;
  std::shared_ptr<Optimizer> gpu_optimizer_;
  std::vector<std::shared_ptr<Optimizer>> cpu_optimizers_;
  std::unordered_map<Optimizer *, CpuOptimizerMetadata> cpu_optimizer_metadata_;
  bool overlap_cpu_optimizer_d2h_h2d_ = false;
  Device device_;
  core::Stream *d2h_stream_ = nullptr;
  core::Stream *h2d_stream_ = nullptr;
  core::Event *d2h_ready_event_ = nullptr;
};
}  // namespace optimizers
}  // namespace infini_train
