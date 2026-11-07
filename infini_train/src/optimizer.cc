#include "infini_train/include/optimizer.h"

#include <stddef.h>

#include <format>
#include <ostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/core/runtime/runtime_common.h"
#include "infini_train/include/core/runtime/stream_guard.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train {
Optimizer::Optimizer(const std::vector<std::shared_ptr<Tensor>> &params,
                     float learning_rate)
    : params_(params), learning_rate_(learning_rate) {}

Optimizer::Optimizer(const NamedParameterList &named_params,
                     float learning_rate)
    : learning_rate_(learning_rate) {
  if (named_params.empty()) {
    return;
  }

  params_.reserve(named_params.size());
  parameter_names_.reserve(named_params.size());
  for (const auto &[name, parameter] : named_params) {
    params_.push_back(parameter);
    parameter_names_.push_back(name);
  }
}

void Optimizer::ZeroGrad(bool set_to_none) {
  for (auto param : params_) {
    param->ZeroGrad(set_to_none);
  }
}

std::unordered_map<std::string, std::shared_ptr<Tensor>> Optimizer::StateDict()
    const {
  return {};
}

NamedParameterList Optimizer::NamedParameters() const {
  NamedParameterList result;
  result.reserve(params_.size());
  for (size_t i = 0; i < params_.size(); ++i) {
    result.emplace_back(
        parameter_names_.empty() ? std::to_string(i) : parameter_names_[i],
        params_[i]);
  }
  return result;
}

void Optimizer::Step() {
  // TODO(zbl): Pre step hooks

  StepImpl();

  // Post step hooks
  const auto num_hooks = optimizer_step_post_hooks_.size();
  for (size_t i = 0; i < num_hooks; ++i) {
    const auto hook = optimizer_step_post_hooks_[i];
    if (hook) {
      hook(*this);
    }
  }
}

void Optimizer::set_learning_rate(float lr) { learning_rate_ = lr; }

float Optimizer::learning_rate() const { return learning_rate_; }

std::shared_ptr<HookHandle> Optimizer::RegisterPostStepHook(
    OptimizerPostStepHook hook) {
  CHECK(hook) << "Optimizer cannot register a null post-step hook.";
  optimizer_step_post_hooks_.push_back(std::move(hook));
  return std::make_shared<HookHandleImpl<OptimizerPostStepHook>>(
      &optimizer_step_post_hooks_, optimizer_step_post_hooks_.size() - 1);
}

float Optimizer::initial_learning_rate() const {
  CHECK(initial_lr_set_) << "Optimizer: initial_learning_rate not set. "
                            "Use with an LRScheduler first.";
  return initial_learning_rate_;
}

bool Optimizer::initial_lr_set() const { return initial_lr_set_; }

void Optimizer::set_initial_learning_rate(float lr) {
  CHECK(!initial_lr_set_)
      << "Optimizer: initial_learning_rate has already been set.";
  initial_learning_rate_ = lr;
  initial_lr_set_ = true;
}

namespace optimizers {

SGD::SGD(const std::vector<std::shared_ptr<Tensor>> &params,
         float learning_rate)
    : Optimizer(params, learning_rate) {}

SGD::SGD(const NamedParameterList &named_params, float learning_rate)
    : Optimizer(named_params, learning_rate) {}

void SGD::StepImpl() {
  for (auto param : params_) {
    if (!param->grad()) {
      LOG(INFO) << "Skipping param with null grad.";
      continue;
    }
    auto device = param->GetDevice();
    core::DeviceGuard guard(device);
    auto kernel =
        Dispatcher::Instance().GetKernel({device.type(), "AccumulateGrad"});
    kernel.Call<void>(param->grad(), -learning_rate_, param);
  }
}

OptimizerCreator SGD::Create(float learning_rate) {
  return [learning_rate](const std::vector<std::shared_ptr<Tensor>> &params) {
    return std::make_shared<SGD>(params, learning_rate);
  };
}

OptimizerCreatorNamed SGD::CreateNamed(float learning_rate) {
  return [learning_rate](const NamedParameterList &named_params) {
    return std::make_shared<SGD>(named_params, learning_rate);
  };
}

Adam::Adam(const std::vector<std::shared_ptr<Tensor>> &params,
           float learning_rate, float beta1, float beta2, float eps)
    : Optimizer(params, learning_rate),
      t_(0),
      beta1_(beta1),
      beta2_(beta2),
      eps_(eps) {
  for (const auto &param : params_) {
    m_.emplace_back(std::make_shared<Tensor>(param->Dims(), param->Dtype(),
                                             param->GetDevice()));
    v_.emplace_back(std::make_shared<Tensor>(param->Dims(), param->Dtype(),
                                             param->GetDevice()));
    m_.back()->Fill(0.0);
    v_.back()->Fill(0.0);
  }
}

Adam::Adam(const NamedParameterList &named_params, float learning_rate,
           float beta1, float beta2, float eps)
    : Optimizer(named_params, learning_rate),
      t_(0),
      beta1_(beta1),
      beta2_(beta2),
      eps_(eps) {
  for (const auto &[name, param] : named_params) {
    m_.emplace_back(std::make_shared<Tensor>(param->Dims(), param->Dtype(),
                                             param->GetDevice()));
    v_.emplace_back(std::make_shared<Tensor>(param->Dims(), param->Dtype(),
                                             param->GetDevice()));
    m_.back()->Fill(0.0);
    v_.back()->Fill(0.0);
  }
}

void Adam::StepImpl() {
  ++t_;

  for (size_t i = 0; i < params_.size(); ++i) {
    auto &param = params_[i];
    const auto &grad = param->grad();
    if (!grad) {
      LOG(INFO) << "Skipping param with null grad.";
      continue;
    }
    auto &m = m_[i];
    auto &v = v_[i];

    auto device = param->GetDevice();
    core::DeviceGuard guard(device);
    auto kernel =
        Dispatcher::Instance().GetKernel({device.type(), "AdamAccumulateGrad"});
    kernel.Call<void>(grad, param, m, v, learning_rate_, beta1_, beta2_, eps_,
                      t_);
  }
}

OptimizerCreator Adam::Create(float learning_rate, float beta1, float beta2,
                              float eps) {
  return [=](const std::vector<std::shared_ptr<Tensor>> &params) {
    return std::make_shared<Adam>(params, learning_rate, beta1, beta2, eps);
  };
}

OptimizerCreatorNamed Adam::CreateNamed(float learning_rate, float beta1,
                                        float beta2, float eps) {
  return [=](const NamedParameterList &named_params) {
    return std::make_shared<Adam>(named_params, learning_rate, beta1, beta2,
                                  eps);
  };
}

std::unordered_map<std::string, std::shared_ptr<Tensor>> Adam::StateDict()
    const {
  std::unordered_map<std::string, std::shared_ptr<Tensor>> state;
  for (size_t i = 0; i < m_.size(); ++i) {
    const auto suffix =
        parameter_names_.empty() ? std::to_string(i) : parameter_names_[i];
    state.emplace(std::string(kAdamFirstMomentPrefix) + suffix, m_[i]);
    state.emplace(std::string(kAdamSecondMomentPrefix) + suffix, v_[i]);
  }

  auto t_tensor = std::make_shared<Tensor>(std::vector<int64_t>{},
                                           DataType::kINT64, Device());
  *static_cast<int64_t *>(t_tensor->DataPtr()) = t_;
  state.emplace(std::string(kAdamStepKey), t_tensor);
  return state;
}

void Adam::LoadStateDict(
    const std::unordered_map<std::string, std::shared_ptr<Tensor>>
        &state_dict) {
  for (size_t i = 0; i < m_.size(); ++i) {
    const auto suffix =
        parameter_names_.empty() ? std::to_string(i) : parameter_names_[i];
    const auto m_key = std::string(kAdamFirstMomentPrefix) + suffix;
    const auto v_key = std::string(kAdamSecondMomentPrefix) + suffix;
    CHECK(state_dict.contains(m_key)) << "Missing optimizer state: " << m_key;
    CHECK(state_dict.contains(v_key)) << "Missing optimizer state: " << v_key;
    m_[i]->CopyFrom(state_dict.at(m_key));
    v_[i]->CopyFrom(state_dict.at(v_key));
  }

  const std::string step_key(kAdamStepKey);
  CHECK(state_dict.contains(step_key))
      << "Missing optimizer state: " << step_key;
  const Tensor t_cpu = state_dict.at(step_key)->To(Device());
  t_ = *static_cast<const int64_t *>(t_cpu.DataPtr());
}
namespace {
std::shared_ptr<Tensor> ToCpuInnerParam(const std::shared_ptr<Tensor> &tensor,
                                        DataType dtype) {
  auto cpu_tensor = std::make_shared<Tensor>(tensor->Detach()->To(Device()));
  core::GetDeviceGuardImpl(tensor->GetDevice().type())
      ->SynchronizeDevice(tensor->GetDevice());
  if (cpu_tensor->Dtype() == dtype) {
    return cpu_tensor;
  }
  return std::make_shared<Tensor>(cpu_tensor->To(dtype));
}

std::string PrefixKey(std::string_view prefix, std::string_view key) {
  return std::format("{}.{}", prefix, key);
}

std::unordered_map<std::string, std::shared_ptr<Tensor>> ExtractPrefixedState(
    const std::unordered_map<std::string, std::shared_ptr<Tensor>> &state_dict,
    std::string_view prefix) {
  std::unordered_map<std::string, std::shared_ptr<Tensor>> state;
  const std::string key_prefix = std::format("{}.", prefix);
  for (const auto &[key, tensor] : state_dict) {
    if (key.starts_with(key_prefix)) {
      state.emplace(key.substr(key_prefix.size()), tensor);
    }
  }
  return state;
}
}  // namespace

void HybridDeviceOptimizer::RegisterCpuParamCopyBackHook(
    const std::shared_ptr<Optimizer> &cpu_optimizer) {
  cpu_optimizer->RegisterPostStepHook([this](Optimizer &optimizer) {
    auto &metadata = cpu_optimizer_metadata_.at(&optimizer);
    core::DeviceGuard guard(device_);
    auto *impl = core::GetDeviceGuardImpl(device_.type());
    auto *current_stream = impl->GetStream(device_);
    impl->EventRecord(metadata.h2d_ready, current_stream);
    impl->StreamWaitEvent(h2d_stream_, metadata.h2d_ready, 0);
    {
      core::StreamGuard stream_guard(device_, h2d_stream_);
      for (const auto original_index : metadata.original_indices) {
        const auto offloaded_index =
            param_mappings_[original_index].offloaded_index;
        CHECK_LT(offloaded_index, offloaded_params_.size())
            << "HybridDeviceOptimizer CPU parameter copy-back hook got invalid "
               "index.";
        auto &entry = offloaded_params_[offloaded_index];
        const auto &original = params_[original_index];
        const auto &cpu_param = param_mappings_[original_index].inner_param;
        if (original->Dtype() == cpu_param->Dtype()) {
          original->CopyFrom(cpu_param);
        } else {
          entry.h2d_staging = std::make_shared<Tensor>(
              cpu_param->Detach()->To(original->Dtype()));
          original->CopyFrom(entry.h2d_staging);
        }
      }
    }
    impl->EventRecord(metadata.h2d_done, h2d_stream_);
    impl->StreamWaitEvent(current_stream, metadata.h2d_done, 0);
  });
}

HybridDeviceOptimizer::HybridDeviceOptimizer(
    const std::vector<std::shared_ptr<Tensor>> &params, float offload_fraction,
    OptimizerCreator cpu_optimizer_creator,
    OptimizerCreator gpu_optimizer_creator, bool param_update_in_fp32,
    bool pin_cpu_grads, bool pin_cpu_params, bool overlap_cpu_optimizer_d2h_h2d)
    : Optimizer(params, 0.0f) {
  CHECK_GE(offload_fraction, 0.0f) << "HybridDeviceOptimizer: Optimizer CPU "
                                      "offload fraction must be in [0, 1].";
  CHECK_LE(offload_fraction, 1.0f) << "HybridDeviceOptimizer: Optimizer CPU "
                                      "offload fraction must be in [0, 1].";
  CHECK(offload_fraction == 0.0f || cpu_optimizer_creator)
      << "HybridDeviceOptimizer requires cpu_optimizer_creator for CPU "
         "parameters.";
  CHECK(offload_fraction == 1.0f || gpu_optimizer_creator)
      << "HybridDeviceOptimizer requires gpu_optimizer_creator for GPU "
         "parameters.";

  if (offload_fraction > 0.0f && (pin_cpu_params || pin_cpu_grads)) {
    // TODO(zbl): Add a pinned host allocator to TensorBuffer and plumb it
    // through CPU offload buffers.
    LOG(FATAL) << "HybridDeviceOptimizer does not support pinned CPU offload "
                  "buffers yet. ";
  }
  BuildParamPartitions(offload_fraction, cpu_optimizer_creator,
                       gpu_optimizer_creator, param_update_in_fp32,
                       overlap_cpu_optimizer_d2h_h2d);
  if (!offloaded_params_.empty()) {
    device_ = params_[cpu_optimizer_metadata_.at(cpu_optimizers_.front().get())
                          .original_indices.front()]
                  ->GetDevice();
    for (const auto &cpu_optimizer : cpu_optimizers_) {
      for (const auto original_index :
           cpu_optimizer_metadata_.at(cpu_optimizer.get()).original_indices) {
        if (params_[original_index]->GetDevice() != device_) {
          // TODO(zbl): Create one D2H/H2D stream pair per non-CPU device.
          LOG(FATAL) << "HybridDeviceOptimizer currently requires all "
                        "offloaded params on one device.";
        }
      }
    }
    overlap_cpu_optimizer_d2h_h2d_ = overlap_cpu_optimizer_d2h_h2d;
    core::DeviceGuard guard(device_);
    auto *impl = core::GetDeviceGuardImpl(device_.type());
    d2h_stream_ = impl->GetStream(device_);
    h2d_stream_ = d2h_stream_;
    if (overlap_cpu_optimizer_d2h_h2d_) {
      d2h_stream_ = impl->CreateStream(device_);
      h2d_stream_ = impl->CreateStream(device_);
    }
    impl->EventCreateWithFlags(&d2h_ready_event_,
                               core::EventFlag::kDisableTiming);
    for (const auto &cpu_optimizer : cpu_optimizers_) {
      auto &metadata = cpu_optimizer_metadata_.at(cpu_optimizer.get());
      impl->EventCreateWithFlags(&metadata.d2h_done,
                                 core::EventFlag::kDisableTiming);
      impl->EventCreateWithFlags(&metadata.h2d_ready,
                                 core::EventFlag::kDisableTiming);
      impl->EventCreateWithFlags(&metadata.h2d_done,
                                 core::EventFlag::kDisableTiming);
    }
    if (overlap_cpu_optimizer_d2h_h2d_) {
      LOG(INFO)
          << "HybridDeviceOptimizer: enabled device/host stream overlap on "
          << device_;
    }
  }
}

HybridDeviceOptimizer::~HybridDeviceOptimizer() {
  if (!d2h_ready_event_) {
    return;
  }
  core::DeviceGuard guard(device_);
  auto *impl = core::GetDeviceGuardImpl(device_.type());
  impl->SynchronizeStream(d2h_stream_);
  if (h2d_stream_ != d2h_stream_) {
    impl->SynchronizeStream(h2d_stream_);
  }
  impl->EventDestroy(d2h_ready_event_);
  for (const auto &cpu_optimizer : cpu_optimizers_) {
    const auto &metadata = cpu_optimizer_metadata_.at(cpu_optimizer.get());
    impl->EventDestroy(metadata.d2h_done);
    impl->EventDestroy(metadata.h2d_ready);
    impl->EventDestroy(metadata.h2d_done);
  }
  if (overlap_cpu_optimizer_d2h_h2d_) {
    impl->DestroyStream(d2h_stream_);
    impl->DestroyStream(h2d_stream_);
  }
}

void HybridDeviceOptimizer::BuildParamPartitions(
    float offload_fraction, const OptimizerCreator &cpu_optimizer_creator,
    const OptimizerCreator &gpu_optimizer_creator, bool param_update_in_fp32,
    bool overlap_cpu_optimizer_d2h_h2d) {
  // TODO(zbl): Preserve and synchronize optimizer param_groups after
  // InfiniTrain adds group-level attributes.
  param_mappings_.resize(params_.size());

  size_t device_num_elements = 0;
  for (const auto &param : params_) {
    if (param->GetDevice().IsCPU()) {
      // TODO(zbl): Support pre-existing CPU parameters in mixed CPU/device
      // optimizer inputs.
      LOG(FATAL) << "HybridDeviceOptimizer currently expects every original "
                    "parameter on a non-CPU device.";
    }
    device_num_elements += param->NumElements();
  }

  const double target_offload_num_elements =
      static_cast<double>(device_num_elements) * offload_fraction;
  size_t selected_num_elements = 0;
  size_t offloaded_num_elements = 0;
  size_t offloaded_param_bytes = 0;
  std::vector<std::shared_ptr<Tensor>> gpu_params;
  std::vector<std::shared_ptr<Tensor>> cpu_params;
  std::vector<size_t> cpu_param_indices;

  for (size_t original_index = 0; original_index < params_.size();
       ++original_index) {
    const auto &param = params_[original_index];
    if (selected_num_elements >= target_offload_num_elements) {
      // Not offload, stay in GPU
      auto inner_param = param;
      if (param_update_in_fp32 && param->Dtype() != DataType::kFLOAT32) {
        inner_param =
            std::make_shared<Tensor>(param->Detach()->To(DataType::kFLOAT32));
      }
      param_mappings_[original_index].inner_param = inner_param;
      param_mappings_[original_index].uses_master_param =
          inner_param.get() != param.get();
      if (inner_param.get() != param.get()) {
        gpu_fp32_param_indices_.push_back(original_index);
      }
      gpu_params.push_back(std::move(inner_param));
    } else {
      // Offload to CPU
      if (param->Dtype() != DataType::kFLOAT32 && !param_update_in_fp32) {
        // TODO(zbl): Add CPU optimizer support for non-fp32 inner params before
        // allowing this path.
        LOG(FATAL)
            << "HybridDeviceOptimizer requires param_update_in_fp32=true for "
               "non-fp32 offloaded params.";
      }
      const auto inner_dtype =
          param_update_in_fp32 ? DataType::kFLOAT32 : param->Dtype();
      auto cpu_param = ToCpuInnerParam(param, inner_dtype);
      auto cpu_grad =
          std::make_shared<Tensor>(param->Dims(), inner_dtype, Device());
      std::shared_ptr<Tensor> cpu_grad_staging = nullptr;
      if (param->Dtype() != inner_dtype) {
        cpu_grad_staging =
            std::make_shared<Tensor>(param->Dims(), param->Dtype(), Device());
      }
      cpu_params.push_back(cpu_param);
      cpu_param_indices.push_back(original_index);
      param_mappings_[original_index].inner_param = cpu_param;
      param_mappings_[original_index].offloaded_index =
          offloaded_params_.size();
      param_mappings_[original_index].uses_master_param = param_update_in_fp32;
      offloaded_params_.push_back({cpu_grad, cpu_grad_staging, nullptr});
      selected_num_elements += param->NumElements();
      offloaded_num_elements += param->NumElements();
      offloaded_param_bytes += param->SizeInBytes();
    }
  }

  // Build optimizers and register hooks
  if (!gpu_params.empty()) {
    gpu_optimizer_ = gpu_optimizer_creator(gpu_params);
    if (!gpu_fp32_param_indices_.empty()) {
      gpu_optimizer_->RegisterPostStepHook([this](Optimizer &) {
        for (const auto original_index : gpu_fp32_param_indices_) {
          const auto &original = params_[original_index];
          const auto &fp32_param = param_mappings_[original_index].inner_param;
          original->CopyFrom(std::make_shared<Tensor>(
              fp32_param->Detach()->To(original->Dtype())));
        }
      });
    }
  }
  if (!cpu_params.empty()) {
    if (overlap_cpu_optimizer_d2h_h2d) {
      // NOTE(zbl): If overlap is on, use one optimizer per CPU param, so each
      // param can independently finish D2H, optimizer step, and H2D copy-back
      // as soon as its gradient is ready.
      for (size_t i = 0; i < cpu_params.size(); ++i) {
        auto cpu_optimizer = cpu_optimizer_creator({cpu_params[i]});
        cpu_optimizer_metadata_.emplace(
            cpu_optimizer.get(), CpuOptimizerMetadata{{cpu_param_indices[i]}});
        cpu_optimizers_.push_back(cpu_optimizer);
        RegisterCpuParamCopyBackHook(cpu_optimizer);
      }
    } else {
      // NOTE(zbl): If overlap is off, all gradients are managed by one
      // optimizer.
      auto cpu_optimizer = cpu_optimizer_creator(cpu_params);
      cpu_optimizer_metadata_.emplace(
          cpu_optimizer.get(),
          CpuOptimizerMetadata{std::move(cpu_param_indices)});
      cpu_optimizers_.push_back(cpu_optimizer);
      RegisterCpuParamCopyBackHook(cpu_optimizer);
    }
  }

  LOG(INFO) << "HybridDeviceOptimizer: offloaded " << offloaded_num_elements
            << " / " << device_num_elements << " device elements ("
            << offloaded_param_bytes / (1024.0 * 1024.0) << " MiB of params).";
}

void HybridDeviceOptimizer::ZeroGrad(bool set_to_none) {
  Optimizer::ZeroGrad(set_to_none);
  if (gpu_optimizer_) {
    gpu_optimizer_->ZeroGrad(set_to_none);
  }
  for (const auto &cpu_optimizer : cpu_optimizers_) {
    cpu_optimizer->ZeroGrad(set_to_none);
  }
}

void HybridDeviceOptimizer::StepImpl() {
  // Set grads for device FP32 master parameters on the current stream.
  for (const auto original_index : gpu_fp32_param_indices_) {
    const auto &original = params_[original_index];
    const auto &inner_param = param_mappings_[original_index].inner_param;
    const auto &grad = original->grad();
    if (!grad || grad->Dtype() == inner_param->Dtype()) {
      inner_param->set_grad(grad);
    } else {
      inner_param->set_grad(
          std::make_shared<Tensor>(grad->To(inner_param->Dtype())));
    }
  }

  // Copy each CPU optimizer's grads on the D2H stream, then record one
  // completion event for that optimizer.
  if (!offloaded_params_.empty()) {
    core::DeviceGuard guard(device_);
    auto *impl = core::GetDeviceGuardImpl(device_.type());
    auto *current_stream = impl->GetStream(device_);
    impl->EventRecord(d2h_ready_event_, current_stream);
    impl->StreamWaitEvent(d2h_stream_, d2h_ready_event_, 0);
    core::StreamGuard stream_guard(device_, d2h_stream_);
    for (const auto &cpu_optimizer : cpu_optimizers_) {
      auto &metadata = cpu_optimizer_metadata_.at(cpu_optimizer.get());
      for (const auto original_index : metadata.original_indices) {
        auto &entry =
            offloaded_params_[param_mappings_[original_index].offloaded_index];
        const auto &grad = params_[original_index]->grad();
        const auto &cpu_param = param_mappings_[original_index].inner_param;
        if (!grad) {
          cpu_param->set_grad(nullptr);
        } else if (grad->Dtype() == cpu_param->Dtype()) {
          entry.cpu_grad->CopyFrom(grad);
          cpu_param->set_grad(entry.cpu_grad);
        } else {
          CHECK(entry.cpu_grad_staging)
              << "Missing CPU grad staging buffer for non-fp32 grad.";
          entry.cpu_grad_staging->CopyFrom(grad);
        }
      }
      impl->EventRecord(metadata.d2h_done, d2h_stream_);
    }
  }

  // Step the device optimizer while CPU gradients are being copied.
  if (gpu_optimizer_) {
    gpu_optimizer_->Step();
  }

  // Wait for each CPU optimizer's gradients, finish host dtype conversion, and
  // step it.
  for (const auto &cpu_optimizer : cpu_optimizers_) {
    const auto &metadata = cpu_optimizer_metadata_.at(cpu_optimizer.get());
    const auto status = core::GetDeviceGuardImpl(device_.type())
                            ->EventSynchronize(metadata.d2h_done);
    CHECK(status == core::RuntimeStatus::kSuccess)
        << "HybridDeviceOptimizer device-to-host event sync failed: "
        << core::RuntimeStatusToString(status);
    for (const auto original_index : metadata.original_indices) {
      auto &entry =
          offloaded_params_[param_mappings_[original_index].offloaded_index];
      const auto &grad = params_[original_index]->grad();
      const auto &cpu_param = param_mappings_[original_index].inner_param;
      if (grad && grad->Dtype() != cpu_param->Dtype()) {
        auto typed_grad = std::make_shared<Tensor>(
            entry.cpu_grad_staging->To(cpu_param->Dtype()));
        entry.cpu_grad->CopyFrom(typed_grad);
        cpu_param->set_grad(entry.cpu_grad);
      }
    }
    cpu_optimizer->Step();
  }
}

std::unordered_map<std::string, std::shared_ptr<Tensor>>
HybridDeviceOptimizer::StateDict() const {
  std::unordered_map<std::string, std::shared_ptr<Tensor>> state;

  auto layout = std::make_shared<Tensor>(
      std::vector<int64_t>{static_cast<int64_t>(params_.size() + 1)},
      DataType::kINT64, Device());
  auto *layout_data = static_cast<int64_t *>(layout->DataPtr());
  layout_data[0] = overlap_cpu_optimizer_d2h_h2d_ ? 1 : 0;
  for (size_t original_index = 0; original_index < params_.size();
       ++original_index) {
    layout_data[original_index + 1] =
        param_mappings_[original_index].offloaded_index ==
                static_cast<size_t>(-1)
            ? 0
            : 1;
    if (param_mappings_[original_index].uses_master_param) {
      state.emplace(std::format("hybrid.master_param.{}", original_index),
                    param_mappings_[original_index].inner_param);
    }
  }
  state.emplace("hybrid.layout", std::move(layout));

  if (gpu_optimizer_) {
    for (const auto &[key, tensor] : gpu_optimizer_->StateDict()) {
      state.emplace(PrefixKey("hybrid.device", key), tensor);
    }
  }
  for (size_t i = 0; i < cpu_optimizers_.size(); ++i) {
    const auto prefix = std::format("hybrid.cpu.{}", i);
    for (const auto &[key, tensor] : cpu_optimizers_[i]->StateDict()) {
      state.emplace(PrefixKey(prefix, key), tensor);
    }
  }
  return state;
}

void HybridDeviceOptimizer::LoadStateDict(
    const std::unordered_map<std::string, std::shared_ptr<Tensor>>
        &state_dict) {
  CHECK(state_dict.contains("hybrid.layout"))
      << "Missing optimizer state: hybrid.layout";
  const Tensor layout = state_dict.at("hybrid.layout")->To(Device());
  CHECK(layout.Dtype() == DataType::kINT64)
      << "HybridDeviceOptimizer checkpoint layout must be int64.";
  CHECK_EQ(layout.NumElements(), params_.size() + 1)
      << "HybridDeviceOptimizer checkpoint parameter count does not match the "
         "current optimizer.";
  const auto *layout_data = static_cast<const int64_t *>(layout.DataPtr());
  if ((layout_data[0] != 0) != overlap_cpu_optimizer_d2h_h2d_) {
    // TODO(zbl): Remap checkpoint state after Optimizer exposes parameter-keyed
    // state like PyTorch.
    LOG(FATAL) << "HybridDeviceOptimizer cannot load a checkpoint with a "
                  "different CPU optimizer overlap layout.";
  }
  for (size_t original_index = 0; original_index < params_.size();
       ++original_index) {
    const bool offloaded = param_mappings_[original_index].offloaded_index !=
                           static_cast<size_t>(-1);
    if ((layout_data[original_index + 1] != 0) != offloaded) {
      // TODO(zbl): Remap checkpoint state after Optimizer exposes
      // parameter-keyed state like PyTorch.
      LOG(FATAL) << "HybridDeviceOptimizer cannot load a checkpoint with a "
                    "different offload partition.";
    }
  }

  ReloadModelParams();
  for (size_t original_index = 0; original_index < params_.size();
       ++original_index) {
    const auto master_param_key =
        std::format("hybrid.master_param.{}", original_index);
    if (param_mappings_[original_index].uses_master_param) {
      CHECK(state_dict.contains(master_param_key))
          << "Missing optimizer state: " << master_param_key;
      param_mappings_[original_index].inner_param->CopyFrom(
          state_dict.at(master_param_key));
    }
  }

  if (gpu_optimizer_) {
    gpu_optimizer_->LoadStateDict(
        ExtractPrefixedState(state_dict, "hybrid.device"));
  }
  for (size_t i = 0; i < cpu_optimizers_.size(); ++i) {
    cpu_optimizers_[i]->LoadStateDict(
        ExtractPrefixedState(state_dict, std::format("hybrid.cpu.{}", i)));
  }
}

void HybridDeviceOptimizer::ReloadModelParams() {
  // Preserve optimizer state; only refresh CPU copies and device FP32 masters.
  for (size_t original_index = 0; original_index < params_.size();
       ++original_index) {
    const auto &original = params_[original_index];
    const auto &inner_param = param_mappings_[original_index].inner_param;
    if (inner_param.get() == original.get()) {
      continue;
    }
    if (inner_param->GetDevice().IsCPU()) {
      inner_param->CopyFrom(ToCpuInnerParam(original, inner_param->Dtype()));
    } else if (inner_param->Dtype() == original->Dtype()) {
      inner_param->CopyFrom(original);
    } else {
      inner_param->CopyFrom(std::make_shared<Tensor>(
          original->Detach()->To(inner_param->Dtype())));
    }
  }
}

void HybridDeviceOptimizer::set_learning_rate(float lr) {
  Optimizer::set_learning_rate(lr);
  if (gpu_optimizer_) {
    gpu_optimizer_->set_learning_rate(lr);
  }
  for (const auto &cpu_optimizer : cpu_optimizers_) {
    cpu_optimizer->set_learning_rate(lr);
  }
}

float HybridDeviceOptimizer::learning_rate() const {
  if (gpu_optimizer_) {
    return gpu_optimizer_->learning_rate();
  }
  if (!cpu_optimizers_.empty()) {
    return cpu_optimizers_.front()->learning_rate();
  }
  return Optimizer::learning_rate();
}

OptimizerCreator HybridDeviceOptimizer::Create(
    float offload_fraction, OptimizerCreator cpu_optimizer_creator,
    OptimizerCreator gpu_optimizer_creator, bool param_update_in_fp32,
    bool pin_cpu_grads, bool pin_cpu_params,
    bool overlap_cpu_optimizer_d2h_h2d) {
  return [=](const std::vector<std::shared_ptr<Tensor>> &params) {
    return std::make_shared<HybridDeviceOptimizer>(
        params, offload_fraction, cpu_optimizer_creator, gpu_optimizer_creator,
        param_update_in_fp32, pin_cpu_grads, pin_cpu_params,
        overlap_cpu_optimizer_d2h_h2d);
  };
}

}  // namespace optimizers
}  // namespace infini_train
