#include "infini_train/include/optimizer.h"

#include <stddef.h>

#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/core/runtime/device_guard.h"
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

void Optimizer::set_learning_rate(float lr) { learning_rate_ = lr; }

float Optimizer::learning_rate() const { return learning_rate_; }

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

void SGD::Step() {
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
    steps_.push_back(0);
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
    steps_.push_back(0);
    m_.back()->Fill(0.0);
    v_.back()->Fill(0.0);
  }
}

void Adam::Step() {
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
                      ++steps_[i]);
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
    auto step = std::make_shared<Tensor>(std::vector<int64_t>{},
                                         DataType::kINT64, Device());
    *static_cast<int64_t *>(step->DataPtr()) = steps_[i];
    state.emplace(std::string(kAdamParameterStepPrefix) + suffix, step);
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
  auto read_step = [&](const std::string &key) {
    CHECK(state_dict.contains(key)) << "Missing optimizer state: " << key;
    const auto &value = state_dict.at(key);
    CHECK(value && value->Dtype() == DataType::kINT64 && value->Dims().empty())
        << "Invalid Adam step tensor: " << key;
    const Tensor cpu = value->To(Device());
    core::GetDeviceGuardImpl(value->GetDevice().type())
        ->SynchronizeDevice(value->GetDevice());
    const int64_t count = *static_cast<const int64_t *>(cpu.DataPtr());
    CHECK_GE(count, 0) << "Negative Adam step: " << key;
    return count;
  };
  const int64_t global_step = read_step(std::string(kAdamStepKey));
  bool has_parameter_steps = false;
  for (const auto &[key, value] : state_dict) {
    if (key.starts_with(kAdamParameterStepPrefix)) {
      has_parameter_steps = true;
      break;
    }
  }
  std::vector<int64_t> loaded_steps;
  loaded_steps.reserve(params_.size());
  // Validate all keys/shapes before mutating optimizer state.
  for (size_t i = 0; i < m_.size(); ++i) {
    const auto suffix =
        parameter_names_.empty() ? std::to_string(i) : parameter_names_[i];
    for (const auto &prefix : {"adam.m.", "adam.v."}) {
      const std::string key = std::string(prefix) + suffix;
      CHECK(state_dict.contains(key)) << "Missing optimizer state: " << key;
      const auto &value = state_dict.at(key);
      CHECK(value && value->Dims() == params_[i]->Dims() &&
            value->Dtype() == params_[i]->Dtype())
          << "Invalid Adam moment tensor: " << key;
    }
    loaded_steps.push_back(
        has_parameter_steps
            ? read_step(std::string(kAdamParameterStepPrefix) + suffix)
            : global_step);
  }
  if (!has_parameter_steps && !params_.empty()) {
    LOG(WARNING) << "Loading legacy Adam state without per-parameter steps; "
                    "using `adam.t` for every parameter. "
                    "Past missing-gradient history cannot be reconstructed.";
  }
  for (size_t i = 0; i < m_.size(); ++i) {
    const auto suffix =
        parameter_names_.empty() ? std::to_string(i) : parameter_names_[i];
    m_[i]->CopyFrom(state_dict.at("adam.m." + suffix));
    v_[i]->CopyFrom(state_dict.at("adam.v." + suffix));
    // Keep temporary checkpoint host storage alive until asynchronous copies
    // finish.
    const auto device = params_[i]->GetDevice();
    core::GetDeviceGuardImpl(device.type())->SynchronizeDevice(device);
  }
  t_ = global_step;
  steps_ = std::move(loaded_steps);
}
}  // namespace optimizers
}  // namespace infini_train
