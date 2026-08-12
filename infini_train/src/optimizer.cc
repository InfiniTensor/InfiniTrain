#include "infini_train/include/optimizer.h"

#include <unordered_map>
#include <vector>

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train {
Optimizer::Optimizer(const std::vector<std::shared_ptr<Tensor>> &params, float learning_rate,
                     const NamedParameterList &named_parameters)
    : params_(params), learning_rate_(learning_rate) {
    if (named_parameters.empty()) {
        return;
    }

    std::unordered_map<const Tensor *, std::string> parameter_name_by_tensor;
    parameter_name_by_tensor.reserve(named_parameters.size());
    for (const auto &[name, parameter] : named_parameters) {
        CHECK(parameter);
        parameter_name_by_tensor.emplace(parameter.get(), name);
    }

    parameter_names_.reserve(params_.size());
    for (const auto &parameter : params_) {
        const auto it = parameter_name_by_tensor.find(parameter.get());
        CHECK(it != parameter_name_by_tensor.end()) << "Optimizer parameter is not registered in the model";
        parameter_names_.push_back(it->second);
    }
}

void Optimizer::ZeroGrad(bool set_to_none) {
    for (auto param : params_) { param->ZeroGrad(set_to_none); }
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
    CHECK(!initial_lr_set_) << "Optimizer: initial_learning_rate has already been set.";
    initial_learning_rate_ = lr;
    initial_lr_set_ = true;
}

namespace optimizers {

SGD::SGD(const std::vector<std::shared_ptr<Tensor>> &params, float learning_rate,
         const NamedParameterList &named_parameters)
    : Optimizer(params, learning_rate, named_parameters) {}

void SGD::Step() {
    for (auto param : params_) {
        if (!param->grad()) {
            LOG(INFO) << "Skipping param with null grad.";
            continue;
        }
        auto device = param->GetDevice();
        core::DeviceGuard guard(device);
        auto kernel = Dispatcher::Instance().GetKernel({device.type(), "AccumulateGrad"});
        kernel.Call<void>(param->grad(), -learning_rate_, param);
    }
}

OptimizerCreator SGD::Create(float learning_rate) {
    return [learning_rate](const std::vector<std::shared_ptr<Tensor>> &params,
                           const NamedParameterList &named_parameters) {
        return std::make_shared<SGD>(params, learning_rate, named_parameters);
    };
}

Adam::Adam(const std::vector<std::shared_ptr<Tensor>> &params, float learning_rate, float beta1, float beta2, float eps,
           const NamedParameterList &named_parameters)
    : Optimizer(params, learning_rate, named_parameters), t_(0), beta1_(beta1), beta2_(beta2), eps_(eps) {

    for (const auto &param : params_) {
        m_.emplace_back(std::make_shared<Tensor>(param->Dims(), param->Dtype(), param->GetDevice()));
        v_.emplace_back(std::make_shared<Tensor>(param->Dims(), param->Dtype(), param->GetDevice()));
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
        auto kernel = Dispatcher::Instance().GetKernel({device.type(), "AdamAccumulateGrad"});
        kernel.Call<void>(grad, param, m, v, learning_rate_, beta1_, beta2_, eps_, t_);
    }
}

OptimizerCreator Adam::Create(float learning_rate, float beta1, float beta2, float eps) {
    return [=](const std::vector<std::shared_ptr<Tensor>> &params, const NamedParameterList &named_parameters) {
        return std::make_shared<Adam>(params, learning_rate, beta1, beta2, eps, named_parameters);
    };
}

std::unordered_map<std::string, std::shared_ptr<Tensor>> Adam::StateDict() const {
    std::unordered_map<std::string, std::shared_ptr<Tensor>> state;
    for (size_t i = 0; i < m_.size(); ++i) {
        const auto suffix = parameter_names_.empty() ? std::to_string(i) : parameter_names_[i];
        state.emplace("adam.m." + suffix, m_[i]);
        state.emplace("adam.v." + suffix, v_[i]);
    }

    auto t_tensor = std::make_shared<Tensor>(std::vector<int64_t>{}, DataType::kINT64, Device());
    *static_cast<int64_t *>(t_tensor->DataPtr()) = t_;
    state.emplace("adam.t", t_tensor);
    return state;
}

void Adam::LoadStateDict(const std::unordered_map<std::string, std::shared_ptr<Tensor>> &state_dict) {
    for (size_t i = 0; i < m_.size(); ++i) {
        const auto suffix = parameter_names_.empty() ? std::to_string(i) : parameter_names_[i];
        const auto m_key = "adam.m." + suffix;
        const auto v_key = "adam.v." + suffix;
        CHECK(state_dict.contains(m_key)) << "Missing optimizer state: " << m_key;
        CHECK(state_dict.contains(v_key)) << "Missing optimizer state: " << v_key;
        m_[i]->CopyFrom(state_dict.at(m_key));
        v_[i]->CopyFrom(state_dict.at(v_key));
    }

    CHECK(state_dict.contains("adam.t")) << "Missing optimizer state: adam.t";
    const Tensor t_cpu = state_dict.at("adam.t")->To(Device());
    t_ = *static_cast<const int64_t *>(t_cpu.DataPtr());
}
} // namespace optimizers
} // namespace infini_train
