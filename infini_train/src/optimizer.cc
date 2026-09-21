#include "infini_train/include/optimizer.h"

#include <unordered_map>
#include <vector>

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/sparse_row_grad.h"
#include "infini_train/include/tensor.h"

namespace infini_train {
thread_local std::unordered_map<const Tensor *, std::shared_ptr<Tensor>> g_shadow_registry;

// Free function for autocast.h to query shadow weights (decoupled from the concrete Optimizer type).
// Returns the shadow on a hit; nullptr on a miss (activations, or shadow disabled), letting autocast fall back to Cast.
std::shared_ptr<Tensor> GetShadow(const Tensor *param) {
    auto it = g_shadow_registry.find(param);
    return it != g_shadow_registry.end() ? it->second : nullptr;
}

Optimizer::Optimizer(const std::vector<std::shared_ptr<Tensor>> &params, float learning_rate)
    : params_(params), learning_rate_(learning_rate) {}

Optimizer::Optimizer(const NamedParameterList &named_params, float learning_rate) : learning_rate_(learning_rate) {
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
        auto *sparse_state = GetSparseRowGradState(param.get());
        if (sparse_state) {
            auto device = param->GetDevice();
            core::DeviceGuard guard(device);
            // Zero only the rows made dirty since the last clear: the persistent buffer is zero
            // everywhere else by construction, so no full-size memset is needed. This also bumps
            // the claim generation, re-arming the dedup for the next accumulation cycle.
            ClearSparseRowGradRows(sparse_state);
            // Non-poisoned: the cleared buffer *is* the accumulator, so nothing else has to be
            // reset. Poisoned: the live storage is some dense foreign buffer and needs the dense
            // reset path (grad_.reset() or a full fill).
            if (set_to_none || sparse_state->poisoned) {
                param->ZeroGrad(set_to_none);
            }
            continue;
        }
        param->ZeroGrad(set_to_none);
    }
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

SGD::SGD(const std::vector<std::shared_ptr<Tensor>> &params, float learning_rate) : Optimizer(params, learning_rate) {}

SGD::SGD(const NamedParameterList &named_params, float learning_rate) : Optimizer(named_params, learning_rate) {}

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
    return [learning_rate](const std::vector<std::shared_ptr<Tensor>> &params) {
        return std::make_shared<SGD>(params, learning_rate);
    };
}

OptimizerCreatorNamed SGD::CreateNamed(float learning_rate) {
    return [learning_rate](const NamedParameterList &named_params) {
        return std::make_shared<SGD>(named_params, learning_rate);
    };
}

Adam::Adam(const std::vector<std::shared_ptr<Tensor>> &params, float learning_rate, float beta1, float beta2, float eps)
    : Optimizer(params, learning_rate), t_(0), beta1_(beta1), beta2_(beta2), eps_(eps) {

    for (const auto &param : params_) {
        m_.emplace_back(std::make_shared<Tensor>(param->Dims(), param->Dtype(), param->GetDevice()));
        v_.emplace_back(std::make_shared<Tensor>(param->Dims(), param->Dtype(), param->GetDevice()));
        m_.back()->Fill(0.0);
        v_.back()->Fill(0.0);
    }
}

Adam::Adam(const NamedParameterList &named_params, float learning_rate, float beta1, float beta2, float eps)
    : Optimizer(named_params, learning_rate), t_(0), beta1_(beta1), beta2_(beta2), eps_(eps) {
    for (const auto &[name, param] : named_params) {
        m_.emplace_back(std::make_shared<Tensor>(param->Dims(), param->Dtype(), param->GetDevice()));
        v_.emplace_back(std::make_shared<Tensor>(param->Dims(), param->Dtype(), param->GetDevice()));
        m_.back()->Fill(0.0);
        v_.back()->Fill(0.0);
    }
}

void Adam::EnableShadowWeights(DataType shadow_dtype) {
    shadow_enable_ = true;
    shadow_dtype_ = shadow_dtype;
    shadow_weights_.clear();
    shadow_weights_.reserve(params_.size());
    // init shadow weights form param
    for (auto &param : params_) {
        auto shadow_weight = std::make_shared<Tensor>(param->Dims(), shadow_dtype_, param->GetDevice());
        auto casted = param->To(shadow_dtype);
        shadow_weight->CopyFrom(casted);
        shadow_weights_.push_back(shadow_weight);
        g_shadow_registry[param.get()] = shadow_weight;
    }
    LOG(INFO) << "Enable shadow weights for Adam optimizer, shadow dtype: " << static_cast<int>(shadow_dtype_);
}

void Adam::DisableShadowWeights() {
    if (shadow_enable_ == false) {
        return;
    }
    shadow_enable_ = false;
    for (auto &param : params_) { g_shadow_registry.erase(param.get()); }
    shadow_weights_.clear();
    LOG(INFO) << "Disable shadow weights for Adam optimizer";
}

void Adam::RefreshShadowWeights() {
    if (!shadow_enable_) {
        return;
    }
    // The param was updated in place by the checkpoint (LoadStateDict uses CopyFrom, so the pointer is unchanged):
    // just re-cast the current FP32 master weights into the shadows; the registry mapping needs no changes.
    for (size_t i = 0; i < params_.size(); ++i) {
        auto casted = params_[i]->To(shadow_dtype_);
        shadow_weights_[i]->CopyFrom(casted);
    }
    LOG(INFO) << "Refreshed " << shadow_weights_.size() << " shadow weights from current params";
}

std::shared_ptr<Tensor> Adam::GetShadow(const Tensor *param) const {
    auto it = g_shadow_registry.find(param);
    if (it != g_shadow_registry.end()) {
        return it->second;
    } else {
        LOG(WARNING) << "Shadow weight not found for the given parameter.";
        return nullptr;
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
        auto *sparse_state = GetSparseRowGradState(param.get());
        if (sparse_state && !sparse_state->poisoned) {
            // Sparse weight (embedding): only the rows hit since the last clear exist in the
            // persistent buffer, so update just those rows of param/m/v/shadow.
            if (shadow_enable_) {
                auto shadow_weight = shadow_weights_[i];
                auto kernel = Dispatcher::Instance().GetKernel({device.type(), "AdamSparseRowsShadow"});
                kernel.Call<void>(grad, param, shadow_weight, m, v, sparse_state->row_list, sparse_state->count,
                                  learning_rate_, beta1_, beta2_, eps_, t_);
            } else {
                auto kernel = Dispatcher::Instance().GetKernel({device.type(), "AdamSparseRows"});
                kernel.Call<void>(grad, param, m, v, sparse_state->row_list, sparse_state->count, learning_rate_,
                                  beta1_, beta2_, eps_, t_);
            }
            continue;
        }
        if (shadow_enable_) {
            auto shadow_weight = shadow_weights_[i];
            auto kernel = Dispatcher::Instance().GetKernel({device.type(), "AdamAccumulateGradShadow"});
            kernel.Call<void>(grad, param, shadow_weight, m, v, learning_rate_, beta1_, beta2_, eps_, t_);
        } else {
            auto kernel = Dispatcher::Instance().GetKernel({device.type(), "AdamAccumulateGrad"});
            kernel.Call<void>(grad, param, m, v, learning_rate_, beta1_, beta2_, eps_, t_);
        }
    }
}

OptimizerCreator Adam::Create(float learning_rate, float beta1, float beta2, float eps) {
    return [=](const std::vector<std::shared_ptr<Tensor>> &params) {
        return std::make_shared<Adam>(params, learning_rate, beta1, beta2, eps);
    };
}

OptimizerCreatorNamed Adam::CreateNamed(float learning_rate, float beta1, float beta2, float eps) {
    return [=](const NamedParameterList &named_params) {
        return std::make_shared<Adam>(named_params, learning_rate, beta1, beta2, eps);
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
