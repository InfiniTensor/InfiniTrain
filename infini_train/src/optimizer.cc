#include "infini_train/include/optimizer.h"

#include <format>
#include <vector>

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train {
Optimizer::Optimizer(const std::vector<std::shared_ptr<Tensor>> &params) : params_(params) {}

void Optimizer::ZeroGrad(bool set_to_none) {
    for (auto param : params_) { param->ZeroGrad(set_to_none); }
}

namespace optimizers {

SGD::SGD(const std::vector<std::shared_ptr<Tensor>> &params, float learning_rate)
    : Optimizer(params), learning_rate_(learning_rate) {}

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

Adam::Adam(const std::vector<std::shared_ptr<Tensor>> &params, float learning_rate, float beta1, float beta2, float eps)
    : Optimizer(params), t_(0), learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), eps_(eps) {

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

std::unordered_map<std::string, std::shared_ptr<Tensor>> Adam::StateDict() const {
    std::unordered_map<std::string, std::shared_ptr<Tensor>> state;
    for (size_t i = 0; i < m_.size(); ++i) {
        state.emplace(std::format("adam.m.{}", i), m_[i]);
        state.emplace(std::format("adam.v.{}", i), v_[i]);
    }

    auto t_tensor = std::make_shared<Tensor>(std::vector<int64_t>{}, DataType::kINT64, Device());
    *static_cast<int64_t *>(t_tensor->DataPtr()) = t_;
    state.emplace("adam.t", t_tensor);
    return state;
}

void Adam::LoadStateDict(const std::unordered_map<std::string, std::shared_ptr<Tensor>> &state_dict) {
    for (size_t i = 0; i < m_.size(); ++i) {
        const auto m_key = std::format("adam.m.{}", i);
        const auto v_key = std::format("adam.v.{}", i);
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
