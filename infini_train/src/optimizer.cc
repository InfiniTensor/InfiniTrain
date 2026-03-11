#include "infini_train/include/optimizer.h"

#include <vector>

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train {
Optimizer::Optimizer(const std::vector<std::shared_ptr<Tensor>> &params, float learning_rate)
    : params_(params), learning_rate_(learning_rate) {}

void Optimizer::ZeroGrad(bool set_to_none) {
    for (auto param : params_) { param->ZeroGrad(set_to_none); }
}

void Optimizer::SetLearningRate(float lr) { learning_rate_ = lr; }

float Optimizer::GetLearningRate() const { return learning_rate_; }

float Optimizer::GetInitialLearningRate() const {
    CHECK(initial_lr_set_) << "Optimizer: initial_learning_rate not set. "
                              "Use with an LRScheduler first.";
    return initial_learning_rate_;
}

void Optimizer::SetInitialLearningRate(float lr) {
    if (!initial_lr_set_) {
        initial_learning_rate_ = lr;
        initial_lr_set_ = true;
    }
}
namespace optimizers {

SGD::SGD(const std::vector<std::shared_ptr<Tensor>> &params, float learning_rate) : Optimizer(params, learning_rate) {}

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
    : Optimizer(params, learning_rate), t_(0), beta1_(beta1), beta2_(beta2), eps_(eps) {

    for (const auto &param : params_) {
        m_.emplace_back(std::make_shared<Tensor>(param->Dims(), param->Dtype(), param->GetDevice()));
        v_.emplace_back(std::make_shared<Tensor>(param->Dims(), param->Dtype(), param->GetDevice()));
        DispatchFunc<INFINI_ALL_TYPES>(
            param->Dtype(),
            [this]<typename T>() {
                m_.back()->Fill<T>(0);
                v_.back()->Fill<T>(0);
            },
            "CUDA Adam");
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
} // namespace optimizers
} // namespace infini_train
