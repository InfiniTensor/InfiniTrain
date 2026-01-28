#include "infini_train/include/optimizer.h"

#include <vector>

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
        device->SetDevice();
        auto kernel = Dispatcher::Instance().GetKernel({device->Type(), "AccumulateGrad"});
        kernel.Call<void>(param->grad(), -learning_rate_, param);
    }
}

Adam::Adam(const std::vector<std::shared_ptr<Tensor>> &params, float learning_rate, float beta1, float beta2, float eps)
    : Optimizer(params), t_(0), learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), eps_(eps) {

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
        device->SetDevice();
        auto kernel = Dispatcher::Instance().GetKernel({device->Type(), "AdamAccumulateGrad"});
        kernel.Call<void>(grad, param, m, v, learning_rate_, beta1_, beta2_, eps_, t_);
    }
}
} // namespace optimizers
} // namespace infini_train
