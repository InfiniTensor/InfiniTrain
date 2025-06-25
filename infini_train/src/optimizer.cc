#include "infini_train/include/optimizer.h"

#include <vector>

#include "infini_train/include/autograd/accumulate.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train {
Optimizer::Optimizer(const std::vector<std::shared_ptr<Tensor>> &params) : params_(params) {}

void Optimizer::ZeroGrad() {
    for (auto param : params_) { param->ZeroGrad(); }
}

namespace optimizers {

SGD::SGD(const std::vector<std::shared_ptr<Tensor>> &params, float learning_rate)
    : Optimizer(params), learning_rate_(learning_rate) {}

void SGD::Step() {
    for (auto param : params_) {
        auto device = param->GetDevice()->Type();
        auto accumulate_function = std::make_shared<infini_train::autograd::AccumulateGrad>(param, -learning_rate_);
        accumulate_function->BackwardPartial(param->grad(), 0);
    }
}

Adam::Adam(const std::vector<std::shared_ptr<Tensor>> &params, float learning_rate, float beta1, float beta2, float eps)
    : Optimizer(params), t_(0), learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), eps_(eps) {

    for (const auto &param : params_) {
        m_.emplace_back(std::make_shared<Tensor>(param->Dims(), param->Dtype(), param->GetDevice()));
        v_.emplace_back(std::make_shared<Tensor>(param->Dims(), param->Dtype(), param->GetDevice()));
        DispatchFunc<INFINI_ALL_TYPES>(
            param->Dtype(),
            [=]<typename T>() {
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
        auto &m = m_[i];
        auto &v = v_[i];

        auto device = param->GetDevice()->Type();
        auto kernel = Dispatcher::Instance().GetKernel({device, "AdamAccumulateGrad"});
        kernel.Call<void>(grad, param, m, v, learning_rate_, beta1_, beta2_, eps_, t_);
    }
}
} // namespace optimizers
} // namespace infini_train
