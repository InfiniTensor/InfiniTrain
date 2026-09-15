#include "infini_train/include/optimizer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/process_group.h"
#include "infini_train/include/nn/parallel/utils.h"
#include "infini_train/include/tensor.h"

namespace infini_train {
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
    for (auto param : params_) { param->ZeroGrad(set_to_none); }
}

namespace {
struct GradientNormStats {
    double sum = 0.0;
    double max_abs = 0.0;
    double min_abs = std::numeric_limits<double>::infinity();
    uint64_t nonzero = 0;
    bool has_gradient = false;
    bool finite = true;
};

GradientNormStats ComputeGradientNormStats(const std::vector<std::shared_ptr<Tensor>> &parameters, float norm_type) {
    GradientNormStats stats;
    std::unordered_set<const Tensor *> seen;
    for (const auto &parameter : parameters) {
        if (!parameter || !seen.insert(parameter.get()).second || !parameter->grad()) {
            continue;
        }
        const auto &gradient = parameter->grad();
        if (gradient->NumElements() == 0) {
            continue;
        }
        stats.has_gradient = true;
        std::shared_ptr<Tensor> host;
        if (gradient->GetDevice().IsCPU()) {
            host = std::make_shared<Tensor>(*gradient, 0, gradient->Dims());
        } else {
            host = std::make_shared<Tensor>(gradient->To(Device()));
        }
        if (gradient->GetDevice().IsCUDA()) {
            core::GetDeviceGuardImpl(gradient->GetDevice().type())->SynchronizeDevice(gradient->GetDevice());
        }
        std::shared_ptr<Tensor> fp32;
        if (host->Dtype() == DataType::kFLOAT32) {
            fp32 = host;
        } else {
            fp32 = std::make_shared<Tensor>(host->To(DataType::kFLOAT32));
        }
        const float *data = static_cast<const float *>(fp32->DataPtr());
        bool tensor_has_nonzero = false;
        for (size_t i = 0; i < fp32->NumElements(); ++i) {
            const double value = static_cast<double>(data[i]);
            const double abs_value = std::abs(value);
            if (!std::isfinite(value)) {
                stats.finite = false;
                if (norm_type == -std::numeric_limits<float>::infinity()) {
                    stats.min_abs = std::numeric_limits<double>::quiet_NaN();
                } else if (std::isinf(norm_type)) {
                    stats.max_abs = std::numeric_limits<double>::quiet_NaN();
                }
            }
            if (norm_type == -std::numeric_limits<float>::infinity()) {
                stats.min_abs = std::min(stats.min_abs, abs_value);
            } else if (std::isinf(norm_type)) {
                stats.max_abs = std::max(stats.max_abs, abs_value);
            } else if (norm_type == 0.0f) {
                if (abs_value != 0.0) {
                    tensor_has_nonzero = true;
                }
            } else {
                stats.sum += std::pow(abs_value, static_cast<double>(norm_type));
            }
        }
        if (norm_type == 0.0f && tensor_has_nonzero) {
            ++stats.nonzero;
        }
    }
    return stats;
}

std::shared_ptr<Tensor> MakeScalar(float value) {
    auto result = std::make_shared<Tensor>(std::vector<int64_t>{}, DataType::kFLOAT32, Device());
    *static_cast<float *>(result->DataPtr()) = value;
    return result;
}

void ScaleGradientInplace(const std::shared_ptr<Tensor> &gradient, float scale) {
    if (!gradient || scale == 1.0f) {
        return;
    }
    auto device = gradient->GetDevice();
    core::DeviceGuard guard(device);
    auto kernel = Dispatcher::Instance().GetKernel({device.type(), "ScaleInplace"});
    kernel.Call<void>(gradient, scale);
}
} // namespace

void Optimizer::SetClipGradNormConfig(float max_norm, float norm_type, bool error_if_nonfinite,
                                      std::optional<bool> foreach) {
    clip_grad_norm_config_ = ClipGradNormConfig{max_norm, norm_type, error_if_nonfinite, foreach};
}

std::shared_ptr<Tensor> Optimizer::ClipGradNormConfigured() {
    if (!clip_grad_norm_config_) {
        return nullptr;
    }
    const auto &config = *clip_grad_norm_config_;
    return ClipGradNorm_(params_, config.max_norm, config.norm_type, config.error_if_nonfinite, config.foreach);
}

void Optimizer::ScaleGradients_(const std::vector<std::shared_ptr<Tensor>> &parameters, float scale, bool foreach) {
    std::unordered_set<const Tensor *> seen;
    using BatchKey = std::tuple<int, int, int>;
    std::map<BatchKey, std::vector<std::shared_ptr<Tensor>>> batches;
    for (const auto &parameter : parameters) {
        if (!parameter || !seen.insert(parameter.get()).second || !parameter->grad()) {
            continue;
        }
        if (foreach) {
            const auto device = parameter->grad()->GetDevice();
            batches[{static_cast<int>(device.type()), static_cast<int>(device.index()),
                     static_cast<int>(parameter->grad()->Dtype())}]
                .push_back(parameter->grad());
        } else {
            ScaleGradientInplace(parameter->grad(), scale);
        }
    }
    if (foreach) {
        for (const auto &[_, batch] : batches) {
            if (batch.empty()) {
                continue;
            }
            core::DeviceGuard guard(batch.front()->GetDevice());
            auto kernel = Dispatcher::Instance().GetKernel({batch.front()->GetDevice().type(), "ScaleInplaceMulti"});
            kernel.Call<void>(batch, scale);
        }
    }
}

std::shared_ptr<Tensor> Optimizer::ClipGradNorm_(const std::vector<std::shared_ptr<Tensor>> &parameters, float max_norm,
                                                 float norm_type, bool error_if_nonfinite,
                                                 std::optional<bool> foreach) {
    CHECK_GE(max_norm, 0.0f) << "max_norm must be non-negative.";
    CHECK(std::isfinite(norm_type) || std::isinf(norm_type))
        << "norm_type must be finite, +inf, or -inf.";

    const auto stats = ComputeGradientNormStats(parameters, norm_type);
    double total_norm = 0.0;
    if (stats.has_gradient) {
        if (norm_type == 0.0f) {
            total_norm = static_cast<double>(stats.nonzero);
        } else if (norm_type == -std::numeric_limits<float>::infinity()) {
            total_norm = stats.min_abs;
        } else if (norm_type == std::numeric_limits<float>::infinity()) {
            total_norm = stats.max_abs;
        } else {
            total_norm = std::pow(stats.sum, 1.0 / norm_type);
        }
    }

    // Pipeline stages own disjoint parameter sets.  Reduce their statistics so
    // every stage uses one global clipping coefficient.
    if (stats.has_gradient && infini_train::nn::parallel::global::GetPipelineParallelSize() > 1) {
        const auto device = [&]() {
            for (const auto &parameter : parameters) {
                if (parameter && parameter->grad()) {
                    return parameter->grad()->GetDevice();
                }
            }
            return Device();
        }();
        auto reduced = std::make_shared<Tensor>(std::vector<int64_t>{}, DataType::kFLOAT32, device);
        const double local_stat = norm_type == 0.0f ? static_cast<double>(stats.nonzero)
                                 : norm_type == -std::numeric_limits<float>::infinity() ? stats.min_abs
                                 : norm_type == std::numeric_limits<float>::infinity() ? stats.max_abs
                                                                                         : stats.sum;
        reduced->Fill(static_cast<float>(local_stat));
        const auto *group = nn::parallel::ProcessGroupFactory::Instance(device.type())
                                ->Get(nn::parallel::GetPipelineParallelProcessGroupName(device.Rank().GlobalRank()));
        CHECK(group) << "Pipeline process group is not initialized.";
        group->AllReduce(reduced,
                         norm_type == -std::numeric_limits<float>::infinity()
                             ? nn::parallel::function::ReduceOpType::kMin
                             : norm_type == std::numeric_limits<float>::infinity()
                                   ? nn::parallel::function::ReduceOpType::kMax
                                   : nn::parallel::function::ReduceOpType::kSum,
                         false);
        Tensor reduced_cpu
            = reduced->GetDevice().IsCPU() ? Tensor(*reduced, 0, reduced->Dims()) : reduced->To(Device());
        const float value = *static_cast<const float *>(reduced_cpu.DataPtr());
        total_norm = norm_type == 0.0f ? static_cast<double>(value)
                                      : norm_type == -std::numeric_limits<float>::infinity()
                                            ? static_cast<double>(value)
                                            : norm_type == std::numeric_limits<float>::infinity()
                                                  ? static_cast<double>(value)
                                                  : std::pow(static_cast<double>(value), 1.0 / norm_type);
    }
    if (error_if_nonfinite && (!stats.finite || !std::isfinite(total_norm))) {
        LOG(FATAL) << "The total gradient norm is non-finite.";
    }

    const double coefficient
        = std::isinf(max_norm) ? 1.0 : std::min(static_cast<double>(max_norm) / (total_norm + 1e-6), 1.0);
    if (stats.has_gradient) {
        ScaleGradients_(parameters, static_cast<float>(coefficient), foreach.value_or(false));
    }
    return MakeScalar(static_cast<float>(total_norm));
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
