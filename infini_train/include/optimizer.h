#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "infini_train/include/datatype.h"

namespace infini_train {
class Tensor;
}
namespace infini_train {
class Optimizer;

using NamedParameter = std::pair<std::string, std::shared_ptr<Tensor>>;
using NamedParameterList = std::vector<NamedParameter>;
using OptimizerCreator = std::function<std::shared_ptr<Optimizer>(const std::vector<std::shared_ptr<Tensor>> &params)>;
using OptimizerCreatorNamed = std::function<std::shared_ptr<Optimizer>(const NamedParameterList &named_params)>;

class Optimizer {
public:
    explicit Optimizer(const std::vector<std::shared_ptr<Tensor>> &params, float learning_rate);

    Optimizer(const NamedParameterList &named_params, float learning_rate);

    virtual ~Optimizer() = default;

    virtual void ZeroGrad(bool set_to_none = true);

    virtual void Step() = 0;

    virtual std::unordered_map<std::string, std::shared_ptr<Tensor>> StateDict() const { return {}; };

    virtual void LoadStateDict(const std::unordered_map<std::string, std::shared_ptr<Tensor>> &state_dict) {}

    virtual void set_learning_rate(float lr);

    virtual float learning_rate() const;

    float initial_learning_rate() const;

    bool initial_lr_set() const;

    void set_initial_learning_rate(float lr);

    // ========== Shadow weights API (default no-op; Adam overrides) ==========
    // 启用影子权重：为每个 param 创建低精度副本并注册到全局表，供 autocast 直接取用，
    // 跳过 forward 中重复的 CastKernel。默认 no-op，仅 Adam 实现。
    virtual void EnableShadowWeights(DataType shadow_dtype = DataType::kBFLOAT16) {}
    virtual void DisableShadowWeights() {}
    // 从当前 FP32 主权重重新同步所有影子权重（checkpoint 恢复后调用）。
    virtual void RefreshShadowWeights() {}
    virtual bool ShadowWeightsEnabled() const { return false; }

protected:
    std::vector<std::shared_ptr<Tensor>> params_;
    std::vector<std::string> parameter_names_;
    float learning_rate_ = 0.0f;
    float initial_learning_rate_ = 0.0f;
    bool initial_lr_set_ = false;
};

namespace optimizers {
class SGD : public Optimizer {
public:
    SGD(const std::vector<std::shared_ptr<Tensor>> &params, float learning_rate);
    SGD(const NamedParameterList &named_params, float learning_rate);

    void Step() override;

    static OptimizerCreator Create(float learning_rate);
    static OptimizerCreatorNamed CreateNamed(float learning_rate);
};

class Adam : public Optimizer {
public:
    Adam(const std::vector<std::shared_ptr<Tensor>> &params, float learning_rate = 1e-3, float beta1 = 0.9,
         float beta2 = 0.999, float eps = 1e-8);
    Adam(const NamedParameterList &named_params, float learning_rate = 1e-3, float beta1 = 0.9, float beta2 = 0.999,
         float eps = 1e-8);

    void Step() override;

    std::unordered_map<std::string, std::shared_ptr<Tensor>> StateDict() const override;

    void LoadStateDict(const std::unordered_map<std::string, std::shared_ptr<Tensor>> &state_dict) override;
    static OptimizerCreator Create(float learning_rate = 1e-3, float beta1 = 0.9, float beta2 = 0.999,
                                   float eps = 1e-8);
    static OptimizerCreatorNamed CreateNamed(float learning_rate = 1e-3, float beta1 = 0.9, float beta2 = 0.999,
                                             float eps = 1e-8);
    
    // add shadow weights API                                         
    void EnableShadowWeights(DataType shadow_dtype) override;

    void DisableShadowWeights() override;

    void RefreshShadowWeights() override;

    bool ShadowWeightsEnabled() const override { return shadow_enable_; }
    
    std::shared_ptr<Tensor> GetShadow(const Tensor* param) const;

    ~Adam() override {
        DisableShadowWeights();
    }

private:
    int64_t t_;
    const float beta1_;
    const float beta2_;
    const float eps_;
    std::vector<std::shared_ptr<Tensor>> m_;
    std::vector<std::shared_ptr<Tensor>> v_;
    // add shadow weights variable
    bool shadow_enable_ = false;
    DataType shadow_dtype_ = DataType::kBFLOAT16;
    std::vector<std::shared_ptr<Tensor>> shadow_weights_;
};
} // namespace optimizers
} // namespace infini_train
