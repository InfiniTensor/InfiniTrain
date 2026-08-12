#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace infini_train {
class Tensor;
}
namespace infini_train {
class Optimizer;

using NamedParameter = std::pair<std::string, std::shared_ptr<Tensor>>;
using NamedParameterList = std::vector<NamedParameter>;
using OptimizerCreator = std::function<std::shared_ptr<Optimizer>(const std::vector<std::shared_ptr<Tensor>> &params,
                                                                  const NamedParameterList &named_parameters)>;

class Optimizer {
public:
    explicit Optimizer(const std::vector<std::shared_ptr<Tensor>> &params, float learning_rate = 0.0f,
                       const NamedParameterList &named_parameters = {});

    virtual void ZeroGrad(bool set_to_none = true);

    virtual void Step() = 0;

    virtual std::unordered_map<std::string, std::shared_ptr<Tensor>> StateDict() const { return {}; };

    virtual void LoadStateDict(const std::unordered_map<std::string, std::shared_ptr<Tensor>> &state_dict) {}

    virtual void set_learning_rate(float lr);

    virtual float learning_rate() const;

    float initial_learning_rate() const;

    bool initial_lr_set() const;

    void set_initial_learning_rate(float lr);

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
    SGD(const std::vector<std::shared_ptr<Tensor>> &params, float learning_rate,
        const NamedParameterList &named_parameters = {});

    void Step() override;

    static OptimizerCreator Create(float learning_rate);
};

class Adam : public Optimizer {
public:
    Adam(const std::vector<std::shared_ptr<Tensor>> &params, float learning_rate = 1e-3, float beta1 = 0.9,
         float beta2 = 0.999, float eps = 1e-8, const NamedParameterList &named_parameters = {});

    void Step() override;

    std::unordered_map<std::string, std::shared_ptr<Tensor>> StateDict() const override;

    void LoadStateDict(const std::unordered_map<std::string, std::shared_ptr<Tensor>> &state_dict) override;
    static OptimizerCreator Create(float learning_rate = 1e-3, float beta1 = 0.9, float beta2 = 0.999,
                                   float eps = 1e-8);

private:
    int64_t t_;
    const float beta1_;
    const float beta2_;
    const float eps_;
    std::vector<std::shared_ptr<Tensor>> m_;
    std::vector<std::shared_ptr<Tensor>> v_;
};
} // namespace optimizers
} // namespace infini_train
