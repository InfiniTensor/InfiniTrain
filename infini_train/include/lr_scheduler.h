#pragma once

#include <cstdint>
#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace infini_train {

class Optimizer;

using StateValue = std::variant<int64_t, float, double, std::string,
                                std::vector<float>>;
using StateDict = std::unordered_map<std::string, StateValue>;

class LRScheduler {
public:
    explicit LRScheduler(std::shared_ptr<Optimizer> optimizer,
                         int64_t last_step = -1);

    virtual ~LRScheduler() = default;

    LRScheduler(const LRScheduler &) = delete;
    LRScheduler &operator=(const LRScheduler &) = delete;

    virtual void Step();

    float GetLR() const;
    float BaseLR() const;
    int64_t LastStep() const;

    void ResetStep(int64_t step = -1);

    virtual StateDict State() const;
    virtual void LoadState(const StateDict &state);

protected:

    virtual float ComputeLR() const = 0;

    void ApplyLR(float lr);

    std::shared_ptr<Optimizer> optimizer_;
    int64_t last_step_;
    float current_lr_;
    float base_lr_;
};

namespace lr_schedulers {

class ConstantLR : public LRScheduler {
public:
    ConstantLR(std::shared_ptr<Optimizer> optimizer, 
               float factor = 1.0f / 3.0f, 
               int total_iters = 5, 
               int64_t last_step = -1);
    ~ConstantLR() override = default;

protected:
    float ComputeLR() const override;

private:
    const float factor_;
    const int64_t total_iters_;
};

class StepLR : public LRScheduler {
public:
    StepLR(std::shared_ptr<Optimizer> optimizer, 
           int64_t step_size, 
           float gamma = 0.1f, 
           int64_t last_step = -1);
    ~StepLR() override = default;

protected:
    float ComputeLR() const override;

private:
    const int64_t step_size_;
    const float gamma_;
};

class LinearWarmupLR : public LRScheduler {
public: 
    LinearWarmupLR(std::shared_ptr<Optimizer> optimizer,
                   int64_t warmup_steps, 
                   float start_factor = 0.0f, 
                   int64_t last_step = -1);
    ~LinearWarmupLR() override = default;

protected:
    float ComputeLR() const override;

private:
    const int64_t warmup_steps_;
    const float start_factor_;
};

class LambdaLR : public LRScheduler {
public:
    using LambdaFunc = std::function<float(int64_t)>;

    LambdaLR(std::shared_ptr<Optimizer> optimizer, 
                LambdaFunc lr_lambda, 
                int64_t last_step = -1);
    ~LambdaLR() override = default;

protected:
    float ComputeLR() const override;

private:
    const LambdaFunc lr_lambda_;
};


class SequentialLR : public LRScheduler {
public:
    SequentialLR(std::shared_ptr<Optimizer> optimizer, 
                 std::vector<std::shared_ptr<LRScheduler>> schedulers,
                 std::vector<int64_t> milestones, 
                 int64_t last_step = -1);
    ~SequentialLR() override = default;

    void Step() override;

    StateDict State() const override;
    void LoadState(const StateDict &state) override;
    
protected:
    float ComputeLR() const override { return 0.0f; }

private:
    std::vector<std::shared_ptr<LRScheduler>> schedulers_;
    std::vector<int64_t> milestones_;
};

}  // namespace lr_schedulers
}  // namespace infini_train