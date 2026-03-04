#pragma once

#include <cstdint>
#include <cmath>
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

    void Step();

    float GetLR() const;

    int64_t LastStep() const;

    virtual StateDict State() const;

    virtual void LoadState(const StateDict &state);

protected:
    virtual float ComputeLR() = 0;

    std::shared_ptr<Optimizer> optimizer_;
    int64_t last_step_;
    float current_lr_;
    float base_lr_;
};

namespace lr_schedulers {
class ConstantLR : public LRScheduler {
public:
    ConstantLR(std::shared_ptr<Optimizer> optimizer, float factor = 1.0f / 3.0f, int total_iters = 5, 
                int64_t last_step = -1);

    ~ConstantLR() override = default;

protected:
    float ComputeLR() override ;

private:
    const float factor_;
    const int64_t total_iters_;
};

class StepLR : public LRScheduler {
public:
    StepLR(std::shared_ptr<Optimizer> optimizer, int64_t step_size, float gamma = 0.1f, int64_t last_step = -1);
    ~StepLR() override = default;

protected:
    float ComputeLR() override;
private:
    const int64_t step_size_;
    const float gamma_;
};

class LinearWarmupLR : public LRScheduler {
public: 
    LinearWarmupLR(std::shared_ptr<Optimizer> optimizer, int64_t warmup_steps, float start_factor = 0.0f, int64_t last_step = -1);
    ~LinearWarmupLR() override = default;

protected:
    float ComputeLR() override ;

private:
    const int64_t warmup_steps_;
    const float start_factor_;

};

}  // namespace lr_schedulers
}  // namespace infini_train