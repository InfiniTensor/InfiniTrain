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
    template<typename T, typename... Args>
    static std::shared_ptr<T> Create(Args&&... args) {
        auto scheduler = std::make_shared<T>(std::forward<Args>(args)...);
        scheduler->InitialStep();
        return scheduler;
    }

    explicit LRScheduler(std::shared_ptr<Optimizer> optimizer,
                         int64_t last_step = -1);
    virtual ~LRScheduler() = default;

    LRScheduler(const LRScheduler &) = delete;
    LRScheduler &operator=(const LRScheduler &) = delete;

    virtual void Step();
    virtual void Step(int64_t epoch);
    virtual void InitialStep();

    float GetLR() const;
    float BaseLR() const;
    int64_t LastStep() const;

    void ResetStep(int64_t step = -1);
    virtual StateDict State() const;
    virtual void LoadState(const StateDict &state);

protected:
    virtual float GetClosedFormLR() const = 0;
    virtual float GetChainedFormLR() const;
    void ApplyLR(float lr);

    std::shared_ptr<Optimizer> optimizer_;
    int64_t last_step_;
    float current_lr_;
    float base_lr_;
    bool is_initial_ = false;
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
    float GetChainedFormLR() const override;
    float GetClosedFormLR() const override;

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
    float GetChainedFormLR() const override;
    float GetClosedFormLR() const override;

private:
    const int64_t step_size_;
    const float gamma_;
};

class LinearLR : public LRScheduler {
public:
    LinearLR(std::shared_ptr<Optimizer> optimizer,
             float start_factor = 1.0f / 3.0f,
             float end_factor = 1.0f,
             int64_t total_iters = 5,
             int64_t last_step = -1);
    ~LinearLR() override = default;

protected:
    float GetChainedFormLR() const override;
    float GetClosedFormLR() const override;

private:
    const float start_factor_;
    const float end_factor_;
    const int64_t total_iters_;
};

class LambdaLR : public LRScheduler {
public:
    using LambdaFunc = std::function<float(int64_t)>;

    LambdaLR(std::shared_ptr<Optimizer> optimizer, 
                LambdaFunc lr_lambda, 
                int64_t last_step = -1);
    ~LambdaLR() override = default;

protected:
    float GetClosedFormLR() const override;

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
    void InitialStep() override;

    StateDict State() const override;
    void LoadState(const StateDict &state) override;

protected:
    float GetClosedFormLR() const override { return current_lr_; }
    void UndoChildInitialSteps();

private:
    std::vector<std::shared_ptr<LRScheduler>> schedulers_;
    std::vector<int64_t> milestones_;
};

}  // namespace lr_schedulers
}  // namespace infini_train