#pragma once

#include <cstdint>
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

}  // namespace infini_train