#pragma once

#include <atomic>
#include <memory>
#include <sys/types.h>
#include <vector>

#include "infini_train/include/autograd/function.h"

namespace infini_train {
class Tensor;
}

namespace infini_train::autograd {
class AccumulateGrad final : public Function {
public:
    AccumulateGrad(std::shared_ptr<Tensor> tensor, float learning_rate = 1.0f);

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &) override;

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &) override;

private:
    std::shared_ptr<Tensor> tensor_ = nullptr;
    float learning_rate_ = 1.0f;

    uint64_t id_ = 0;

    static std::atomic<uint64_t> global_id_counter_;
};
} // namespace infini_train::autograd
