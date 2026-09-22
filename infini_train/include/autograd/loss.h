#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "infini_train/include/autograd/function.h"

namespace infini_train {
class Tensor;
}

namespace infini_train::autograd {
class CrossEntropy : public Function {
public:
    static constexpr char kType[] = "CrossEntropyFunction";

    explicit CrossEntropy(int64_t ignore_index = -100) : Function(kType), ignore_index_(ignore_index) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    int64_t ignore_index_ = -100;
    int64_t valid_count_ = 0;
};
} // namespace infini_train::autograd
