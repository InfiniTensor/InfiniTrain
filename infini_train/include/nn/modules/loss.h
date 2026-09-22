#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "infini_train/include/nn/modules/module.h"

namespace infini_train::nn {
class CrossEntropyLoss : public CloneableModule<CrossEntropyLoss> {
public:
    static constexpr char kType[] = "CrossEntropyLoss";
    explicit CrossEntropyLoss(int64_t ignore_index = -100) : CloneableModule(kType), ignore_index_(ignore_index) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

private:
    int64_t ignore_index_ = -100;
};
} // namespace infini_train::nn
