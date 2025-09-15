#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/nn/modules/module.h"

namespace infini_train {
class Tensor;
}

namespace infini_train::nn {
class Sigmoid : public CloneableModule<Sigmoid> {
public:
    Sigmoid() = default;
    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
};
} // namespace infini_train::nn
