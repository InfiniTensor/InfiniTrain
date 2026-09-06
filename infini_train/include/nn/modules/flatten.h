#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "infini_train/include/nn/modules/module.h"

namespace infini_train {
class Tensor;
}

namespace infini_train::nn {
class Flatten : public CloneableModule<Flatten> {
public:
    static constexpr char kType[] = "Flatten";

    Flatten(int64_t start_dim = 1, int64_t end_dim = -1)
        : CloneableModule(kType), start_dim_(start_dim), end_dim_(end_dim) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

private:
    int64_t start_dim_;
    int64_t end_dim_;
};
} // namespace infini_train::nn
