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
    static constexpr char kType[] = "Sigmoid";
    Sigmoid() : CloneableModule(kType) {}
    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
};

class NewGELU : public CloneableModule<NewGELU> {
public:
    static constexpr char kType[] = "NewGELU";
    NewGELU() : CloneableModule(kType) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &x) override;
};

class SwiGLU : public CloneableModule<SwiGLU> {
public:
    static constexpr char kType[] = "SwiGLU";
    SwiGLU() : CloneableModule(kType) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &x) override;
};
} // namespace infini_train::nn
