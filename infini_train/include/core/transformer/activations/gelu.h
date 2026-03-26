#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/nn/modules/module.h"

namespace infini_train::nn {

class NewGELU : public infini_train::nn::CloneableModule<NewGELU> {
public:
    static constexpr char kType[] = "NewGELU";
    NewGELU() : CloneableModule(kType) {}

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;
};
} // namespace infini_train::nn
