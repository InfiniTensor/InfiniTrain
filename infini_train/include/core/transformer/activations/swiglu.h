#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/nn/modules/module.h"

namespace infini_train::nn {

class SwiGLU : public infini_train::nn::CloneableModule<SwiGLU> {
public:
    static constexpr char kType[] = "SwiGLU";
    SwiGLU() : CloneableModule(kType) {}

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;
};
} // namespace infini_train::nn
