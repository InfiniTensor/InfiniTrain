#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/nn/modules/module.h"

namespace infini_train::nn {

class RMSNorm : public infini_train::nn::CloneableModule<RMSNorm> {
public:
    static constexpr char kType[] = "RMSNorm";
    static constexpr char kParamWeightName[] = "weight";

    explicit RMSNorm(int64_t dim, float eps = 1e-6f, infini_train::Device device = infini_train::Device());

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;

private:
    float eps_ = 1e-5f;
};
} // namespace infini_train::nn
