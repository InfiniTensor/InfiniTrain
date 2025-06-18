#pragma once

#include <memory>
#include <string>
#include <vector>

#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
class Dropout : public Module {
public:
    static constexpr char kType[] = "Dropout";

    Dropout(float p = 0.5f, bool training = true, bool inplace = false)
        : p_(p), training_(training), inplace_(inplace) {}
    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

private:
    void ResetParameters();
    const float p_ = 0.5f;
    const bool training_ = true;
    const bool inplace_ = false;
};
} // namespace infini_train::nn