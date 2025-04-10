#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
class LayerNorm : public Module {
public:
    LayerNorm(int64_t embed_dim, float eps = 1e-5f, Device device = Device());
    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

protected:
    float eps_;
};
} // namespace infini_train::nn
