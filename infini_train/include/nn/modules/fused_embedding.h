#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
class FusedEmbedding : public Module {
public:
    FusedEmbedding(int64_t vocab_size, int64_t max_position, int64_t embed_dim, Device device = Device());
    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
};
} // namespace infini_train::nn
