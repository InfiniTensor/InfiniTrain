#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
class Embedding : public Module {
public:
    static constexpr char kType[] = "Embedding";

    static constexpr char kParamWeightName[] = "weight";

    Embedding(int num_embeddings, int embedding_dim, Device device = Device());
    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

private:
    void ResetParameters();
};
} // namespace infini_train::nn
