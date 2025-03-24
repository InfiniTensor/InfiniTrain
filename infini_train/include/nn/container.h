#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/nn/network.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
class Sequential : public Network {
public:
    // TODO(dcj): Use better ctor signature later.
    explicit Sequential(std::vector<std::unique_ptr<Network>> &&layers);

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
};
} // namespace infini_train::nn
