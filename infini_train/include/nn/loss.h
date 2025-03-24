#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/device.h"
#include "infini_train/include/nn/network.h"
#include "infini_train/include/ops.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
class CrossEntropyLoss : public Network {
public:
    CrossEntropyLoss();

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

private:
    void ToImpl(Device device) override;

    std::unique_ptr<ops::CrossEntropy> cross_entropy_op_;
};
} // namespace infini_train::nn
