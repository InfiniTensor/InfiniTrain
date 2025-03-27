#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/device.h"
#include "infini_train/include/nn/network.h"
#include "infini_train/include/ops.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
class Linear : public Network {
public:
    Linear(int64_t in_dim, int64_t out_dim);
    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

private:
    void ToImpl(Device device) override;

    std::unique_ptr<ops::Linear> linear_op_ = nullptr;
};
} // namespace infini_train::nn
