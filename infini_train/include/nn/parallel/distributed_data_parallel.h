#pragma once

#include <memory>

#include "infini_train/include/nn/modules/module.h"

namespace infini_train {
class Tensor;
class Device;
} // namespace infini_train

namespace infini_train::nn::parallel {

class DistributedDataParallel : public nn::Module {
public:
    DistributedDataParallel(std::shared_ptr<nn::Module> module, int device_id);

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
};

} // namespace infini_train::nn::parallel
