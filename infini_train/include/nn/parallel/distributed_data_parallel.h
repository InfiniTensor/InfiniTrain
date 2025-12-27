#pragma once

#include <memory>

#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/reducer.h"

namespace infini_train {
class Tensor;
class Device;
} // namespace infini_train

namespace infini_train::nn::parallel {

class DistributedDataParallel : public nn::Module {
public:
    DistributedDataParallel(std::shared_ptr<nn::Module> module, int device_id,
                            const ReducerOptions &opts = ReducerOptions{});

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

private:
    std::shared_ptr<Reducer> reducer_ = nullptr;
};

} // namespace infini_train::nn::parallel
