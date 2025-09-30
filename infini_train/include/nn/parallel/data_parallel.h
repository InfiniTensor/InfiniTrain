#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::parallel {
class DataParallel : public Module {
public:
    DataParallel(const std::shared_ptr<Module> &module, int dim = 0);

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

private:
    int dim_ = 0;
    std::vector<const Device *> devices_;
    const Device *output_device_ = nullptr;
    const Device *src_device_ = nullptr;
};
} // namespace infini_train::nn::parallel
