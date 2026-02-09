#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/module.h"

namespace infini_train {
class Tensor;
} // namespace infini_train

namespace infini_train::nn::parallel {
class DataParallel : public Module {
public:
    DataParallel(const std::shared_ptr<Module> &module, int dim, Device::DeviceType device_type);

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

private:
    int dim_ = 0;
    std::vector<Device> devices_;
    Device output_device_;
    Device src_device_;
};
} // namespace infini_train::nn::parallel
