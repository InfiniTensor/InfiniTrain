#include "infini_train/include/nn/modules/module.h"

#include <vector>

#include "infini_train/include/device.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
std::vector<Tensor *> Module::Parameters() const {
    std::vector<Tensor *> params;
    for (auto &[_, param] : parameters_) { params.push_back(param.get()); }
    for (auto &[_, layer] : modules_) {
        for (auto &param : layer->Parameters()) { params.push_back(param); }
    }
    return params;
}

void Module::To(Device device) {
    if (device == device_) {
        return;
    }

    std::unordered_map<std::string, std::shared_ptr<Tensor>> new_parameters;
    for (auto &[name, param] : parameters_) {
        new_parameters.emplace(name, std::make_shared<Tensor>(param->To(device)));
    }
    parameters_ = std::move(new_parameters);

    ToImpl(device);
    device_ = device;

    for (auto &[_, layer] : modules_) { layer->To(device); }
}
} // namespace infini_train::nn
