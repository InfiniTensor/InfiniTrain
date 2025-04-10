#include "infini_train/include/nn/modules/module.h"

#include <memory>
#include <vector>

#include "infini_train/include/device.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
std::vector<std::shared_ptr<Tensor>> Module::Parameters() const {
    std::vector<std::shared_ptr<Tensor>> params;
    for (auto &[_, param] : parameters_) { params.push_back(param); }
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
    device_ = device;

    for (auto &[_, layer] : modules_) { layer->To(device); }
}
} // namespace infini_train::nn
