#include "infini_train/include/nn/modules/module.h"

#include <memory>
#include <vector>

#include "infini_train/include/device.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
const std::string &Module::type() const { return type_; }

std::vector<std::shared_ptr<Tensor>> Module::Parameters() const {
    std::vector<std::shared_ptr<Tensor>> params;
    for (auto &[_, param] : parameters_) { params.push_back(param); }
    for (auto &[_, layer] : modules_) {
        for (auto &param : layer->Parameters()) { params.push_back(param); }
    }
    return params;
}

bool Module::has_parameter(const std::string &name) const { return parameters_.find(name) != parameters_.end(); }

std::shared_ptr<Tensor> *Module::mutable_parameter(const std::string &name) {
    CHECK(parameters_.find(name) != parameters_.end());
    return &parameters_.at(name);
}

const std::shared_ptr<Tensor> &Module::parameter(const std::string &name) const {
    CHECK(parameters_.find(name) != parameters_.end());
    return parameters_.at(name);
}

std::vector<Module *> Module::modules() const {
    std::vector<Module *> modules;
    for (auto &[_, module] : modules_) { modules.push_back(module.get()); }
    return modules;
}

Module *Module::mutable_module(const std::string &name) {
    CHECK(modules_.find(name) != modules_.end());
    return modules_.at(name).get();
}

const Module &Module::module(const std::string &name) const {
    CHECK(modules_.find(name) != modules_.end());
    return *modules_.at(name).get();
}

std::unordered_map<std::string, std::shared_ptr<Tensor>> Module::StateDict() const {
    std::unordered_map<std::string, std::shared_ptr<Tensor>> state;
    for (auto &[name, param] : parameters_) { state.emplace(name, param); }
    for (auto &[name, layer] : modules_) {
        for (auto &[sub_name, param] : layer->StateDict()) { state.emplace(name + "." + sub_name, param); }
    }
    return state;
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

void Module::Apply(std::function<void(Module *)> fn) {
    for (auto *module : modules()) { module->Apply(fn); }
    fn(this);
}
} // namespace infini_train::nn
