#include "infini_train/include/nn/modules/module.h"

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/device.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {

Module::Module(DataType dtype) : Module(kUndefinedType, dtype) {}

Module::Module(const std::string &type, DataType dtype)
    : type_(type), dtype_(dtype), device_(DeviceManager::Instance()->GetDefaultDevice()) {}

const std::string &Module::type() const { return type_; }

std::vector<std::shared_ptr<Tensor>> Module::Parameters() const {
    std::vector<std::shared_ptr<Tensor>> params;
    for (auto &[_, param] : parameters_) { params.push_back(param); }
    for (auto &[_, module] : modules_) {
        for (auto &param : module->Parameters()) { params.push_back(param); }
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

std::vector<std::shared_ptr<Tensor>> Module::Buffers() const {
    std::vector<std::shared_ptr<Tensor>> buffers;
    for (auto &[_, buffer] : buffers_) { buffers.push_back(buffer); }
    for (auto &[_, module] : modules_) {
        for (auto &buffer : module->Buffers()) { buffers.push_back(buffer); }
    }
    return buffers;
}

std::vector<std::shared_ptr<Module>> Module::modules() {
    std::vector<std::shared_ptr<Module>> modules;
    auto named_modules = NamedModules();
    for (auto &[_, module] : named_modules) {
        if (_ != "") {
            modules.push_back(module);
        }
    }
    modules.insert(modules.begin(), named_modules[""]);
    return modules;
}

// FIXME(dcj): can not call this function in constructor
std::unordered_map<std::string, std::shared_ptr<Module>>
Module::NamedModules(const std::string &prefix, bool remove_duplicate, std::unordered_set<Module *> *memory) {
    std::unordered_set<Module *> local_memory;
    if (memory == nullptr) {
        memory = &local_memory;
    }
    std::unordered_map<std::string, std::shared_ptr<Module>> named_modules;
    if (!memory->contains(this)) {
        if (remove_duplicate) {
            memory->insert(this);
        }
        CHECK(!named_modules.contains(prefix));
        named_modules.emplace(prefix, shared_from_this());
        for (auto &[name, module] : modules_) {
            if (!module) {
                continue;
            }
            auto submodule_prefix = (prefix.empty() ? "" : prefix + ".") + name;
            for (auto &[sub_name, sub_module] : module->NamedModules(submodule_prefix, remove_duplicate, memory)) {
                CHECK(!named_modules.contains(sub_name));
                named_modules.emplace(sub_name, sub_module);
            }
        }
    }
    return named_modules;
}

std::shared_ptr<Module> Module::mutable_module(const std::string &name) { return modules_.at(name); }

const Module &Module::module(const std::string &name) const {
    CHECK(modules_.find(name) != modules_.end());
    return *modules_.at(name).get();
}

std::unordered_map<std::string, std::shared_ptr<Tensor>> Module::StateDict() const {
    std::unordered_map<std::string, std::shared_ptr<Tensor>> state;
    for (auto &[name, param] : parameters_) { state.emplace(name, param); }
    for (auto &[name, buffer] : buffers_) { state.emplace(name, buffer); }
    for (auto &[name, module] : modules_) {
        for (auto &[sub_name, param] : module->StateDict()) { state.emplace(name + "." + sub_name, param); }
    }
    return state;
}

void Module::To(const Device *device) {
    CHECK_NOTNULL(device);
    if (device == device_) {
        return;
    }

    std::unordered_map<std::string, std::shared_ptr<Tensor>> new_parameters;
    std::unordered_map<std::string, std::shared_ptr<Tensor>> new_buffers;
    for (auto &[name, param] : parameters_) {
        new_parameters.emplace(name, std::make_shared<Tensor>(param->To(device)));
    }
    for (auto &[name, buffer] : buffers_) { new_buffers.emplace(name, std::make_shared<Tensor>(buffer->To(device))); }
    parameters_ = std::move(new_parameters);
    buffers_ = std::move(new_buffers);
    device_ = device;

    for (auto &[_, module] : modules_) { module->To(device); }
}

void Module::To(DataType dtype) {
    if (dtype == dtype_) {
        return;
    }

    std::unordered_map<std::string, std::shared_ptr<Tensor>> new_parameters;
    for (auto &[name, param] : parameters_) {
        new_parameters.emplace(name, std::make_shared<Tensor>(param->To(dtype)));
    }
    parameters_ = std::move(new_parameters);
    dtype_ = dtype;

    for (auto &[_, layer] : modules_) { layer->To(dtype); }
}

void Module::Apply(std::function<void(Module *)> fn) {
    for (auto &[_, module] : modules_) { module->Apply(fn); }
    fn(this);
}

std::shared_ptr<Module> Module::ReplicateForDataParallel(int device_idx) const {
    // TODO(dcj): use device_idx later
    return std::make_shared<Module>(*this);
}
} // namespace infini_train::nn
