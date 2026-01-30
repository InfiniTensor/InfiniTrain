#include "infini_train/include/nn/modules/module.h"

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/autograd/function.h"
#include "infini_train/include/common/hook.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"
#include "infini_train/include/utils/global_module_hook_registry.h"

#ifndef UNLIKELY
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif

namespace infini_train::nn {

Module::Module() : Module(kUndefinedType) {}

Module::Module(const std::string &type) : type_(type), device_(DeviceManager::Instance()->GetDefaultDevice()) {}

const std::string &Module::type() const { return type_; }

std::vector<std::shared_ptr<Tensor>> Module::Parameters() const {
    std::vector<std::shared_ptr<Tensor>> params;
    std::unordered_set<const Tensor *> visited;

    auto AddIfUnvisited = [&](const std::shared_ptr<Tensor> &param) {
        if (visited.insert(param.get()).second) {
            params.push_back(param);
        }
    };

    // Add parameters of this module
    for (const auto &[_, param] : parameters_) { AddIfUnvisited(param); }

    // Recursively add parameters of submodules
    for (const auto &[_, module] : modules_) {
        for (const auto &param : module->Parameters()) { AddIfUnvisited(param); }
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
        if (name.starts_with("__pp")) {
            continue;
        }
        for (auto &[sub_name, param] : module->StateDict()) { state.emplace(name + "." + sub_name, param); }
    }
    return state;
}

std::vector<std::shared_ptr<Tensor>> Module::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    LOG(FATAL) << "Forward function not implemented for this module";
    return {};
}

std::vector<std::shared_ptr<Tensor>> Module::operator()(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    // 1. Call global module forward pre-hooks
    utils::GlobalModuleHookRegistry::Instance().CallModuleForwardPreHooks(this, input_tensors);

    // 2. Call local forward pre-hooks
    for (const auto &hook : forward_pre_hooks_) {
        if (hook) {
            hook(this, input_tensors);
        }
    }

    // 3. Call actual Forward implementation
    auto output_tensors = Forward(input_tensors);

    // 4. Call local forward post-hooks
    for (const auto &hook : forward_post_hooks_) {
        if (hook) {
            hook(this, input_tensors, output_tensors);
        }
    }

    // 5. Call global module forward hooks
    utils::GlobalModuleHookRegistry::Instance().CallModuleForwardHooks(this, input_tensors, output_tensors);

    // 6. Register backward hooks on output tensors' grad_fn
    const bool has_local_backward_hooks = !backward_pre_hooks_.empty() || !backward_post_hooks_.empty();
    const bool has_global_backward_hooks = utils::GlobalModuleHookRegistry::Instance().HasModuleBackwardHooks();

    if (UNLIKELY(has_local_backward_hooks || has_global_backward_hooks)) {
        for (const auto &output : output_tensors) {
            if (output && output->grad_fn()) {
                // Local backward prehooks
                if (!backward_pre_hooks_.empty()) {
                    output->grad_fn()->RegisterBackwardPreHook(
                        [this](autograd::Function *, const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
                            for (const auto &hook : backward_pre_hooks_) {
                                if (hook) {
                                    hook(this, grad_outputs);
                                }
                            }
                        });
                }
                // Local backward post-hooks
                if (!backward_post_hooks_.empty()) {
                    output->grad_fn()->RegisterBackwardPostHook(
                        [this](autograd::Function *, const std::vector<std::shared_ptr<Tensor>> &grad_inputs,
                               const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
                            for (const auto &hook : backward_post_hooks_) {
                                if (hook) {
                                    hook(this, grad_inputs, grad_outputs);
                                }
                            }
                        });
                }
                // Global backward hooks (only register once per grad_fn using atomic CAS)
                if (has_global_backward_hooks && output->grad_fn()->TryMarkModuleBackwardHookRegistered()) {
                    output->grad_fn()->RegisterBackwardPostHook(
                        [this](autograd::Function *, const std::vector<std::shared_ptr<Tensor>> &grad_inputs,
                               const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
                            // Registry convention: (grad_outputs, grad_inputs) - PyTorch style
                            utils::GlobalModuleHookRegistry::Instance().CallModuleFullBackwardHooks(this, grad_outputs,
                                                                                                    grad_inputs);
                        });
                }
            }
        }
    }

    return output_tensors;
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
    std::unordered_map<std::string, std::shared_ptr<Tensor>> new_parameters;
    std::unordered_map<std::string, std::shared_ptr<Tensor>> new_buffers;
    for (auto &[name, param] : parameters_) {
        new_parameters.emplace(name, std::make_shared<Tensor>(param->To(dtype)));
    }
    for (auto &[name, buffer] : buffers_) { new_buffers.emplace(name, std::make_shared<Tensor>(buffer->To(dtype))); }
    parameters_ = std::move(new_parameters);
    buffers_ = std::move(new_buffers);

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

std::shared_ptr<infini_train::HookHandle> Module::RegisterForwardPreHook(ModulePreHook hook) {
    forward_pre_hooks_.push_back(std::move(hook));
    return std::make_shared<ModuleHookHandleImpl<ModulePreHook>>(&forward_pre_hooks_, forward_pre_hooks_.size() - 1);
}

std::shared_ptr<infini_train::HookHandle> Module::RegisterForwardPostHook(ModulePostHook hook) {
    forward_post_hooks_.push_back(std::move(hook));
    return std::make_shared<ModuleHookHandleImpl<ModulePostHook>>(&forward_post_hooks_, forward_post_hooks_.size() - 1);
}

std::shared_ptr<infini_train::HookHandle> Module::RegisterBackwardPreHook(ModulePreHook hook) {
    backward_pre_hooks_.push_back(std::move(hook));
    return std::make_shared<ModuleHookHandleImpl<ModulePreHook>>(&backward_pre_hooks_, backward_pre_hooks_.size() - 1);
}

std::shared_ptr<infini_train::HookHandle> Module::RegisterBackwardPostHook(ModulePostHook hook) {
    backward_post_hooks_.push_back(std::move(hook));
    return std::make_shared<ModuleHookHandleImpl<ModulePostHook>>(&backward_post_hooks_,
                                                                  backward_post_hooks_.size() - 1);
}
} // namespace infini_train::nn
