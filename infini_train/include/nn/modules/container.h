#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "infini_train/include/nn/modules/module.h"

namespace infini_train {
class Tensor;
}

namespace infini_train::nn {
class Sequential : public CloneableModule<Sequential> {
public:
    // TODO(dcj): Use better ctor signature later.
    explicit Sequential(std::vector<std::shared_ptr<Module>> &&layers);

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
};

class ModuleDict : public CloneableModule<ModuleDict> {
public:
    // TODO(dcj): in torch, there is a dict with the order of insertion
    explicit ModuleDict(std::unordered_map<std::string, std::shared_ptr<Module>> modules);

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
};

class ModuleList : public CloneableModule<ModuleList> {
public:
    static constexpr char kType[] = "ModuleList";

    explicit ModuleList(std::vector<std::shared_ptr<Module>> &&layers);

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

    auto begin() { return module_list_.begin(); }
    auto end() { return module_list_.end(); }
    auto begin() const { return module_list_.begin(); }
    auto end() const { return module_list_.end(); }

private:
    std::vector<std::shared_ptr<Module>> module_list_;
};
} // namespace infini_train::nn
