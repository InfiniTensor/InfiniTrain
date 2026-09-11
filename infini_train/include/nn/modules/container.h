#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "infini_train/include/nn/modules/module.h"

namespace infini_train {
class Tensor;
}

namespace infini_train::nn {
class Sequential : public CloneableModule<Sequential> {
public:
    static constexpr char kType[] = "Sequential";
    // TODO(dcj): Use better ctor signature later.
    explicit Sequential(std::vector<std::shared_ptr<Module>> &&layers);

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
};

class ModuleDict : public CloneableModule<ModuleDict> {
public:
    static constexpr char kType[] = "ModuleDict";
    using Item = std::pair<std::string, std::shared_ptr<Module>>;
    explicit ModuleDict(std::vector<Item> modules);

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
};

class ModuleList : public CloneableModule<ModuleList> {
public:
    static constexpr char kType[] = "ModuleList";

    explicit ModuleList(std::vector<std::shared_ptr<Module>> &&layers);

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

    std::vector<std::shared_ptr<Module>>::iterator begin();
    std::vector<std::shared_ptr<Module>>::iterator end();
    std::vector<std::shared_ptr<Module>>::const_iterator begin() const;
    std::vector<std::shared_ptr<Module>>::const_iterator end() const;

    std::shared_ptr<Module> &operator[](std::size_t idx);

private:
    std::vector<std::shared_ptr<Module>> module_list_;
};
} // namespace infini_train::nn
