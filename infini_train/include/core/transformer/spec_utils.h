#pragma once

#include <any>
#include <functional>
#include <memory>
#include <typeindex>

#include "glog/logging.h"

#include "infini_train/include/core/transformer/transformer_config.h"

namespace infini_train::nn {

class Module;

struct ModuleSpec {
    ModuleSpec() = default;

    explicit ModuleSpec(std::type_index m) : module_(m) {}

    std::type_index module_{typeid(void)};

    TransformerConfig config_;

    std::unordered_map<std::string, std::any> params_;

    std::unordered_map<std::string, ModuleSpec> submodules_;

    std::function<std::shared_ptr<Module>(const TransformerConfig &, const ModuleSpec &)> build{nullptr};
};

using ModuleCreator = std::function<std::shared_ptr<Module>(const TransformerConfig &, const ModuleSpec &)>;

class ModuleRegistry {
public:
    static ModuleRegistry &Instance() {
        static ModuleRegistry inst;
        return inst;
    }

    void Register(std::type_index type, ModuleCreator creator);

    ModuleCreator Get(std::type_index type) const;

private:
    std::unordered_map<std::type_index, ModuleCreator> registry_;
};

std::shared_ptr<Module> build_module(const TransformerConfig &config, const ModuleSpec &spec);

} // namespace infini_train::nn
