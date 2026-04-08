#include "infini_train/include/nn/modules/transformer/spec_utils.h"

#include <any>
#include <string>
#include <typeindex>

#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/modules/transformer/transformer_config.h"

namespace infini_train::nn {

ModuleSpec &ModuleSpec::WithParam(const std::string &key, std::any value) {
    params_[key] = std::move(value);
    return *this;
}

ModuleSpec &ModuleSpec::WithSubmodule(const std::string &name, ModuleSpec spec) {
    submodules_[name] = std::move(spec);
    return *this;
}

ModuleSpec &
ModuleSpec::WithBuild(std::function<std::shared_ptr<Module>(const TransformerConfig &, const ModuleSpec &)> build_fn) {
    build = std::move(build_fn);
    return *this;
}

void ModuleRegistry::Register(std::type_index type, ModuleCreator creator) {
    CHECK(!registry_.contains(type)) << "Module type already registered: " << type.name();

    registry_[type] = std::move(creator);
}

ModuleCreator ModuleRegistry::Get(std::type_index type) const {
    auto it = registry_.find(type);
    if (it == registry_.end()) {
        return nullptr;
    }
    return it->second;
}

bool ModuleRegistry::Has(std::type_index type) const { return registry_.contains(type); }

std::unordered_set<std::type_index> ModuleRegistry::RegisteredTypes() const {
    std::unordered_set<std::type_index> types;
    for (const auto &[type, _] : registry_) { types.insert(type); }
    return types;
}

std::shared_ptr<Module> BuildModule(const TransformerConfig &config, const ModuleSpec &spec) {
    if (spec.build) {
        return spec.build(config, spec);
    }

    CHECK(spec.module_ != typeid(void)) << "ModuleSpec.module is not set";

    auto creator = ModuleRegistry::Instance().Get(spec.module_);

    CHECK(creator) << "Module not registered: " << spec.module_.name();

    auto module = creator(config, spec);

    return module;
}
} // namespace infini_train::nn
