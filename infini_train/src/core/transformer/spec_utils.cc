#include "infini_train/include/core/transformer/spec_utils.h"

#include <any>
#include <stdexcept>
#include <string>
#include <typeindex>

#include "infini_train/include/core/transformer/transformer_config.h"
#include "infini_train/include/nn/modules/module.h"

namespace infini_train::nn {

void ModuleRegistry::Register(std::type_index type, ModuleCreator creator) { registry_[type] = std::move(creator); }

ModuleCreator ModuleRegistry::Get(std::type_index type) const {
    auto it = registry_.find(type);
    if (it == registry_.end()) {
        return nullptr;
    }
    return it->second;
}

std::shared_ptr<Module> build_module(const TransformerConfig &config, const ModuleSpec &spec) {
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
