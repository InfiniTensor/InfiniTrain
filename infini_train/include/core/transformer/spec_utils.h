#pragma once

#include <any>
#include <functional>
#include <memory>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>

#include "glog/logging.h"

#include "infini_train/include/core/transformer/transformer_config.h"

namespace infini_train::nn {

class Module;

struct ModuleSpec {
    ModuleSpec() = default;

    explicit ModuleSpec(std::type_index m) : module_(m) {}

    ModuleSpec &WithParam(const std::string &key, std::any value);

    ModuleSpec &WithSubmodule(const std::string &name, ModuleSpec spec);

    ModuleSpec &
    WithBuild(std::function<std::shared_ptr<Module>(const TransformerConfig &, const ModuleSpec &)> build_fn);

    std::type_index module_{typeid(void)};
    std::unordered_map<std::string, std::any> params_;
    std::unordered_map<std::string, ModuleSpec> submodules_;
    std::function<std::shared_ptr<Module>(const TransformerConfig &, const ModuleSpec &)> build{nullptr};
};

using ModuleCreator = std::function<std::shared_ptr<Module>(const TransformerConfig &, const ModuleSpec &)>;

class ModuleRegistry {
public:
    static ModuleRegistry &Instance() {
        static ModuleRegistry instance;
        return instance;
    }

    void Register(std::type_index type, ModuleCreator creator);

    ModuleCreator Get(std::type_index type) const;

    bool Has(std::type_index type) const;

    std::unordered_set<std::type_index> RegisteredTypes() const;

private:
    std::unordered_map<std::type_index, ModuleCreator> registry_;
};

// Register a module type with automatic creator inference
#define INFINI_TRAIN_REGISTER_MODULE(ModuleClass)                                                                      \
    namespace {                                                                                                        \
    struct ModuleClass##Registry {                                                                                     \
        ModuleClass##Registry() {                                                                                      \
            ModuleRegistry::Instance().Register(typeid(ModuleClass),                                                   \
                                                [](const TransformerConfig &config, const ModuleSpec &spec) {          \
                                                    return std::make_shared<ModuleClass>(config, spec);                \
                                                });                                                                    \
        }                                                                                                              \
    };                                                                                                                 \
    static ModuleClass##Registry g_##ModuleClass##_registry;                                                           \
    }

// Register a module type with custom creator function
#define INFINI_TRAIN_REGISTER_MODULE_CUSTOM(ModuleClass, CreatorFunc)                                                  \
    namespace {                                                                                                        \
    struct ModuleClass##Registry {                                                                                     \
        ModuleClass##Registry() { ModuleRegistry::Instance().Register(typeid(ModuleClass), CreatorFunc); }             \
    };                                                                                                                 \
    static ModuleClass##Registry g_##ModuleClass##_registry;                                                           \
    }

// Get a required parameter from ModuleSpec
template <typename T> T GetRequiredParam(const ModuleSpec &spec, const std::string &key) {
    CHECK(spec.params_.contains(key)) << "Missing required parameter: " << key;

    const T *value = std::any_cast<T>(&spec.params_.at(key));
    CHECK(value) << "Parameter type mismatch for key '" << key << "': expected " << typeid(T).name() << ", got "
                 << spec.params_.at(key).type().name();
    return *value;
}

std::shared_ptr<Module> BuildModule(const TransformerConfig &config, const ModuleSpec &spec);
} // namespace infini_train::nn
