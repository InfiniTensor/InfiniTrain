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

    // Chainable builder methods for fluent API
    ModuleSpec &with_param(const std::string &key, std::any value) {
        params_[key] = std::move(value);
        return *this;
    }

    ModuleSpec &with_submodule(const std::string &name, ModuleSpec spec) {
        submodules_[name] = std::move(spec);
        return *this;
    }

    ModuleSpec &
    with_build(std::function<std::shared_ptr<Module>(const TransformerConfig &, const ModuleSpec &)> build_fn) {
        build = std::move(build_fn);
        return *this;
    }

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

    bool Has(std::type_index type) const { return registry_.contains(type); }

    std::unordered_set<std::type_index> RegisteredTypes() const {
        std::unordered_set<std::type_index> types;
        for (const auto &[type, _] : registry_) { types.insert(type); }
        return types;
    }

private:
    std::unordered_map<std::type_index, ModuleCreator> registry_;
};

/**
 * @brief Register a module type with automatic creator inference
 *
 * Usage:
 *   REGISTER_MODULE(LayerNorm);
 *   REGISTER_MODULE(RMSNorm);
 *
 * The macro assumes the module class has a constructor that accepts (const TransformerConfig&, const ModuleSpec&)
 * or provides a static Create() method with the same signature.
 */
#define REGISTER_MODULE(ModuleClass)                                                                                   \
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

/**
 * @brief Register a module type with custom creator function
 *
 * Usage:
 *   REGISTER_MODULE_CUSTOM(LayerNorm, [](const TransformerConfig &config, const ModuleSpec &spec) {
 *       return std::make_shared<LayerNorm>(config.n_embd);
 *   });
 */
#define REGISTER_MODULE_CUSTOM(ModuleClass, CreatorFunc)                                                               \
    namespace {                                                                                                        \
    struct ModuleClass##Registry {                                                                                     \
        ModuleClass##Registry() { ModuleRegistry::Instance().Register(typeid(ModuleClass), CreatorFunc); }             \
    };                                                                                                                 \
    static ModuleClass##Registry g_##ModuleClass##_registry;                                                           \
    }

/**
 * @brief Safely get a required parameter from ModuleSpec
 * @throws std::runtime_error if parameter is missing or has wrong type
 */
template <typename T> inline T GetRequiredParam(const ModuleSpec &spec, const std::string &key) {
    auto it = spec.params_.find(key);
    if (it == spec.params_.end()) {
        throw std::runtime_error("Missing required parameter: " + key);
    }
    try {
        return std::any_cast<T>(it->second);
    } catch (const std::bad_any_cast &e) {
        throw std::runtime_error("Parameter type mismatch for key '" + key + "': expected " + typeid(T).name());
    }
}

/**
 * @brief Get an optional parameter from ModuleSpec with default value
 */
template <typename T> inline T GetOptionalParam(const ModuleSpec &spec, const std::string &key, T default_value) {
    auto it = spec.params_.find(key);
    if (it == spec.params_.end()) {
        return default_value;
    }
    try {
        return std::any_cast<T>(it->second);
    } catch (const std::bad_any_cast &e) { return default_value; }
}

std::shared_ptr<Module> build_module(const TransformerConfig &config, const ModuleSpec &spec);
} // namespace infini_train::nn
