#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>

#include "infini_train/include/training/training.h"

namespace infini_train::training {

// A factory owns model-specific configuration and checkpoint import. Binding
// a factory must not construct a model: Pretrain invokes it after rank setup.
using ModelFactory = std::function<std::shared_ptr<nn::TransformerModel>(const std::string &weights)>;

class ModelProviderRegistry {
public:
    static ModelProviderRegistry &Instance();

    // Register during startup, before concurrent lookups/training begin.
    // Each model/alias must have exactly one nonempty factory.
    void Register(const std::string &name, ModelFactory factory);

    // Return an independently owned callback; invocation is deferred to Pretrain.
    ModelProvider Resolve(const std::string &name, const std::string &weights) const;

    // Stable ordering for CLI help and unknown-model diagnostics.
    std::string Names() const;

private:
    std::map<std::string, ModelFactory> factories_;
};

} // namespace infini_train::training
