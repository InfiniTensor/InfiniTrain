#include "infini_train/include/training/model_provider_registry.h"

#include <utility>

#include "glog/logging.h"

namespace infini_train::training {

ModelProviderRegistry &ModelProviderRegistry::Instance() {
    static ModelProviderRegistry registry;
    return registry;
}

void ModelProviderRegistry::Register(const std::string &name, ModelFactory factory) {
    CHECK(!name.empty()) << "Model provider name must not be empty";
    CHECK(factory) << "Model factory must not be empty: " << name;
    CHECK(!factories_.contains(name)) << "Model provider already registered: " << name;
    factories_.emplace(name, std::move(factory));
}

ModelProvider ModelProviderRegistry::Resolve(const std::string &name, const std::string &weights) const {
    const auto it = factories_.find(name);
    CHECK(it != factories_.end()) << "Unknown model: " << name << ". Available models: " << Names();
    return [factory = it->second, weights] { return factory(weights); };
}

std::string ModelProviderRegistry::Names() const {
    std::string names;
    for (const auto &[name, factory] : factories_) {
        if (!names.empty()) {
            names += ", ";
        }
        names += name;
    }
    return names;
}

} // namespace infini_train::training
