#include <memory>
#include <string>

#include "gtest/gtest.h"

#include "infini_train/include/training/model_provider_registry.h"

using infini_train::training::ModelFactory;
using infini_train::training::ModelProvider;
using infini_train::training::ModelProviderRegistry;

TEST(ModelProviderRegistryTest, ResolvesWithoutBuildingAndOwnsBoundArguments) {
    int calls = 0;
    std::string received;
    ModelProvider provider;
    {
        ModelProviderRegistry registry;
        registry.Register("unused", [](const std::string &) {
            ADD_FAILURE() << "Resolved the wrong model factory";
            return std::shared_ptr<infini_train::nn::TransformerModel>{};
        });
        registry.Register("selected", [&](const std::string &weights) {
            ++calls;
            received = weights;
            return std::shared_ptr<infini_train::nn::TransformerModel>{};
        });
        std::string weights = "initial.bin";
        provider = registry.Resolve("selected", weights);
        weights = "changed.bin";
        EXPECT_EQ(calls, 0); // Model construction must wait for rank initialization.
    }
    // A resolved provider must not depend on registry or argument lifetimes.
    provider();
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(received, "initial.bin");
    provider();
    EXPECT_EQ(calls, 2); // Each rank invokes its provider independently.
}

TEST(ModelProviderRegistryTest, RejectsDuplicateAndInvalidRegistrations) {
    ModelProviderRegistry registry;
    ModelFactory factory = [](const std::string &) { return std::shared_ptr<infini_train::nn::TransformerModel>{}; };
    registry.Register("existing", factory);
    EXPECT_DEATH(registry.Register("existing", factory), "Model provider already registered: existing");
    EXPECT_DEATH(registry.Register("", factory), "Model provider name must not be empty");
    EXPECT_DEATH(registry.Register("empty", {}), "Model factory must not be empty: empty");
}

TEST(ModelProviderRegistryTest, ListsRegisteredNamesInUnknownModelError) {
    ModelProviderRegistry registry;
    ModelFactory factory = [](const std::string &) { return std::shared_ptr<infini_train::nn::TransformerModel>{}; };
    registry.Register("second", factory);
    registry.Register("first", factory);
    EXPECT_EQ(registry.Names(), "first, second");
    EXPECT_DEATH(registry.Resolve("missing", ""), "Unknown model: missing. Available models: first, second");
}
