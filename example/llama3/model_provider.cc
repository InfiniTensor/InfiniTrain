#include "example/llama3/checkpoint_loader.h"
#include "example/llama3/config.h"
#include "infini_train/include/nn/modules/transformer/transformer.h"
#include "infini_train/include/training/model_provider_registry.h"

namespace llama3 {
namespace {

[[maybe_unused]] const bool registered = [] {
    infini_train::training::ModelProviderRegistry::Instance().Register("llama3", [](const std::string &weights) {
        if (!weights.empty()) {
            return LoadFromLLMC(weights);
        }
        auto config = LLaMA3Config();
        SanitizeLLaMA3Config(config);
        return std::make_shared<infini_train::nn::TransformerModel>(config);
    });
    return true;
}();

} // namespace
} // namespace llama3
