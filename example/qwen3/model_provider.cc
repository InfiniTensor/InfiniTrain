#include "example/qwen3/checkpoint_loader.h"
#include "example/qwen3/config.h"
#include "infini_train/include/nn/modules/transformer/transformer.h"
#include "infini_train/include/training/model_provider_registry.h"

namespace qwen3 {
namespace {

[[maybe_unused]] const bool registered = [] {
    infini_train::training::ModelProviderRegistry::Instance().Register("qwen3", [](const std::string &weights) {
        if (!weights.empty()) {
            return LoadFromLLMC(weights);
        }
        auto config = Qwen3Config();
        SanitizeQwen3Config(config);
        return std::make_shared<infini_train::nn::TransformerModel>(config);
    });
    return true;
}();

} // namespace
} // namespace qwen3
