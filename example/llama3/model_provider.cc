#include <memory>
#include <string>

#include "infini_train/include/nn/modules/transformer/transformer.h"
#include "infini_train/include/nn/modules/transformer/transformer_config.h"
#include "infini_train/include/training/model_provider_registry.h"

#include "example/llama3/checkpoint_loader.h"
#include "example/llama3/config.h"

namespace llama3 {
namespace {

[[maybe_unused]] const bool registered = [] {
  infini_train::training::ModelProviderRegistry::Instance().Register(
      "llama3",
      [](const std::string &weights,
         const infini_train::nn::ActivationRecomputeOptions &recompute) {
        if (!weights.empty()) {
          return LoadFromLLMC(weights, recompute);
        }
        auto config = LLaMA3Config();
        SanitizeLLaMA3Config(config);
        infini_train::nn::ApplyActivationRecomputeOptions(&config, recompute);
        return std::make_shared<infini_train::nn::TransformerModel>(config);
      });
  return true;
}();

}  // namespace
}  // namespace llama3
