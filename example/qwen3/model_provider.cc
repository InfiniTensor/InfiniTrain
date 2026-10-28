#include <memory>
#include <string>

#include "infini_train/include/nn/modules/transformer/transformer.h"
#include "infini_train/include/nn/modules/transformer/transformer_config.h"
#include "infini_train/include/training/model_provider_registry.h"

#include "example/qwen3/checkpoint_loader.h"
#include "example/qwen3/config.h"

namespace qwen3 {
namespace {

[[maybe_unused]] const bool registered = [] {
  infini_train::training::ModelProviderRegistry::Instance().Register(
      "qwen3",
      [](const std::string &weights,
         const infini_train::nn::ActivationRecomputeOptions &recompute) {
        if (!weights.empty()) {
          return LoadFromLLMC(weights, recompute);
        }
        auto config = Qwen3Config();
        SanitizeQwen3Config(config);
        infini_train::nn::ApplyActivationRecomputeOptions(&config, recompute);
        return std::make_shared<infini_train::nn::TransformerModel>(config);
      });
  return true;
}();

}  // namespace
}  // namespace qwen3
