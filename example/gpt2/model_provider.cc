#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "infini_train/include/nn/modules/transformer/transformer.h"
#include "infini_train/include/nn/modules/transformer/transformer_config.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/training/model_provider_registry.h"

#include "example/gpt2/checkpoint_loader.h"
#include "example/gpt2/config.h"

namespace gpt2 {
namespace {

[[maybe_unused]] const bool registered = [] {
  auto &registry = infini_train::training::ModelProviderRegistry::Instance();
  for (const auto &[name, size] : kModelSizes) {
    registry.Register(
        name,
        [name = name](
            const std::string &weights,
            const infini_train::nn::ActivationRecomputeOptions &recompute) {
          if (!weights.empty()) {
            return LoadFromLLMC(weights, recompute);
          }
          auto config = GPT2Config(
              name,
              infini_train::nn::parallel::global::GetTensorParallelSize());
          SanitizeGPT2Config(config);
          infini_train::nn::ApplyActivationRecomputeOptions(&config, recompute);
          return std::make_shared<infini_train::nn::TransformerModel>(config);
        });
  }
  return true;
}();

}  // namespace
}  // namespace gpt2
