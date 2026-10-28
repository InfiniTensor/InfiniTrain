#include <memory>
#include <optional>
#include <ostream>
#include <string>

#include "glog/logging.h"

#include "infini_train/include/nn/modules/transformer/transformer.h"
#include "infini_train/include/nn/modules/transformer/transformer_config.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/training/model_provider_registry.h"

#include "example/mixtral/checkpoint_loader.h"
#include "example/mixtral/config.h"

namespace mixtral {
namespace {

[[maybe_unused]] const bool registered = [] {
  infini_train::training::ModelProviderRegistry::Instance().Register(
      "mixtral",
      [](const std::string &weights,
         const infini_train::nn::ActivationRecomputeOptions &recompute) {
        namespace global = infini_train::nn::parallel::global;
        CHECK_EQ(global::GetTensorParallelSize(), 1)
            << "Mixtral currently requires TP=1";
        CHECK_EQ(global::GetExpertTensorParallelSize(), 1)
            << "Mixtral currently requires ETP=1";
        CHECK(!global::GetSequenceParallelEnabled())
            << "Mixtral currently requires SP disabled";
        auto config =
            weights.empty() ? TinyMixtralConfig() : ConfigFromLLMC(weights);
        SanitizeTinyMixtralConfig(config);
        CHECK_EQ(
            config.moe_config->num_experts % global::GetExpertParallelSize(),
            0);
        infini_train::nn::ApplyActivationRecomputeOptions(&config, recompute);
        if (!weights.empty()) {
          return LoadFromLLMC(weights, config);
        }
        return std::make_shared<infini_train::nn::TransformerModel>(config);
      });
  return true;
}();

}  // namespace
}  // namespace mixtral
