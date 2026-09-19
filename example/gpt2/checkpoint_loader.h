#pragma once

#include <memory>
#include <string>

#include "infini_train/include/nn/modules/transformer/transformer_config.h"
#include "infini_train/include/nn/parallel/pp/pipeline_layout.h"

namespace infini_train::nn {
class TransformerModel;
} // namespace infini_train::nn

namespace gpt2 {
// Load using the default model config; checkpoint specifications must match.
std::shared_ptr<infini_train::nn::TransformerModel> LoadFromLLMC(const std::string &filepath);
// Validate the external config and layout before constructing modules or loading weights.
// A null layout selects the legacy model path. Mismatches throw std::invalid_argument.
std::shared_ptr<infini_train::nn::TransformerModel>
LoadFromLLMC(const std::string &filepath, const infini_train::nn::TransformerConfig &expected_config,
             std::shared_ptr<const infini_train::nn::parallel::PipelineLayout> pipeline_layout = nullptr);

} // namespace gpt2
