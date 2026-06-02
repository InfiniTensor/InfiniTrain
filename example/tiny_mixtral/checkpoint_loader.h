#pragma once

#include <memory>
#include <string>

#include "infini_train/include/nn/modules/transformer/transformer_config.h"

namespace infini_train::nn {
class TransformerModel;
} // namespace infini_train::nn

namespace tiny_mixtral {

infini_train::nn::TransformerConfig ConfigFromLLMC(const std::string &filepath);

void CheckLLMCConfig(const std::string &filepath, const infini_train::nn::TransformerConfig &expected_config);

std::shared_ptr<infini_train::nn::TransformerModel>
LoadFromLLMC(const std::string &filepath, const infini_train::nn::TransformerConfig &expected_config);

} // namespace tiny_mixtral
