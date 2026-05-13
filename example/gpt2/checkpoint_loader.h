#pragma once

#include <memory>
#include <string>

#include "infini_train/include/nn/modules/transformer/transformer_config.h"

namespace infini_train::nn {
class TransformerModel;
} // namespace infini_train::nn

namespace gpt2 {
std::shared_ptr<infini_train::nn::TransformerModel> LoadFromLLMC(const std::string &filepath);
std::shared_ptr<infini_train::nn::TransformerModel>
LoadFromLLMC(const std::string &filepath, const infini_train::nn::TransformerConfig &runtime_config);
} // namespace gpt2
