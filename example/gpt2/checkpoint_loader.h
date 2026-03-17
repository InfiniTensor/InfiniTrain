#pragma once

#include <memory>
#include <string>

namespace infini_train::nn {
class TransformerModel;
} // namespace infini_train::nn

namespace gpt2 {
std::shared_ptr<infini_train::nn::TransformerModel> LoadFromLLMC(const std::string &filepath);
void SaveAsLLMC(const std::shared_ptr<infini_train::nn::TransformerModel> &model, const std::string &filepath);
} // namespace gpt2
