#pragma once

#include <cstring>
#include <memory>
#include <string>

namespace infini_train::nn {
class TransformerModel;
}

namespace llama3 {
std::shared_ptr<infini_train::nn::TransformerModel> LoadFromLLMC(const std::string &filepath);
void SaveAsLLMC(const std::shared_ptr<infini_train::nn::TransformerModel> &model, const std::string &filepath);
} // namespace llama3
