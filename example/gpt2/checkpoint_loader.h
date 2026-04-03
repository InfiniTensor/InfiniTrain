#pragma once

#include <memory>
#include <string>

namespace infini_train::nn {
class TransformerModel;
enum class ModelType;
} // namespace infini_train::nn

namespace gpt2 {
int GetChunkSize();
std::shared_ptr<infini_train::nn::TransformerModel> LoadFromLLMC(const std::string &filepath);
std::shared_ptr<infini_train::nn::TransformerModel> FromPretrained(infini_train::nn::ModelType model_type);
} // namespace gpt2
