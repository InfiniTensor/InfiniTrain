#pragma once

#include "example/common/parser.h"

#include <memory>
#include <string>

namespace infini_train::nn {
class TransformerModel;
} // namespace infini_train::nn

namespace llama3 {
std::shared_ptr<infini_train::nn::TransformerModel> LoadFromLLMC(const std::string &filepath);
std::shared_ptr<infini_train::nn::TransformerModel>
LoadFromLLMC(const std::string &filepath, const infini_train::examples::PipelineLayoutRequest &layout_request);

} // namespace llama3
