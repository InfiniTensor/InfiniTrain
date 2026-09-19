#pragma once

#include "example/common/parser.h"

#include <memory>
#include <string>

namespace infini_train::nn {
class TransformerModel;
} // namespace infini_train::nn

namespace gpt2 {
std::shared_ptr<infini_train::nn::TransformerModel> LoadFromLLMC(const std::string &filepath);
std::shared_ptr<infini_train::nn::TransformerModel>
LoadFromLLMC(const std::string &filepath, const infini_train::examples::PipelineLayoutRequest &layout_request);

} // namespace gpt2
