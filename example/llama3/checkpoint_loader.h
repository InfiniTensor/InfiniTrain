#pragma once

#include <memory>
#include <string>

namespace infini_train::nn {
class TransformerModel;
} // namespace infini_train::nn

namespace llama3 {
std::shared_ptr<infini_train::nn::TransformerModel> LoadFromLLMC(const std::string &filepath,
                                                                 const std::string &pipeline_layer_partition,
                                                                 const std::string &pipeline_layer_costs,
                                                                 const std::string &pipeline_chunk_layout,
                                                                 const std::string &pipeline_model_layout);
} // namespace llama3
