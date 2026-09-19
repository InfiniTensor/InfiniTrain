#pragma once

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "infini_train/include/nn/parallel/pp/pipeline_layout.h"

namespace infini_train::examples {

using PipelineLayoutRequest
    = std::variant<std::monostate, std::vector<nn::parallel::LayerIndex>, std::vector<nn::parallel::PipelineChunkSpec>>;

PipelineLayoutRequest ParsePipelineLayoutRequest(std::string_view partition, std::string_view chunks, int pp_size,
                                                 int vpp_size);

bool IsExplicitChunkRequest(const PipelineLayoutRequest &request);

std::shared_ptr<const nn::parallel::PipelineLayout> ResolvePipelineLayout(nn::parallel::LayerIndex num_layers,
                                                                          int pp_size, int vpp_size,
                                                                          const PipelineLayoutRequest &request);

std::vector<nn::parallel::LayerIndex> ParsePipelineLayerPartition(std::string_view value);

std::shared_ptr<const nn::parallel::PipelineLayout>
BuildPipelineLayoutFromCLI(nn::parallel::LayerIndex num_layers, int num_stages,
                           std::span<const nn::parallel::LayerIndex> counts, int chunks_per_stage = 1);

std::string FormatPipelineLayout(const nn::parallel::PipelineLayout &layout);

} // namespace infini_train::examples
