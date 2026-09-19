#include "example/common/parser.h"

#include <charconv>
#include <cstddef>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace infini_train::examples {

using LayerIndex = nn::parallel::LayerIndex;
using PipelineLayout = nn::parallel::PipelineLayout;
namespace {
std::invalid_argument InvalidPartition(std::string_view value, std::string_view reason) {
    return std::invalid_argument("Invalid --pipeline_layer_partition=\"" + std::string(value)
                                 + "\": " + std::string(reason));
}

} // namespace
std::vector<LayerIndex> ParsePipelineLayerPartition(std::string_view value) {
    if (value.empty()) {
        return {};
    }

    std::vector<LayerIndex> result;
    std::size_t begin = 0;
    std::size_t item_index = 0;

    while (begin <= value.size()) {
        const std::size_t comma = value.find(',', begin);
        const std::size_t end = comma == std::string_view::npos ? value.size() : comma;
        std::string_view token = value.substr(begin, end - begin);

        if (token.empty()) {
            throw InvalidPartition(value, "item " + std::to_string(item_index) + " is empty");
        }

        for (char c : token) {
            if (c < '0' || c > '9') {
                throw InvalidPartition(value,
                                       "item " + std::to_string(item_index) + " must contain decimal digits only");
            }
        }
        LayerIndex count = 0;
        const auto [ptr, error] = std::from_chars(token.data(), token.data() + token.size(), count);

        if (error == std::errc::result_out_of_range) {
            throw InvalidPartition(value, "item " + std::to_string(item_index) + " is outside the LayerIndex range");
        }
        if (error != std::errc{} || ptr != token.data() + token.size()) {
            throw InvalidPartition(value, "item " + std::to_string(item_index)
                                              + " must be a base-10 integer without trailing characters");
        }
        if (count <= 0) {
            throw InvalidPartition(value, "item " + std::to_string(item_index) + " must be greater than zero");
        }

        result.push_back(count);
        if (comma == std::string_view::npos) {
            break;
        }
        begin = comma + 1;
        ++item_index;
    }

    return result;
}

std::shared_ptr<const PipelineLayout> BuildPipelineLayoutFromCLI(LayerIndex num_layers, int num_stages,
                                                                 std::span<const LayerIndex> counts,
                                                                 int chunks_per_stage) {
    if (num_stages <= 0 || chunks_per_stage <= 0
        || static_cast<LayerIndex>(num_stages) * chunks_per_stage > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("PP/vPP must be positive and their product must fit int");
    }
    if (counts.empty() && num_layers >= 0
        && num_layers < static_cast<LayerIndex>(num_stages) * chunks_per_stage) {
        return nullptr;
    }
    PipelineLayout layout = counts.empty()
                              ? PipelineLayout::BuildUniformLayout(num_layers, num_stages, chunks_per_stage)
                              : PipelineLayout::BuildCustomLayout(num_layers, num_stages, counts, chunks_per_stage);
    return std::make_shared<const PipelineLayout>(std::move(layout));
}

bool IsExplicitChunkRequest(const PipelineLayoutRequest &request) {
    return std::holds_alternative<std::vector<nn::parallel::PipelineChunkSpec>>(request);
}

PipelineLayoutRequest ParsePipelineLayoutRequest(std::string_view partition, std::string_view chunks, int pp_size,
                                                 int vpp_size) {
    if (pp_size <= 0 || vpp_size <= 0
        || static_cast<LayerIndex>(pp_size) * vpp_size > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("PP/vPP must be positive and their product must fit int");
    }
    if (!partition.empty() && !chunks.empty()) {
        throw std::invalid_argument("pipeline_layer_partition and pipeline_chunk_layout are mutually exclusive");
    }
    if (!chunks.empty()) {
        std::vector<nn::parallel::PipelineChunkSpec> specs;
        std::size_t begin = 0;
        while (begin <= chunks.size()) {
            const auto comma = chunks.find(',', begin);
            const auto token
                = chunks.substr(begin, comma == std::string_view::npos ? chunks.size() - begin : comma - begin);
            const auto colon = token.find(':');
            if (colon == std::string_view::npos || token.find(':', colon + 1) != std::string_view::npos) {
                throw std::invalid_argument("pipeline_chunk_layout entry " + std::to_string(specs.size())
                                            + " must be stage:count: " + std::string(token));
            }
            const auto owner_text = token.substr(0, colon);
            const auto count_text = token.substr(colon + 1);
            const auto digits = [](std::string_view value) {
                return !value.empty() && value.find_first_not_of("0123456789") == std::string_view::npos;
            };
            int owner = -1;
            LayerIndex count = -1;
            if (!digits(owner_text) || !digits(count_text)) {
                throw std::invalid_argument("invalid pipeline_chunk_layout entry " + std::to_string(specs.size()) + ": "
                                            + std::string(token));
            }
            const auto a = std::from_chars(owner_text.data(), owner_text.data() + owner_text.size(), owner);
            const auto b = std::from_chars(count_text.data(), count_text.data() + count_text.size(), count);
            if (a.ec != std::errc{} || b.ec != std::errc{} || owner < 0 || owner >= pp_size || count <= 0) {
                throw std::invalid_argument("pipeline_chunk_layout entry " + std::to_string(specs.size())
                                            + " has invalid owner/count: " + std::string(token));
            }
            specs.push_back({owner, count});
            if (specs.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                throw std::invalid_argument("too many explicit chunks");
            }
            if (comma == std::string_view::npos) {
                break;
            }
            begin = comma + 1;
        }
        std::vector<int> local_counts(static_cast<std::size_t>(pp_size));
        for (const auto &spec : specs) { ++local_counts[spec.stage_id]; }
        int max_count = 0;
        for (int count : local_counts) {
            if (count == 0) {
                throw std::invalid_argument("each stage must own a chunk");
            }
            if (count > max_count) {
                max_count = count;
            }
        }
        if (max_count != vpp_size) {
            throw std::invalid_argument("explicit layout max local chunk count=" + std::to_string(max_count)
                                        + ", virtual_pipeline_parallel=" + std::to_string(vpp_size));
        }
        return specs;
    }
    if (partition.empty()) {
        return std::monostate{};
    }
    auto counts = ParsePipelineLayerPartition(partition);
    if (counts.size() != static_cast<std::size_t>(pp_size) * vpp_size) {
        throw std::invalid_argument("pipeline_layer_partition needs PP * vPP entries in global chunk order");
    }
    return counts;
}

std::shared_ptr<const PipelineLayout> ResolvePipelineLayout(LayerIndex num_layers, int pp_size, int vpp_size,
                                                            const PipelineLayoutRequest &request) {
    if (pp_size <= 0 || vpp_size <= 0
        || static_cast<LayerIndex>(pp_size) * vpp_size > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("PP/vPP must be positive and their product must fit int");
    }
    if (const auto *specs = std::get_if<std::vector<nn::parallel::PipelineChunkSpec>>(&request)) {
        auto layout = PipelineLayout::BuildChunkLayout(num_layers, pp_size, *specs);
        if (layout.GetMaxLocalChunks() != vpp_size) {
            throw std::invalid_argument("explicit layout does not match virtual_pipeline_parallel");
        }
        return std::make_shared<const PipelineLayout>(std::move(layout));
    }
    if (const auto *counts = std::get_if<std::vector<LayerIndex>>(&request)) {
        if (counts->empty()) {
            throw std::invalid_argument("custom partition must not be empty");
        }
        return BuildPipelineLayoutFromCLI(num_layers, pp_size, *counts, vpp_size);
    }
    return BuildPipelineLayoutFromCLI(num_layers, pp_size, {}, vpp_size);
}

std::string FormatPipelineLayout(const PipelineLayout &layout) {
    std::ostringstream output;
    output << (layout.IsCustom() ? "custom" : "uniform") << " pipeline layout:";

    for (int stage_id = 0; stage_id < layout.GetNumStages(); ++stage_id) {
        const auto &stage = layout.GetStage(stage_id);
        output << "\n  Stage " << stage_id << ":";
        for (const auto &chunk : stage.chunks) {
            output << " chunk " << chunk.global_chunk_id << " (local " << chunk.local_chunk_idx << ") layers ["
                   << chunk.layer_range.begin << "," << chunk.layer_range.end << ")";
        }
        if (stage.has_embedding) {
            output << " embedding";
        }
        if (stage.has_final_norm) {
            output << " final_norm";
        }
        if (stage.has_lm_head) {
            output << " lm_head";
        }
    }
    return output.str();
}

} // namespace infini_train::examples
