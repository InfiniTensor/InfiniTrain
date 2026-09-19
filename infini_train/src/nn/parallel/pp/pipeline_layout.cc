#include "infini_train/include/nn/parallel/pp/pipeline_layout.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace infini_train::nn::parallel {
namespace {
void ValidateDimensions(LayerIndex num_layers, int num_stages) {
    if (num_layers <= 0) {
        throw std::invalid_argument("num_layers must be positive");
    }
    if (num_stages <= 0) {
        throw std::invalid_argument("num_stages must be positive");
    }
    if (num_layers < num_stages) {
        throw std::invalid_argument("num_layers must be at least num_stages");
    }
}

std::string FormatLayerCounts(std::span<const LayerIndex> layers_per_stage) {
    std::ostringstream output;
    output << "[";
    for (std::size_t index = 0; index < layers_per_stage.size(); ++index) {
        if (index != 0) {
            output << ",";
        }
        output << layers_per_stage[index];
    }
    output << "]";
    return output.str();
}
} // namespace

LayerIndex LayerRange::Size() const { return end - begin; }

bool LayerRange::Contains(LayerIndex layer_id) const { return layer_id >= begin && layer_id < end; }

PipelineLayout::PipelineLayout(LayerIndex num_layers, int num_stages, bool is_custom,
                               std::vector<PipelineStageLayout> stages)
    : num_layers_(num_layers), num_stages_(num_stages), is_custom_(is_custom), stages_(std::move(stages)) {}

PipelineLayout PipelineLayout::BuildUniformLayout(LayerIndex num_layers, int num_stages, int chunks_per_stage) {
    // Do not call ValidateDimensions(): custom layouts require positive layers and
    // at least one layer per stage. The legacy GetStageInfo() adapter also needs
    // zero/underfilled default metadata; this does not imply empty-stage training support.
    if (num_layers < 0 || num_stages <= 0 || chunks_per_stage <= 0) {
        throw std::invalid_argument("uniform layout requires non-negative layers and positive PP/vPP");
    }

    const LayerIndex total_chunks = static_cast<LayerIndex>(num_stages) * chunks_per_stage;
    if (total_chunks > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("uniform layout chunk count must fit int");
    }

    const LayerIndex quotient = num_layers / total_chunks;
    const LayerIndex remainder = num_layers % total_chunks;
    std::vector<PipelineStageLayout> stages(static_cast<std::size_t>(num_stages));
    for (int stage_id = 0; stage_id < num_stages; ++stage_id) {
        auto &stage = stages[stage_id];
        stage.stage_id = stage_id;
        stage.has_embedding = (stage_id == 0);
        stage.has_final_norm = (stage_id == num_stages - 1);
        stage.has_lm_head = (stage_id == num_stages - 1);
        for (int local_chunk_id = 0; local_chunk_id < chunks_per_stage; ++local_chunk_id) {
            const int gid = local_chunk_id * num_stages + stage_id;
            const LayerIndex begin
                = static_cast<LayerIndex>(gid) * quotient
                + (static_cast<LayerIndex>(gid) > remainder ? remainder : static_cast<LayerIndex>(gid));
            const LayerIndex end = begin + quotient + (gid < remainder ? 1 : 0);

            if (begin == end) {
                continue;
            }
            stage.chunks.push_back(PipelineChunkLayout{local_chunk_id, LayerRange{begin, end}, gid, stage_id});
        }
    }

    return PipelineLayout(num_layers, num_stages, false, std::move(stages));
}

PipelineLayout PipelineLayout::BuildCustomLayout(LayerIndex num_layers, int num_stages,
                                                 std::span<const LayerIndex> counts, int chunks_per_stage) {
    ValidateDimensions(num_layers, num_stages);
    const LayerIndex total_chunks = static_cast<LayerIndex>(num_stages) * chunks_per_stage;
    if (chunks_per_stage <= 0 || total_chunks > std::numeric_limits<int>::max()
        || counts.size() != static_cast<std::size_t>(total_chunks)) {
        throw std::invalid_argument("custom layer count must match num_stages * chunks_per_stage");
    }
    LayerIndex actual_sum = 0;
    for (LayerIndex count : counts) {
        if (count <= 0) {
            throw std::invalid_argument("custom layer counts must be positive");
        }
        if (count > std::numeric_limits<LayerIndex>::max() - actual_sum) {
            throw std::invalid_argument("custom layer count sum overflow");
        }
        actual_sum += count;
    }
    if (actual_sum != num_layers) {
        std::ostringstream msg;
        msg << "Invalid pipeline layer counts: received=" << FormatLayerCounts(counts) << ", num_stages=" << num_stages
            << ", chunks_per_stage=" << chunks_per_stage << ", expected_sum=" << num_layers << ", actual_sum="
            << actual_sum;
        throw std::invalid_argument(msg.str());
    }

    std::vector<PipelineStageLayout> stages(static_cast<std::size_t>(num_stages));
    for (int stage_id = 0; stage_id < num_stages; ++stage_id) {
        stages[stage_id].stage_id = stage_id;
        stages[stage_id].has_embedding = (stage_id == 0);
        stages[stage_id].has_final_norm = (stage_id == num_stages - 1);
        stages[stage_id].has_lm_head = (stage_id == num_stages - 1);
    }
    LayerIndex begin = 0;
    for (int gid = 0; gid < static_cast<int>(total_chunks); ++gid) {
        const int stage_id = gid % num_stages;
        const int local_chunk_id = gid / num_stages;
        const LayerIndex end = begin + counts[static_cast<std::size_t>(gid)];
        stages[stage_id].chunks.push_back(PipelineChunkLayout{local_chunk_id, LayerRange{begin, end}, gid, stage_id});
        begin = end;
    }
    return PipelineLayout(num_layers, num_stages, true, std::move(stages));
}

PipelineLayout PipelineLayout::BuildChunkLayout(LayerIndex num_layers, int num_stages,
                                                std::span<const PipelineChunkSpec> specs) {
    ValidateDimensions(num_layers, num_stages);
    if (specs.empty() || specs.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("invalid explicit chunk count");
    }
    std::vector<PipelineStageLayout> stages(static_cast<std::size_t>(num_stages));
    for (int id = 0; id < num_stages; ++id) { stages[id].stage_id = id; }
    LayerIndex begin = 0;
    int global_id = 0;
    for (const auto &spec : specs) {
        if (spec.stage_id < 0 || spec.stage_id >= num_stages || spec.layer_count <= 0
            || spec.layer_count > num_layers - begin) {
            throw std::invalid_argument("invalid pipeline_chunk_layout entry " + std::to_string(global_id)
                                        + ": stage_id=" + std::to_string(spec.stage_id) + ", count="
                                        + std::to_string(spec.layer_count) + ", stages=" + std::to_string(num_stages)
                                        + ", remaining_layers=" + std::to_string(num_layers - begin));
        }
        auto &stage = stages[spec.stage_id];
        const auto end = begin + spec.layer_count;
        stage.chunks.push_back(PipelineChunkLayout{static_cast<int>(stage.chunks.size()), LayerRange{begin, end},
                                                   global_id++, spec.stage_id});
        begin = end;
    }
    if (begin != num_layers) {
        throw std::invalid_argument("explicit chunk layer sum=" + std::to_string(begin)
                                    + ", expected=" + std::to_string(num_layers));
    }
    for (const auto &stage : stages) {
        if (stage.chunks.empty()) {
            throw std::invalid_argument("stage " + std::to_string(stage.stage_id) + " owns no chunk");
        }
    }
    stages[specs.front().stage_id].has_embedding = true;
    stages[specs.back().stage_id].has_final_norm = true;
    stages[specs.back().stage_id].has_lm_head = true;
    return PipelineLayout(num_layers, num_stages, true, std::move(stages));
}

const PipelineStageLayout &PipelineLayout::GetStage(int stage_id) const {
    if (stage_id < 0 || stage_id >= num_stages_) {
        throw std::out_of_range("stage_id is out of range");
    }
    return stages_[static_cast<std::size_t>(stage_id)];
}

const PipelineChunkLayout &PipelineLayout::GetChunk(int global_chunk_id) const {
    if (global_chunk_id < 0) {
        throw std::out_of_range("global_chunk_id is out of range");
    }
    for (const auto &stage : stages_) {
        for (const auto &chunk : stage.chunks) {
            if (chunk.global_chunk_id == global_chunk_id) {
                return chunk;
            }
        }
    }
    throw std::out_of_range("global_chunk_id is out of range");
}

int PipelineLayout::GetStageForLayer(LayerIndex layer_id) const {
    if (layer_id < 0 || layer_id >= num_layers_) {
        throw std::out_of_range("layer_id is out of range");
    }

    for (const PipelineStageLayout &stage : stages_) {
        for (const auto &chunk : stage.chunks) {
            if (chunk.layer_range.Contains(layer_id)) {
                return stage.stage_id;
            }
        }
    }

    throw std::out_of_range("layer_id is not assigned to a pipeline stage");
}

LayerIndex PipelineLayout::GetLocalLayerIndex(int stage_id, LayerIndex global_layer_id) const {
    const PipelineStageLayout &stage = GetStage(stage_id);

    if (global_layer_id < 0 || global_layer_id >= num_layers_) {
        throw std::out_of_range("layer_id is out of range");
    }

    LayerIndex offset = 0;
    for (const auto &chunk : stage.chunks) {
        if (chunk.layer_range.Contains(global_layer_id)) {
            return offset + global_layer_id - chunk.layer_range.begin;
        }
        offset += chunk.layer_range.Size();
    }
    throw std::out_of_range("layer_id does not belong to the requested stage");
}

int PipelineLayout::GetNumChunks() const {
    int total = 0;
    for (const auto &stage : stages_) { total += static_cast<int>(stage.chunks.size()); }
    return total;
}

int PipelineLayout::GetMaxLocalChunks() const {
    int count = 0;
    for (const auto &stage : stages_) { count = std::max(count, static_cast<int>(stage.chunks.size())); }
    return count;
}

int PipelineLayout::GetInputStage() const {
    for (const auto &stage : stages_) {
        if (stage.has_embedding) {
            return stage.stage_id;
        }
    }
    throw std::logic_error("layout has no input stage");
}

int PipelineLayout::GetOutputStage() const {
    for (const auto &stage : stages_) {
        if (stage.has_final_norm && stage.has_lm_head) {
            return stage.stage_id;
        }
    }
    throw std::logic_error("layout has no output stage");
}

LayerIndex PipelineLayout::GetNumLayers() const { return num_layers_; }

int PipelineLayout::GetNumStages() const { return num_stages_; }

bool PipelineLayout::IsCustom() const { return is_custom_; }

} // namespace infini_train::nn::parallel
