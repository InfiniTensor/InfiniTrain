#pragma once
#include <cstdint>
#include <span>
#include <vector>

namespace infini_train::nn::parallel {
using LayerIndex = std::int64_t;

struct LayerRange {
    LayerIndex begin = 0;
    LayerIndex end = 0;
    LayerIndex Size() const;
    bool Contains(LayerIndex layer_id) const;
};
struct PipelineChunkLayout {
    int local_chunk_idx = -1;
    LayerRange layer_range;
    int global_chunk_id = -1;
    int stage_id = -1;
};
struct PipelineChunkSpec {
    int stage_id;
    LayerIndex layer_count;
};
struct PipelineStageLayout {
    int stage_id = -1;
    std::vector<PipelineChunkLayout> chunks;
    bool has_embedding = false;
    bool has_final_norm = false;
    bool has_lm_head = false;
};
class PipelineLayout {
public:
    PipelineLayout() = delete;
    static PipelineLayout BuildUniformLayout(LayerIndex num_layers, int num_stages, int chunks_per_stage = 1);
    static PipelineLayout BuildCustomLayout(LayerIndex num_layers, int num_stages, std::span<const LayerIndex> counts,
                                            int chunks_per_stage = 1);
    static PipelineLayout BuildChunkLayout(LayerIndex num_layers, int num_stages,
                                           std::span<const PipelineChunkSpec> chunk_specs);

    const PipelineStageLayout &GetStage(int stage_id) const;
    const PipelineChunkLayout &GetChunk(int global_chunk_id) const;

    int GetStageForLayer(LayerIndex global_layer_id) const;
    LayerIndex GetLocalLayerIndex(int stage_id, LayerIndex global_layer_id) const;

    int GetNumChunks() const;
    int GetInputStage() const;
    int GetOutputStage() const;
    int GetMaxLocalChunks() const;
    LayerIndex GetNumLayers() const;
    int GetNumStages() const;
    bool IsCustom() const;

private:
    PipelineLayout(LayerIndex num_layers, int num_stages, bool is_custom, std::vector<PipelineStageLayout> stages);
    LayerIndex num_layers_;
    int num_stages_;
    bool is_custom_;
    std::vector<PipelineStageLayout> stages_;
};
} // namespace infini_train::nn::parallel
