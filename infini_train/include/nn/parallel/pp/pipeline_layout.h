#pragma once

#include <string>
#include <utility>
#include <vector>

namespace infini_train::nn::parallel {

// Describes which layers (and special modules) belong to a single pipeline stage.
struct StageInfo {
    bool is_first_stage = false; // this stage owns the Embedding (first-stage module)
    bool is_last_stage = false;  // this stage owns the Final Norm and LM Head
    // Layer index ranges assigned to this stage, one per (virtual) chunk:
    // (inclusive_start_layer, exclusive_end_layer).
    std::vector<std::pair<int, int>> layer_ranges_per_chunk;
};

// Unified source of truth for the pipeline layer partition. Model construction,
// pipeline stage wrapping and parameter loading all query the same layout so the
// layer-ownership logic is not duplicated across modules.
class PipelineLayout {
public:
    // Build a layout. `partition` holds the number of transformer layers per stage
    // (e.g. {4, 8, 6, 6}); when empty, fall back to the default uniform partition.
    static PipelineLayout Create(int total_layers, int num_stages, int vpp_size,
                                 const std::vector<int> &partition = {});

    StageInfo GetStageInfo(int stage_id) const;
    int StageOfLayer(int layer_id) const;
    bool OwnsLayer(int stage_id, int layer_id) const;

    // Round-robin chunk -> stage / local-chunk mapping used by the pipeline scheduler.
    static int StageOfChunk(int global_chunk_id, int num_stages);
    static int LocalChunkIndexOfChunk(int global_chunk_id, int num_stages);

    std::string Describe() const;

private:
    int num_stages_ = 1;
    int total_layers_ = 0;
    int vpp_size_ = 1;
    int first_stage_idx_ = 0; // stage owning the Embedding
    int last_stage_idx_ = 0;  // stage owning the Final Norm + LM Head
    std::vector<std::vector<std::pair<int, int>>> stage_layer_ranges_;
    std::vector<int> layer_to_stage_;
};

// Parse a comma-separated per-stage layer count string ("4,8,6,6"). Returns an empty
// vector when `str` is empty, meaning "use the default uniform partition".
std::vector<int> ParsePipelineLayerPartition(const std::string &str);

} // namespace infini_train::nn::parallel
