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

// Parse a comma-separated list of non-negative per-layer compute costs ("1.0,2.0,1.5").
// These feed `SuggestBalancedPartition` as the `layer_costs` argument. Returns an empty
// vector when `str` is empty, meaning "no user-provided costs".
std::vector<double> ParsePipelineLayerCosts(const std::string &str);

// Suggest a contiguous per-stage layer partition that approximately minimizes the maximum
// per-stage cost (the classic linear partition problem). `layer_costs[i]` is the cost of
// layer i (e.g. parameter count, measured compute time, or any user-provided cost); when
// empty, every layer is treated as unit cost, i.e. balanced by layer count. Returns
// `num_stages` positive counts that sum to `total_layers`, suitable as the `partition`
// argument of PipelineLayout::Create.
std::vector<int> SuggestBalancedPartition(int total_layers, int num_stages,
                                          const std::vector<double> &layer_costs = {});

// Predicted load / bubble / throughput analysis for a contiguous per-stage partition. This is
// a pure function used to compare layouts (e.g. default uniform vs. a cost-balanced layout)
// without running the model: per-stage time is assumed proportional to the sum of the per-layer
// costs assigned to that stage.
struct PipelineLoadStats {
    int num_stages = 0;
    int num_micro_batches = 1;
    std::vector<double> stage_loads; // predicted per-stage time (sum of assigned layer costs)
    double bottleneck = 0.0;         // max stage load (sets the pipeline clock period)
    double average = 0.0;            // mean stage load
    double efficiency = 0.0;         // average / bottleneck == 1 - imbalance_bubble
    double imbalance_bubble = 0.0;   // 1 - average / bottleneck, idle time caused by load skew
    double structural_bubble = 0.0;  // (S-1)/(S-1+n), GPipe fill/drain overhead
};

// Compute the load analysis for `partition` (per-stage layer counts summing to `total_layers`;
// when empty, fall back to the default uniform partition). `layer_costs[i]` is the compute cost
// of layer i; when empty, every layer has unit cost (load == layer count). `num_micro_batches`
// only affects the structural fill/drain bubble.
PipelineLoadStats ComputePipelineLoadAnalysis(int total_layers, int num_stages, const std::vector<int> &partition,
                                              const std::vector<double> &layer_costs = {}, int num_micro_batches = 1);

} // namespace infini_train::nn::parallel
