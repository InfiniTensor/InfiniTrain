#pragma once

#include <stdexcept>
#include <string>
#include <vector>

namespace infini_train::nn::parallel {

class PipelineLayoutError : public std::runtime_error {
public:
    explicit PipelineLayoutError(const std::string& msg);
};

// Represent the three special modules explicitly.
enum class SpecialModule {
    kEmbedding,
    kFinalNorm,
    kLMHead,
};

struct LayerRange {
    int start = -1;
    int end = -1;

    int size() const;
    bool contains(int layer_id) const;
};

struct SpecialModulePlacement {
    int embedding_stage = -1;
    int final_norm_stage = -1;
    int lm_head_stage = -1;
};

struct ChunkLayout {
    int global_chunk_id = -1; // Global chunk identifier.
    int stage_id = -1;
    int local_chunk_id = -1; // Local chunk index within a stage.
    LayerRange layers;
};

struct StageLayout {
    int stage_id = -1;
    std::vector<int> global_chunk_ids;
};

struct LayerLocation {
    int stage_id = -1;
    int global_chunk_id = -1;
    int local_chunk_id = -1;
    int index_in_chunk = -1;
    int flat_local_layer_index = -1;
};


// Optional validation policies for layout construction.
struct PipelineLayoutPolicy {
    bool allow_empty_stages = false;
    bool require_contiguous_execution = true;
    bool require_boundary_special_modules = true;
};

class PipelineLayout {
public:
    PipelineLayout();

    // Build the default layout with total_chunks = num_stages * vpp_size.
    static PipelineLayout BuildDefault(
        int num_layers,
        int num_stages,
        int vpp_size,
        SpecialModulePlacement placement = {},
        PipelineLayoutPolicy policy = {}
    );

    // Build a user-defined contiguous layout.
    static PipelineLayout BuildContiguous(
        const std::vector<int>& stage_layer_counts,
        SpecialModulePlacement placement = {},
        PipelineLayoutPolicy policy = {}
    );

    int num_layers() const;
    int num_stages() const;
    int vpp_size() const;

    const StageLayout& stage(int stage_id) const;
    const ChunkLayout& chunk(int global_chunk_id) const;
    const std::vector<ChunkLayout>& chunks() const;

    const LayerLocation& locate_layer(int layer_id) const;
    int stage_of_layer(int layer_id) const;
    int local_layer_index(int stage_id, int layer_id) const;
    const ChunkLayout& chunk_of_layer(int layer_id) const;

    bool owns(SpecialModule module, int stage_id) const;
    const SpecialModulePlacement& special_modules() const;
    const PipelineLayoutPolicy& policy() const;

    std::string ToString() const;
    void Validate() const;
    void ValidateForCurrentPipelineTransport() const;

    static std::vector<int> ParseLayerPartition(const std::string &value);

    // Parse a Megatron-style layout expression.  Stages are separated by '|',
    // chunks within a stage by ',', 't' denotes one transformer layer and E/F/H
    // denote embedding/final-norm/LM-head ownership respectively. Parenthesized
    // expressions may be repeated with '*N'.
    static PipelineLayout ParseMegatronStyleLayout(
        const std::string& value,
        int num_layers,
        int pp_size,
        int vpp_size = 1,
        SpecialModulePlacement placement = {},
        PipelineLayoutPolicy policy = {});

    // Suggest a contiguous layer partition that approximately balances the
    // supplied per-layer costs.  The returned vector has pp_size entries and
    // sums to num_layers.
    static std::vector<int> SuggestBalancedPartition(
        int num_layers,
        int pp_size,
        const std::vector<double>& layer_costs = {});

    static PipelineLayout BuildPipelineLayout(
        int num_layers,
        int pp_size,
        int vpp_size,
        const std::string &layer_partition,
        SpecialModulePlacement placement = {},
        PipelineLayoutPolicy policy = {});

private:
    PipelineLayout(
        int num_layers,
        int num_stages,
        int vpp_size,
        std::vector<ChunkLayout> chunks,
        SpecialModulePlacement placement,
        PipelineLayoutPolicy policy);

    static SpecialModulePlacement BuildPlacement(
        SpecialModulePlacement placement,
        int num_stages);

    void BuildIndexes();

    int num_layers_ = 0;
    int num_stages_ = 0;
    int vpp_size_ = 0;
    std::vector<ChunkLayout> chunks_;
    std::vector<StageLayout> stages_;
    SpecialModulePlacement placement_;
    PipelineLayoutPolicy policy_;
    std::vector<LayerLocation> layer_locations_;
};

} // namespace infini_train::nn::parallel
