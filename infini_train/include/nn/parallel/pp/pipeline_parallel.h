// pipeline_parallel.h
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "infini_train/include/nn/modules/module.h"

namespace infini_train {
class Tensor;
class Device;
class Optimizer;
} // namespace infini_train

namespace infini_train::nn::parallel {
class PipelineStage;
class PipelineSchedule;

extern thread_local int pp_rank;

struct StageInfo {
    bool is_first_stage;
    bool is_last_stage;

    // Layer index ranges for chunks assigned to this pipeline stage.
    // Each element is a pair: (inclusive_start_layer, exclusive_end_layer)
    std::vector<std::pair<int, int>> layer_ranges_per_chunk;
};

class PipelineLayout {
public:
    static PipelineLayout Uniform(int total_layers, int pp_size, int chunks_per_stage = 1);
    static PipelineLayout Parse(int total_layers, int pp_size, const std::string &partition, int chunks_per_stage = 1);
    static PipelineLayout FromLayerCosts(int total_layers, int pp_size, const std::string &layer_costs,
                                         int chunks_per_stage = 1);
    static PipelineLayout FromChunkLayout(int total_layers, int pp_size, const std::string &chunk_layout);
    static PipelineLayout FromMegatronLayout(int total_layers, int pp_size, const std::string &model_layout);

    int num_stages() const { return num_stages_; }
    int total_layers() const { return total_layers_; }
    int chunks_per_stage() const { return chunks_per_stage_; }
    int num_global_chunks() const { return static_cast<int>(chunk_stages_.size()); }
    int stage_for_chunk(int global_chunk) const;
    int local_chunk_index(int global_chunk) const;
    const std::pair<int, int> &chunk_range(int global_chunk) const;
    bool is_first_stage(int stage) const;
    bool is_last_stage(int stage) const;
    bool owns_embedding(int stage) const;
    bool owns_final_norm(int stage) const;
    bool owns_lm_head(int stage) const;
    const std::vector<std::pair<int, int>> &layer_ranges(int stage) const;
    int stage_for_layer(int layer) const;
    std::string ToString() const;

private:
    int total_layers_ = 0;
    int num_stages_ = 0;
    int chunks_per_stage_ = 0;
    int embedding_stage_ = 0;
    int final_norm_stage_ = 0;
    int lm_head_stage_ = 0;
    std::vector<std::vector<std::pair<int, int>>> ranges_;
    std::vector<int> chunk_stages_;
    std::vector<int> chunk_local_indices_;
    std::vector<std::pair<int, int>> chunk_ranges_;
};

PipelineLayout ResolvePipelineLayout(int total_layers, int pp_size, int chunks_per_stage,
                                     const std::string &partition, const std::string &layer_costs,
                                     const std::string &chunk_layout = "", const std::string &model_layout = "");

void SetPipelineLayout(std::optional<PipelineLayout> layout);
bool HasPipelineLayout();
const PipelineLayout &GetPipelineLayout();

class PipelineParallel : public Module {
public:
    PipelineParallel(const std::shared_ptr<nn::Module> module, int num_stages, int num_micro_batches,
                     const std::vector<std::vector<int64_t>> &recv_shape, int rank, Device device, int vpp);

    float TrainStep(const std::vector<std::shared_ptr<Tensor>> &input,
                    const std::vector<std::shared_ptr<Tensor>> &target, const std::shared_ptr<Optimizer> &optimizer,
                    const std::shared_ptr<nn::Module> &loss_fn, DataType dtype) override;

    static StageInfo GetStageInfo(int total_layers, int pp_size, int pp_rank, int chunks_per_stage = 1);

    std::vector<std::shared_ptr<Module>> *mutable_chunks();

private:
    void BuildPipelineStage(const std::vector<std::vector<int64_t>> &recv_shape, Device device,
                            std::vector<std::shared_ptr<Module>> &&chunks);

    void SetupSchedule(int num_micro_batches);

    int num_stages_ = -1;
    int rank_ = -1;
    std::shared_ptr<PipelineSchedule> schedule_ = nullptr;
    std::shared_ptr<PipelineStage> pipeline_stage_ = nullptr;
};
} // namespace infini_train::nn::parallel
