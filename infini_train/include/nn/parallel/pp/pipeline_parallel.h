// pipeline_parallel.h
#pragma once

#include <memory>
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

class PipelineParallel : public Module {
public:
    PipelineParallel(const std::shared_ptr<nn::Module> module, int num_stages, int num_micro_batches,
                     const std::vector<std::vector<int64_t>> &recv_shape, int rank, int device_id, int vpp);

    float TrainStep(const std::vector<std::shared_ptr<Tensor>> &input,
                    const std::vector<std::shared_ptr<Tensor>> &target, const std::shared_ptr<Optimizer> &optimizer,
                    const std::shared_ptr<nn::Module> &loss_fn, DataType dtype) override;

    static StageInfo GetStageInfo(int total_layers, int pp_size, int pp_rank, int chunks_per_stage = 1);

    std::vector<std::shared_ptr<Module>> *mutable_chunks();

private:
    void BuildPipelineStage(const std::vector<std::vector<int64_t>> &recv_shape, int device_id,
                            std::vector<std::shared_ptr<Module>> &&chunks);

    void SetupSchedule(int num_micro_batches);

    int num_stages_ = -1;
    int rank_ = -1;
    std::shared_ptr<PipelineSchedule> schedule_ = nullptr;
    std::shared_ptr<PipelineStage> pipeline_stage_ = nullptr;
};
} // namespace infini_train::nn::parallel
