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
    std::vector<std::pair<int, int>> layer_chunks;
};

class PipelineParallel : public Module {
public:
    PipelineParallel(const std::shared_ptr<nn::Module> module, int num_stages, int num_micro_batches,
                     const std::vector<std::vector<int64_t>> &recv_shape, int rank,
                     const std::shared_ptr<Optimizer> &optimizer, int device_id);

    float TrainStep(const std::vector<std::shared_ptr<Tensor>> &input,
                    const std::vector<std::shared_ptr<Tensor>> &target, const std::shared_ptr<nn::Module> &loss_fn,
                    DataType dtype);

    static StageInfo GetStageInfo(int total_layers, int pp_size, int chunks_per_stage = 1);

private:
    int num_stages_ = -1;
    int rank_ = -1;
    std::shared_ptr<PipelineStage> pipeline_stage_ = nullptr;
    std::shared_ptr<PipelineSchedule> schedule_ = nullptr;

    void BuildPipelineStage(const std::shared_ptr<nn::Module> &model, const std::shared_ptr<Optimizer> &optimizer,
                            const std::vector<std::vector<int64_t>> &recv_shape, int device_id);

    void SetupSchedule(int num_micro_batches);
};

} // namespace infini_train::nn::parallel
