#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/datatype.h"

namespace infini_train {
class Tensor;
class Optimizer;
namespace nn {
class Module;
}
} // namespace infini_train

namespace infini_train::nn::parallel {

class PipelineStage;

class PipelineSchedule {
public:
    PipelineSchedule(std::shared_ptr<PipelineStage> stage, int num_stages, int num_micro_batches)
        : stage_(std::move(stage)), num_micro_batches_(num_micro_batches) {}

    virtual ~PipelineSchedule() = default;

    float Step(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target,
               const std::shared_ptr<Optimizer> &optimizer, const std::shared_ptr<nn::Module> &loss_fn, DataType dtype);

    virtual float StepMicroBatches(const std::vector<std::shared_ptr<Tensor>> &arg_mbs,
                                   const std::vector<std::shared_ptr<Tensor>> &target_mbs,
                                   const std::shared_ptr<nn::Module> &loss_fn, DataType dtype);

    std::vector<std::shared_ptr<Tensor>> ReceiveFromPrev(int peer_rank);
    std::vector<std::shared_ptr<Tensor>> SendToNext(const std::vector<std::shared_ptr<Tensor>> &tensors, int peer_rank);

protected:
    int num_micro_batches_ = -1;
    std::shared_ptr<PipelineStage> stage_ = nullptr;
};

class PipelineParallelScheduler {
public:
    struct Task {
        int step;
        int microbatch_id;
        int global_chunk_id;
        int local_chunk_idx;
        bool is_forward;
        int stage_id;
        bool is_first_chunk;
        bool is_last_chunk;
    };

    static Task CreateTask(int step, int mb, int global_chunk, int num_stages, int total_chunks, bool is_forward);

    static std::vector<Task> GenerateGPipeSchedule(int n, int num_stages, int vpp_size);

    static std::vector<Task> GenerateInterleaved1F1BSchedule(int n, int num_stages, int vpp_size);
};

} // namespace infini_train::nn::parallel
