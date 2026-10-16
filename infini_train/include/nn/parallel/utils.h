#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace infini_train::nn::parallel::comm {
enum class ReduceOpType : int8_t;
} // namespace infini_train::nn::parallel::comm

namespace infini_train {
class Tensor;

namespace nn {
class Module;

namespace parallel {
class ProcessGroup;
}
} // namespace nn
} // namespace infini_train

namespace infini_train::nn::parallel {
std::string GetDataParallelProcessGroupName(int global_rank);

std::string GetTensorParallelProcessGroupName(int global_rank);

std::string GetPipelineParallelProcessGroupName(int global_rank);

std::vector<int> GetDataParallelGroupRanks(int global_rank);

std::vector<int> GetTensorParallelGroupRanks(int global_rank);

std::vector<int> GetPipelineParallelGroupRanks(int global_rank);

// Gather rank-local shards and concatenate them along dim.
std::shared_ptr<Tensor> AllGatherAlongDim(const std::shared_ptr<Tensor> &tensor, int64_t dim, const ProcessGroup *pg);

// Reduce and scatter along the first dimension.
std::shared_ptr<Tensor> ReduceScatterAlongFirstDim(const std::shared_ptr<Tensor> &tensor, comm::ReduceOpType reduce_op,
                                                   const ProcessGroup *pg);

void FinalizeModelGrads(const std::vector<std::shared_ptr<nn::Module>> &model_chunks);
} // namespace infini_train::nn::parallel
