#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "infini_train/include/nn/parallel/reduce_op_type.h"

namespace infini_train {
class Tensor;
} // namespace infini_train

namespace infini_train::nn::parallel {
class ProcessGroup;

// DP group generated from the dense rank view.
std::string GetDataParallelProcessGroupName(int global_rank);

// DP group generated from the expert rank view, exposed as EDP.
std::string GetExpertDataParallelProcessGroupName(int global_rank);

std::string GetTensorParallelProcessGroupName(int global_rank);

// TP group generated from the expert rank view, exposed as ETP.
std::string GetExpertTensorParallelProcessGroupName(int global_rank);

std::string GetPipelineParallelProcessGroupName(int global_rank);

std::string GetExpertParallelProcessGroupName(int global_rank);

// The expert rank view's ETP + EP group.
std::string GetExpertTensorAndExpertParallelProcessGroupName(int global_rank);

std::vector<int> GetDataParallelGroupRanks(int global_rank);

std::vector<int> GetExpertDataParallelGroupRanks(int global_rank);

std::vector<int> GetTensorParallelGroupRanks(int global_rank);

std::vector<int> GetExpertTensorParallelGroupRanks(int global_rank);

std::vector<int> GetPipelineParallelGroupRanks(int global_rank);

std::vector<int> GetExpertParallelGroupRanks(int global_rank);

std::vector<int> GetExpertTensorAndExpertParallelGroupRanks(int global_rank);

// Gather rank-local shards and concatenate them along dim.
std::shared_ptr<Tensor> AllGatherAlongDim(const std::shared_ptr<Tensor> &tensor, int64_t dim, const ProcessGroup *pg);

// Reduce and scatter along the first dimension.
std::shared_ptr<Tensor> ReduceScatterAlongFirstDim(const std::shared_ptr<Tensor> &tensor, comm::ReduceOpType reduce_op,
                                                   const ProcessGroup *pg);

} // namespace infini_train::nn::parallel
