#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace infini_train {
class Tensor;
} // namespace infini_train

namespace infini_train::nn::parallel {
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

// TP/SP Communication Helper Functions
std::shared_ptr<Tensor> GatherTensorParallelShard(const std::shared_ptr<Tensor> &tensor, int64_t dim);

std::vector<std::shared_ptr<Tensor>> GatherFromTPRegionFunc(const std::shared_ptr<Tensor> &input);
std::vector<std::shared_ptr<Tensor>> ReduceScatterToSPRegionFunc(const std::shared_ptr<Tensor> &input);
std::vector<std::shared_ptr<Tensor>> GatherFromSPRegionFunc(const std::shared_ptr<Tensor> &input);
std::vector<std::shared_ptr<Tensor>> ScatterToTPRegionFunc(const std::shared_ptr<Tensor> &input);
std::vector<std::shared_ptr<Tensor>> ReduceFromTPRegionFunc(const std::shared_ptr<Tensor> &input);
std::vector<std::shared_ptr<Tensor>> CopyToTPRegionFunc(const std::shared_ptr<Tensor> &input);
} // namespace infini_train::nn::parallel
