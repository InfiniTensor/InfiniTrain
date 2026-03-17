#pragma once

#include <memory>
#include <string>
#include <vector>

namespace infini_train {
class Tensor;
} // namespace infini_train

namespace infini_train::nn::parallel {
std::string GetDataParallelProcessGroupName(int global_rank);

std::string GetTensorParallelProcessGroupName(int global_rank);

std::string GetPipelineParallelProcessGroupName(int global_rank);

std::vector<int> GetDataParallelGroupRanks(int global_rank);

std::vector<int> GetTensorParallelGroupRanks(int global_rank);

std::vector<int> GetPipelineParallelGroupRanks(int global_rank);

// TP/SP Communication Helper Functions
std::vector<std::shared_ptr<Tensor>> GatherFromTPRegionFunc(const std::shared_ptr<Tensor> &input);
std::vector<std::shared_ptr<Tensor>> ReduceScatterToSPRegionFunc(const std::shared_ptr<Tensor> &input);
std::vector<std::shared_ptr<Tensor>> GatherFromSPRegionFunc(const std::shared_ptr<Tensor> &input);
std::vector<std::shared_ptr<Tensor>> ScatterToTPRegionFunc(const std::shared_ptr<Tensor> &input);
std::vector<std::shared_ptr<Tensor>> ReduceFromTPRegionFunc(const std::shared_ptr<Tensor> &input);
std::vector<std::shared_ptr<Tensor>> CopyToTPRegionFunc(const std::shared_ptr<Tensor> &input);
} // namespace infini_train::nn::parallel
