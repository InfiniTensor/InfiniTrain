#pragma once

#include <string>
#include <vector>

namespace infini_train::nn::parallel {
std::string GetDataParallelProcessGroupName(int thread_rank);

std::string GetTensorParallelProcessGroupName(int thread_rank);

std::string GetPipelineParallelProcessGroupName(int thread_rank);

std::vector<int> GetDataParallelGroupRanks(int rank);

std::vector<int> GetTensorParallelGroupRanks(int rank);
} // namespace infini_train::nn::parallel
