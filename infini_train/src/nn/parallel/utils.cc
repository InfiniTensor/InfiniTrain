#include "infini_train/include/nn/parallel/utils.h"

#include "infini_train/include/nn/parallel/global.h"

namespace infini_train::nn::parallel {

std::string GetDataParallelProcessGroupName(int thread_rank) {
    return "DP" + std::to_string(global::GetGroupId(global::DP, thread_rank));
}

std::string GetTensorParallelProcessGroupName(int thread_rank) {
    return "TP" + std::to_string(global::GetGroupId(global::TP, thread_rank));
}

std::string GetPipelineParallelProcessGroupName(int thread_rank) {
    return "PP" + std::to_string(global::GetGroupId(global::PP, thread_rank));
}

std::vector<int> GetDataParallelGroupRanks(int thread_rank) { return global::GetGroupRanks(global::DP, thread_rank); }

std::vector<int> GetTensorParallelGroupRanks(int thread_rank) { return global::GetGroupRanks(global::TP, thread_rank); }

} // namespace infini_train::nn::parallel
