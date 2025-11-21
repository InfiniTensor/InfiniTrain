#include "infini_train/include/nn/parallel/utils.h"

#include "infini_train/include/nn/parallel/global.h"

namespace infini_train::nn::parallel {

std::string GetDataParallelProcessGroupName(int global_rank) {
    return "DP" + std::to_string(global::GetGroupId(global::DP, global_rank));
}

std::string GetTensorParallelProcessGroupName(int global_rank) {
    return "TP" + std::to_string(global::GetGroupId(global::TP, global_rank));
}

std::string GetPipelineParallelProcessGroupName(int global_rank) {
    return "PP" + std::to_string(global::GetGroupId(global::PP, global_rank));
}

std::vector<int> GetDataParallelGroupRanks(int global_rank) { return global::GetGroupRanks(global::DP, global_rank); }

std::vector<int> GetTensorParallelGroupRanks(int global_rank) { return global::GetGroupRanks(global::TP, global_rank); }

std::vector<int> GetPipelineParallelGroupRanks(int global_rank) {
    return global::GetGroupRanks(global::PP, global_rank);
}
} // namespace infini_train::nn::parallel
