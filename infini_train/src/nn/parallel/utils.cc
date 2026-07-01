#include "infini_train/include/nn/parallel/utils.h"

#include "infini_train/include/nn/parallel/global.h"

namespace infini_train::nn::parallel {

std::string GetDataParallelProcessGroupName(int global_rank) {
    return "DP" + std::to_string(global::GetGroupId(global::DP, global_rank));
}

std::string GetDataParallelWithContextProcessGroupName(int global_rank) {
    int dp, tp, cp, pp;
    global::GetCoordOf(global_rank, dp, tp, cp, pp);
    return "DP_CP" + std::to_string(tp * global::GetPipelineParallelSize() + pp);
}

std::string GetTensorParallelProcessGroupName(int global_rank) {
    return "TP" + std::to_string(global::GetGroupId(global::TP, global_rank));
}

std::string GetContextParallelProcessGroupName(int global_rank) {
    return "CP" + std::to_string(global::GetGroupId(global::CP, global_rank));
}

std::string GetPipelineParallelProcessGroupName(int global_rank) {
    return "PP" + std::to_string(global::GetGroupId(global::PP, global_rank));
}

std::vector<int> GetDataParallelGroupRanks(int global_rank) { return global::GetGroupRanks(global::DP, global_rank); }

std::vector<int> GetDataParallelWithContextGroupRanks(int global_rank) {
    int dp, tp, cp, pp;
    global::GetCoordOf(global_rank, dp, tp, cp, pp);
    std::vector<int> ranks;
    ranks.reserve(global::GetDataParallelSize() * global::GetContextParallelSize());
    for (int dp_idx = 0; dp_idx < global::GetDataParallelSize(); ++dp_idx) {
        for (int cp_idx = 0; cp_idx < global::GetContextParallelSize(); ++cp_idx) {
            ranks.push_back(global::GetRankOf(dp_idx, tp, cp_idx, pp));
        }
    }
    return ranks;
}

std::vector<int> GetTensorParallelGroupRanks(int global_rank) { return global::GetGroupRanks(global::TP, global_rank); }

std::vector<int> GetContextParallelGroupRanks(int global_rank) {
    return global::GetGroupRanks(global::CP, global_rank);
}

std::vector<int> GetPipelineParallelGroupRanks(int global_rank) {
    return global::GetGroupRanks(global::PP, global_rank);
}

} // namespace infini_train::nn::parallel
