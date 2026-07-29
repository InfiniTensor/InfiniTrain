#include "infini_train/include/nn/parallel/utils.h"

#include "glog/logging.h"

#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/parallel_functional.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::parallel {

std::string GetDataParallelProcessGroupName(int global_rank) {
    return "DP" + std::to_string(global::GetDenseRankGenerator().GroupId(global::DP, global_rank));
}

std::string GetExpertDataParallelProcessGroupName(int global_rank) {
    return "EDP" + std::to_string(global::GetExpertRankGenerator().GroupId(global::DP, global_rank));
}

std::string GetTensorParallelProcessGroupName(int global_rank) {
    return "TP" + std::to_string(global::GetDenseRankGenerator().GroupId(global::TP, global_rank));
}

std::string GetExpertTensorParallelProcessGroupName(int global_rank) {
    return "ETP" + std::to_string(global::GetExpertRankGenerator().GroupId(global::TP, global_rank));
}

std::string GetPipelineParallelProcessGroupName(int global_rank) {
    return "PP" + std::to_string(global::GetDenseRankGenerator().GroupId(global::PP, global_rank));
}

std::string GetExpertParallelProcessGroupName(int global_rank) {
    return "EP" + std::to_string(global::GetExpertRankGenerator().GroupId(global::EP, global_rank));
}

std::string GetExpertTensorAndExpertParallelProcessGroupName(int global_rank) {
    return "ETP_EP" + std::to_string(global::GetExpertRankGenerator().GroupId({global::TP, global::EP}, global_rank));
}

std::vector<int> GetDataParallelGroupRanks(int global_rank) {
    return global::GetDenseRankGenerator().GroupRanks(global::DP, global_rank);
}

std::vector<int> GetExpertDataParallelGroupRanks(int global_rank) {
    return global::GetExpertRankGenerator().GroupRanks(global::DP, global_rank);
}

std::vector<int> GetTensorParallelGroupRanks(int global_rank) {
    return global::GetDenseRankGenerator().GroupRanks(global::TP, global_rank);
}

std::vector<int> GetExpertTensorParallelGroupRanks(int global_rank) {
    return global::GetExpertRankGenerator().GroupRanks(global::TP, global_rank);
}

std::vector<int> GetPipelineParallelGroupRanks(int global_rank) {
    return global::GetDenseRankGenerator().GroupRanks(global::PP, global_rank);
}

std::vector<int> GetExpertParallelGroupRanks(int global_rank) {
    return global::GetExpertRankGenerator().GroupRanks(global::EP, global_rank);
}

std::vector<int> GetExpertTensorAndExpertParallelGroupRanks(int global_rank) {
    return global::GetExpertRankGenerator().GroupRanks({global::TP, global::EP}, global_rank);
}

std::shared_ptr<Tensor> AllGatherAlongDim(const std::shared_ptr<Tensor> &tensor, int64_t dim, const ProcessGroup *pg) {
    CHECK_NOTNULL(pg);
    if (dim < 0) {
        dim += static_cast<int64_t>(tensor->Dims().size());
    }
    CHECK_GE(dim, 0);
    CHECK_LT(dim, static_cast<int64_t>(tensor->Dims().size()));

    auto gathered = function::AllGather(tensor, pg);

    if (dim == 0) {
        return gathered;
    }

    // AllGather stacks shards along dim 0. Restore rank-major shards before concatenating on the requested dimension.
    auto rank_major_shards = gathered->Split(tensor->Dims()[0], 0);
    return nn::function::Concat(rank_major_shards, dim)->Contiguous();
}

std::shared_ptr<Tensor> ReduceScatterAlongFirstDim(const std::shared_ptr<Tensor> &tensor, comm::ReduceOpType reduce_op,
                                                   const ProcessGroup *pg) {
    return function::ReduceScatter(tensor, reduce_op, pg);
}

} // namespace infini_train::nn::parallel
