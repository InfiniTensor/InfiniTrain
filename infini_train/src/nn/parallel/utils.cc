#include "infini_train/include/nn/parallel/utils.h"

#include "glog/logging.h"

#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/process_group.h"
#include "infini_train/include/nn/parallel/reduce_op_type.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::parallel {
namespace {
const ProcessGroup *GetTensorParallelGroup(const Tensor &tensor) {
    const int global_rank = tensor.GetDevice().Rank().GlobalRank();
    return ProcessGroupFactory::Instance(tensor.GetDevice().type())
        ->Get(GetTensorParallelProcessGroupName(global_rank));
}

void FinalizeSequenceParallelGradients(const std::vector<std::shared_ptr<Tensor>> &params) {
    // SP replicas see different sequence shards, so replicated parameter grads
    // must be summed across TP before the optimizer consumes them.
    if (!global::GetSequenceParallelEnabled() || global::GetTensorParallelSize() <= 1 || params.empty()) {
        return;
    }

    const ProcessGroup *tp_group = nullptr;
    for (const auto &param : params) {
        if (!param || !param->sequence_parallel() || !param->grad()) {
            continue;
        }

        if (tp_group == nullptr) {
            tp_group = GetTensorParallelGroup(*param);
            CHECK_NOTNULL(tp_group);
        }
        tp_group->AllReduce(param->grad(), function::ReduceOpType::kSum, /*async_op=*/false);
    }
}
} // namespace

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

std::shared_ptr<Tensor> GatherTensorParallelShard(const std::shared_ptr<Tensor> &tensor, int64_t dim) {
    const int tp_size = global::GetTensorParallelSize();
    CHECK_GT(tp_size, 0) << "Tensor Parallel group not initialized";
    if (tp_size == 1) {
        return tensor;
    }

    if (dim < 0) {
        dim += static_cast<int64_t>(tensor->Dims().size());
    }
    CHECK_GE(dim, 0);
    CHECK_LT(dim, static_cast<int64_t>(tensor->Dims().size()));

    auto device = tensor->GetDevice();
    auto *tp_group = ProcessGroupFactory::Instance(device.type())
                         ->Get(GetTensorParallelProcessGroupName(device.Rank().GlobalRank()));

    std::vector<int64_t> gathered_dims = tensor->Dims();
    gathered_dims[0] *= tp_size;
    auto gathered = std::make_shared<Tensor>(gathered_dims, tensor->Dtype(), device);
    tp_group->AllGather(gathered, tensor, false);

    if (dim == 0) {
        return gathered;
    }

    // AllGather stacks shards along dim 0. Restore rank-major shards before concatenating on the requested dimension.
    auto rank_major_shards = gathered->Split(tensor->Dims()[0], 0);
    return nn::function::Concat(rank_major_shards, dim)->Contiguous();
}

void FinalizeModelGrads(const std::vector<std::shared_ptr<Tensor>> &params) {
    FinalizeSequenceParallelGradients(params);

    // TODO(zbl): Future model-gradient finalization goes here, such as PP tied embeddings,
    //            MoE shared parameters, and loss normalization.
}

} // namespace infini_train::nn::parallel
