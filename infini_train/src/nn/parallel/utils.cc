#include "infini_train/include/nn/parallel/utils.h"

#include <unordered_set>

#include "glog/logging.h"

#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/ddp/distributed_data_parallel.h"
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

void FinalizeSequenceParallelGradients(const std::vector<LocalGradShard> &param_grads) {
    // SP replicas see different sequence shards, so replicated parameter grads
    // must be summed across TP before the optimizer consumes them.
    if (!global::GetSequenceParallelEnabled() || global::GetTensorParallelSize() <= 1 || param_grads.empty()) {
        return;
    }

    const ProcessGroup *tp_group = nullptr;
    for (const auto &[param, grad] : param_grads) {
        if (!param || !param->sequence_parallel() || !grad) {
            continue;
        }

        if (tp_group == nullptr) {
            tp_group = GetTensorParallelGroup(*param);
            CHECK_NOTNULL(tp_group);
        }
        tp_group->AllReduce(grad, function::ReduceOpType::kSum, /*async_op=*/false);
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

// Call once after all microbatch backwards, before gradient clipping and optimizer updates.
void FinalizeModelGrads(const std::vector<std::shared_ptr<nn::Module>> &model_chunks) {
    // Finish DP gradient synchronization for all chunks before TP reduction.
    for (const auto &model_chunk : model_chunks) {
        if (auto ddp = std::dynamic_pointer_cast<DistributedDataParallel>(model_chunk)) {
            ddp->FinishGradSync();
        }
    }

    if (global::GetTensorParallelSize() <= 1) {
        // DP synchronization is complete; skip TP-specific finalization.
        return;
    }

    for (const auto &model_chunk : model_chunks) {
        // Identify replicated Q/K norm weights whose gradients are split across TP-local heads.
        std::unordered_set<const Tensor *> qk_norm_params;
        for (const auto &[name, param] : model_chunk->NamedParameters()) {
            if (name == "q_norm.weight" || name.ends_with(".q_norm.weight") || name == "k_norm.weight"
                || name.ends_with(".k_norm.weight")) {
                qk_norm_params.insert(param.get());
            }
        }
        // Pair original parameters with the gradient views consumed by the optimizer.
        std::vector<LocalGradShard> param_grads;
        auto ddp = std::dynamic_pointer_cast<DistributedDataParallel>(model_chunk);
        if (ddp && ddp->ddp_config().zero_stage >= 1) {
            // ZeRO-1/2 use DP-local gradient shard views registered by DistOpt.
            for (const auto &group : ddp->bucket_groups()) {
                const auto &shards = group->local_grad_shards();
                param_grads.insert(param_grads.end(), shards.begin(), shards.end());
            }
        } else {
            // Ordinary optimizer / ZeRO-0 use full gradients from param.grad().
            for (const auto &param : model_chunk->Parameters()) { param_grads.emplace_back(param, param->grad()); }
        }
        // TP SUM combines local-token gradients for SP norms, row bias, position embeddings and routers.
        FinalizeSequenceParallelGradients(param_grads);

        // TP SUM combines Q/K norm head-shard gradients regardless of SP; their SP flag is false.
        for (const auto &[param, grad] : param_grads) {
            if (param && param->requires_grad() && grad && qk_norm_params.contains(param.get())) {
                const auto *tp_group = GetTensorParallelGroup(*param);
                CHECK_NOTNULL(tp_group);
                tp_group->AllReduce(grad, function::ReduceOpType::kSum, /*async_op=*/false);
            }
        }
    }

    // NOTE(zbl): Extend this entry for PP tied embeddings, MoE shared parameters and loss normalization.
}

} // namespace infini_train::nn::parallel
