#include "infini_train/include/nn/parallel/utils.h"

#include <unordered_set>
#include <utility>

#include "glog/logging.h"

#include "infini_train/include/device.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/ddp/distributed_data_parallel.h"
#include "infini_train/include/nn/parallel/ddp/distributed_data_parallel_config.h"
#include "infini_train/include/nn/parallel/ddp/param_and_grad_buffer.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/parallel_functional.h"
#include "infini_train/include/nn/parallel/process_group.h"
#include "infini_train/include/nn/parallel/rank.h"
#include "infini_train/include/nn/parallel/reduce_op_type.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::parallel {
namespace {
const ProcessGroup *GetTensorParallelGroup(const Tensor &tensor) {
  const int global_rank = tensor.GetDevice().Rank().GlobalRank();
  return ProcessGroupFactory::Instance(tensor.GetDevice().type())
      ->Get(GetTensorParallelProcessGroupName(global_rank));
}

void FinalizeSequenceParallelGradients(
    const std::vector<LocalGradShard> &param_grads) {
  // SP replicas see different sequence shards, so replicated parameter grads
  // must be summed across TP before the optimizer consumes them.
  if (!global::GetSequenceParallelEnabled() ||
      global::GetTensorParallelSize() <= 1 || param_grads.empty()) {
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
    tp_group->AllReduce(grad, comm::ReduceOpType::kSum, /*async_op=*/false);
  }
}

void FinalizeQKNormWeightGradients(
    const nn::Module &model_chunk,
    const std::vector<LocalGradShard> &param_grads) {
  // CausalSelfAttention Q/K norm weights (e.g. in Qwen3) are shared across
  // heads and replicated across TP ranks. Each rank computes gradients for
  // TP-local heads, so sum them across TP regardless of SP. These weights have
  // sequence_parallel=false and are not reduced by
  // FinalizeSequenceParallelGradients.
  std::unordered_set<const Tensor *> qk_norm_params;
  for (const auto &[name, param] : model_chunk.NamedParameters()) {
    if (name.ends_with("q_norm.weight") || name.ends_with("k_norm.weight")) {
      qk_norm_params.insert(param.get());
    }
  }

  for (const auto &[param, grad] : param_grads) {
    if (param && param->requires_grad() && grad &&
        qk_norm_params.contains(param.get())) {
      const auto *tp_group = GetTensorParallelGroup(*param);
      CHECK_NOTNULL(tp_group);
      tp_group->AllReduce(grad, comm::ReduceOpType::kSum, /*async_op=*/false);
    }
  }
}
}  // namespace

std::string GetDataParallelProcessGroupName(int global_rank) {
  return "DP" + std::to_string(global::GetGroupId(global::kDp, global_rank));
}

std::string GetExpertDataParallelProcessGroupName(int global_rank) {
  return "EDP" + std::to_string(global::GetExpertRankGenerator().GroupId(
                     global::kDp, global_rank));
}

std::string GetDataParallelWithContextProcessGroupName(int global_rank) {
  int dp, tp, cp, pp;
  global::GetCoordOf(global_rank, dp, tp, cp, pp);
  return "DP_CP" + std::to_string(tp * global::GetPipelineParallelSize() + pp);
}

std::string GetTensorParallelProcessGroupName(int global_rank) {
  return "TP" + std::to_string(global::GetGroupId(global::kTp, global_rank));
}

std::string GetExpertTensorParallelProcessGroupName(int global_rank) {
  return "ETP" + std::to_string(global::GetExpertRankGenerator().GroupId(
                     global::kTp, global_rank));
}

std::string GetContextParallelProcessGroupName(int global_rank) {
  return "CP" + std::to_string(global::GetGroupId(global::CP, global_rank));
}

std::string GetPipelineParallelProcessGroupName(int global_rank) {
  return "PP" + std::to_string(global::GetGroupId(global::kPp, global_rank));
}

std::string GetExpertParallelProcessGroupName(int global_rank) {
  return "EP" + std::to_string(global::GetExpertRankGenerator().GroupId(
                    global::kEp, global_rank));
}

std::string GetExpertTensorAndExpertParallelProcessGroupName(int global_rank) {
  return "ETP_EP" + std::to_string(global::GetExpertRankGenerator().GroupId(
                        {global::kTp, global::kEp}, global_rank));
}

std::vector<int> GetDataParallelGroupRanks(int global_rank) {
  return global::GetDenseRankGenerator().GroupRanks(global::kDp, global_rank);
}

std::vector<int> GetExpertDataParallelGroupRanks(int global_rank) {
  return global::GetExpertRankGenerator().GroupRanks(global::kDp, global_rank);
}

std::vector<int> GetDataParallelWithContextGroupRanks(int global_rank) {
  int dp, tp, cp, pp;
  global::GetCoordOf(global_rank, dp, tp, cp, pp);
  std::vector<int> ranks;
  ranks.reserve(global::GetDataParallelSize() *
                global::GetContextParallelSize());
  for (int dp_idx = 0; dp_idx < global::GetDataParallelSize(); ++dp_idx) {
    for (int cp_idx = 0; cp_idx < global::GetContextParallelSize(); ++cp_idx) {
      ranks.push_back(global::GetRankOf(dp_idx, tp, cp_idx, pp));
    }
  }
  return ranks;
}

std::vector<int> GetTensorParallelGroupRanks(int global_rank) {
  return global::GetDenseRankGenerator().GroupRanks(global::kTp, global_rank);
}

std::vector<int> GetExpertTensorParallelGroupRanks(int global_rank) {
  return global::GetExpertRankGenerator().GroupRanks(global::kTp, global_rank);
}

std::vector<int> GetContextParallelGroupRanks(int global_rank) {
  return global::GetGroupRanks(global::CP, global_rank);
}

std::vector<int> GetPipelineParallelGroupRanks(int global_rank) {
  return global::GetDenseRankGenerator().GroupRanks(global::kPp, global_rank);
}

std::vector<int> GetExpertParallelGroupRanks(int global_rank) {
  return global::GetExpertRankGenerator().GroupRanks(global::kEp, global_rank);
}

std::vector<int> GetExpertTensorAndExpertParallelGroupRanks(int global_rank) {
  return global::GetExpertRankGenerator().GroupRanks({global::kTp, global::kEp},
                                                     global_rank);
}

std::shared_ptr<Tensor> AllGatherAlongDim(const std::shared_ptr<Tensor> &tensor,
                                          int64_t dim, const ProcessGroup *pg) {
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

  auto rank_major_shards = gathered->Split(tensor->Dims()[0], 0);
  return nn::function::Concat(rank_major_shards, dim)->Contiguous();
}

std::shared_ptr<Tensor> ReduceScatterAlongFirstDim(
    const std::shared_ptr<Tensor> &tensor, comm::ReduceOpType reduce_op,
    const ProcessGroup *pg) {
  return function::ReduceScatter(tensor, reduce_op, pg);
}

// Call once after all microbatch backwards, before gradient clipping and
// optimizer updates.
void FinalizeModelGrads(
    const std::vector<std::shared_ptr<nn::Module>> &model_chunks) {
  // Finish DP gradient synchronization for all chunks before TP reduction.
  for (const auto &model_chunk : model_chunks) {
    if (auto ddp =
            std::dynamic_pointer_cast<DistributedDataParallel>(model_chunk)) {
      ddp->FinishGradSync();
    }
  }

  if (global::GetTensorParallelSize() <= 1) {
    // DP synchronization is complete; skip TP-specific finalization.
    return;
  }

  for (const auto &model_chunk : model_chunks) {
    // Pair original parameters with the gradient views consumed by the
    // optimizer.
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
      for (const auto &param : model_chunk->Parameters()) {
        param_grads.emplace_back(param, param->grad());
      }
    }
    // TP SUM combines local-token gradients for SP norms, row bias, position
    // embeddings and routers.
    FinalizeSequenceParallelGradients(param_grads);

    // TP SUM combines local-head gradients for replicated Q/K norm weights
    // regardless of SP.
    FinalizeQKNormWeightGradients(*model_chunk, param_grads);
  }

  // NOTE(zbl): Extend this entry for PP tied embeddings, MoE shared parameters
  // and loss normalization.
}

}  // namespace infini_train::nn::parallel
