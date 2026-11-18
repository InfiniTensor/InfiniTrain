#include "infini_train/include/nn/parallel/context_parallel.h"

#include <algorithm>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/nn/modules/transformer/utils.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/model_parallel_config.h"
#include "infini_train/include/nn/parallel/process_group.h"
#include "infini_train/include/nn/parallel/rank.h"
#include "infini_train/include/nn/parallel/utils.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/nn/parallel/cp/cp_common.h"

namespace infini_train::nn::parallel {

thread_local int cp_rank = 0;

// CP State Helper Functions
int GetContextParallelRank() { return cp_rank; }

// CP Communication Helper Functions
std::shared_ptr<Tensor> SliceAlongCPRegionFunc(
    const std::shared_ptr<Tensor> &input, int64_t dim) {
  const int cp_size = global::GetContextParallelSize();
  if (cp_size == 1) {
    return input;
  }

  // FIXME: CP size/layout changes downstream SP token ownership and TP
  // ReduceScatter partitioning. BF16 summation can then round differently
  // from non-CP even with identical local GEMM results. Keep input-dtype
  // communication; validate long-run loss/gradient alignment across layouts.
  return detail::SliceCPRank(input, dim, GetContextParallelRank(), cp_size)
      ->Contiguous();
}

std::shared_ptr<Tensor> MakeCausalMaskAlongCPRegionFunc(int64_t local_t,
                                                        const Device &device) {
  const int cp_size = global::GetContextParallelSize();
  const int rank = GetContextParallelRank();
  const int64_t full_t = local_t * cp_size;
  auto keys = nn::init::Arange(0, full_t, DataType::kFLOAT32, device);
  std::vector<int64_t> query_positions(local_t);
  if (global::GetAttentionCPLayout() == "zigzag" && cp_size > 1) {
    CHECK_EQ(local_t % 2, 0)
        << "Zigzag CP sequence length must contain two equal halves";
    const int64_t half = local_t / 2;
    for (int64_t i = 0; i < half; ++i) {
      query_positions[i] = rank * half + i;
      query_positions[half + i] = (2 * cp_size - rank - 1) * half + i;
    }
  } else {
    for (int64_t i = 0; i < local_t; ++i) {
      query_positions[i] = rank * local_t + i;
    }
  }

  std::vector<std::shared_ptr<Tensor>> mask_rows;
  mask_rows.reserve(local_t);
  for (const int64_t query_position : query_positions) {
    mask_rows.push_back(keys->Gt(query_position));
  }
  return nn::function::Concat(mask_rows, 0)->View({1, 1, local_t, full_t});
}

// CP Attention Backend Functions
std::shared_ptr<Tensor> AttnFuncWithCPAndKVP2P(
    const std::shared_ptr<Tensor> &q, const std::shared_ptr<Tensor> &k,
    const std::shared_ptr<Tensor> &v, const std::shared_ptr<Tensor> &mask) {
  return detail::ApplyAttnWithCPAndKVP2P(q, k, v, mask, detail::GetCPGroup(q));
}

std::shared_ptr<Tensor> AttnFuncWithCPAndKVAllGather(
    const std::shared_ptr<Tensor> &q, const std::shared_ptr<Tensor> &k,
    const std::shared_ptr<Tensor> &v, const std::shared_ptr<Tensor> &mask) {
  CHECK_EQ(k->Dims()[1], v->Dims()[1]);
  CHECK_EQ(q->Dims()[1] % k->Dims()[1], 0);
  // gathered_k: (B, H_kv, T_g, D)
  auto gathered_k = detail::GatherFromCPRegionFunc(k);
  // gathered_v: (B, H_kv, T_g, D)
  auto gathered_v = detail::GatherFromCPRegionFunc(v);
  // k_for_attn: (B, H_q, T_g, D)
  return ScaledDotProductAttention(q, gathered_k, gathered_v, k->Dims()[1],
                                   mask);
}

std::shared_ptr<Tensor> AttnFuncWithCPAndQKVOA2A(
    const std::shared_ptr<Tensor> &q, const std::shared_ptr<Tensor> &k,
    const std::shared_ptr<Tensor> &v, const std::shared_ptr<Tensor> &mask) {
  const int cp_size = global::GetContextParallelSize();
  const int64_t q_heads = q->Dims()[1];
  const int64_t kv_heads = k->Dims()[1];
  CHECK_EQ(kv_heads, v->Dims()[1]);
  CHECK_EQ(q_heads % cp_size, 0)
      << "A2A CP requires local query heads divisible by CP size";
  CHECK_EQ(kv_heads % cp_size, 0)
      << "A2A CP requires local KV heads divisible by CP size";
  CHECK_EQ(q_heads % kv_heads, 0);

  if (cp_size == 1) {
    return ScaledDotProductAttention(q, k, v, kv_heads, mask);
  }

  const auto *cp_group = detail::GetCPGroup(q);

  // q_shard: (B, H_q/CP, T_g, D)
  auto q_shard = detail::AllToAllSeqToHeadCPRegionFunc(q, cp_group);
  // k_shard: (B, H_kv/CP, T_g, D)
  auto k_shard = detail::AllToAllSeqToHeadCPRegionFunc(k, cp_group);
  // v_shard: (B, H_kv/CP, T_g, D)
  auto v_shard = detail::AllToAllSeqToHeadCPRegionFunc(v, cp_group);
  // full_mask: (1, 1, T_g, T_g)
  auto full_mask = mask ? detail::GatherFromCPRegionFunc(mask) : nullptr;

  // output_shard: (B, H_q/CP, T_g, D)
  auto output_shard = ScaledDotProductAttention(q_shard, k_shard, v_shard,
                                                k_shard->Dims()[1], full_mask);

  // output: (B, H_q, T_l, D)
  return detail::AllToAllHeadToSeqCPRegionFunc(output_shard, cp_group);
}

std::shared_ptr<Tensor> AttnFuncWithCPAndQKVOA2AKVP2P(
    const std::shared_ptr<Tensor> &q, const std::shared_ptr<Tensor> &k,
    const std::shared_ptr<Tensor> &v, const std::shared_ptr<Tensor> &mask) {
  const auto &sizes = global::GetHierarchicalContextParallelSizes();
  CHECK_EQ(sizes.size(), 2);
  if (sizes[0] == 1) {
    return AttnFuncWithCPAndKVP2P(q, k, v, mask);
  }
  if (sizes[1] == 1) {
    return AttnFuncWithCPAndQKVOA2A(q, k, v, mask);
  }

  CHECK(mask) << "Hierarchical CP ring attention expects a causal mask";
  CHECK_EQ(q->Dims()[1] % sizes[0], 0)
      << "A2A size must divide local query heads after TP";
  CHECK_EQ(k->Dims()[1] % sizes[0], 0)
      << "A2A size must divide local KV heads after TP";
  CHECK_EQ(k->Dims()[1], v->Dims()[1]);
  CHECK_EQ(q->Dims()[1] % k->Dims()[1], 0);
  const auto device = q->GetDevice();
  const auto *factory = ProcessGroupFactory::Instance(device.type());
  const auto *a2a_group =
      factory->Get(GetHierarchicalContextParallelProcessGroupName(
          device.Rank().GlobalRank(), 0));
  const auto *p2p_group =
      factory->Get(GetHierarchicalContextParallelProcessGroupName(
          device.Rank().GlobalRank(), 1));

  // A2A joins adjacent ranks into a contiguous block or a zigzag pair of
  // blocks, with heads sharded across the A2A group.
  // (B, H, S/CP, D) -> (B, H/A, S/P, D), where CP = A * P.
  auto q_shard = detail::AllToAllSeqToHeadCPRegionFunc(q, a2a_group);
  auto k_shard = detail::AllToAllSeqToHeadCPRegionFunc(k, a2a_group);
  auto v_shard = detail::AllToAllSeqToHeadCPRegionFunc(v, a2a_group);
  // Gather query rows only inside the A2A group: (1, 1, S/CP, S) -> (1, 1, S/P,
  // S).
  auto ring_mask = detail::GatherAlongFirstDim(mask->Transpose(0, 2), a2a_group)
                       ->Transpose(0, 2)
                       ->Contiguous();
  ring_mask = detail::ReorderCPSequence(ring_mask, 2, sizes[0],
                                        /*to_sequence_order=*/true);
  auto output = detail::ApplyAttnWithCPAndKVP2P(q_shard, k_shard, v_shard,
                                                ring_mask, p2p_group);
  return detail::AllToAllHeadToSeqCPRegionFunc(output, a2a_group);
}

std::shared_ptr<Tensor> AttnForwardFuncWithCP(
    const std::shared_ptr<Tensor> &q, const std::shared_ptr<Tensor> &k,
    const std::shared_ptr<Tensor> &v, const std::shared_ptr<Tensor> &mask) {
  CHECK_GT(global::GetContextParallelSize(), 1);
  switch (global::GetContextParallelCommType()) {
    case global::ContextParallelCommType::kP2P:
      return AttnFuncWithCPAndKVP2P(q, k, v, mask);
    case global::ContextParallelCommType::kA2A:
      return AttnFuncWithCPAndQKVOA2A(q, k, v, mask);
    case global::ContextParallelCommType::kAllGather:
      return AttnFuncWithCPAndKVAllGather(q, k, v, mask);
    case global::ContextParallelCommType::kA2AP2P:
      return AttnFuncWithCPAndQKVOA2AKVP2P(q, k, v, mask);
  }
  LOG(FATAL) << "AttnForwardFuncWithCP: unreachable communication type";
}

}  // namespace infini_train::nn::parallel
