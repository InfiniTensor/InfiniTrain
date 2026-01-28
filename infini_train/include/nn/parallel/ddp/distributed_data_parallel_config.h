#pragma once

#include <limits>

namespace infini_train::nn::parallel {
namespace {
// Default bucket size in alignment with PyTorch
constexpr int kFirstBucketCapMB = 1;
constexpr int kNormalBucketCapMB = 25;
} // namespace

class DistributedDataParallelConfig {
public:
    // ======================================================
    // Reducer-related args
    // Ref: https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
    // ======================================================
    // Max capacity for each bucket(in MB).
    size_t first_bucket_cap_mb = kFirstBucketCapMB;
    size_t normal_bucket_cap_mb = kNormalBucketCapMB;

    // When set true, map param.grad directly to the slice of bucket.flat(same address in memory) instead of memcpy.
    bool gradient_as_bucket_view = true;

    // Whether to enable gradient bucketing.
    bool gradient_bucketing_enabled = true;

    // ======================================================
    // DistributedOptimizer-related args
    // Ref:
    // https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/distributed/distributed_data_parallel_config.py
    // ======================================================
    // Whether to enable DistributedOptimizer (ZeRO-1 equivalent).
    // When set true:
    //   1) Gradients/params are managed by ParamAndGradBuffer and reduced in groups.
    //   2) The classic DDP reducer path is not used (i.e., disable reducer/bucketing in the DDP sense).
    bool use_distributed_optimizer = false;

    // Whether to overlap gradient reduce-scatter/all-reduce with backward compute.
    // In this case, grad reduce is triggered immediately when a grad is ready or till all grads are ready.
    bool overlap_grad_reduce = true;

    // Whether to overlap parameter all-gather with forward compute.
    bool overlap_param_gather = true;

    // Whether to average values inside collectives (divide by world size) instead of summing.
    bool average_in_collective = true;

    // Whether to check NaNs/Infs/unusually large in gradients before collectives.
    // TODO(zbl): Unused by now, to be implemented in ParamAndGradBucketGroup::StartGradSync()
    bool check_for_nan_in_grad = false;
    bool check_for_large_grads = false;

    // Number of DistributedOptimizer instances.
    // Multiple DistOpt is used for building hierarchical collective groups for param/grad.
    // TODO(zbl): Unused by now, to be implemented in ParamAndGradBucketGroup
    int num_distributed_optimizer_instances = 1;

    // Maximum number of parameters in each ParamAndGradBucket.
    // NOTE(zbl): This is distinct from DDP Reducer's MB-based bucket caps.
    // TODO(zbl): To unify the definition of bucket_size argument for users
    size_t bucket_size_in_elements = 40000000;

    // Whether to pad bucket sizes to improve NCCL bus bandwidth utilization.
    bool pad_buckets_for_high_nccl_busbw = false;
};
} // namespace infini_train::nn::parallel
