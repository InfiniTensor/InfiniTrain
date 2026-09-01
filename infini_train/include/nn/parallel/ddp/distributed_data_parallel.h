#pragma once

#include <atomic>
#include <memory>
#include <unordered_map>
#include <vector>

#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/ddp/distributed_data_parallel_config.h"
#include "infini_train/include/shard_spec.h"

namespace infini_train {
class Tensor;

namespace nn::parallel {
class Rank;
}  // namespace nn::parallel
}  // namespace infini_train

namespace infini_train::nn::parallel {
class ParamAndGradBucketGroup;
class ParamAndGradBuffer;
class ProcessGroup;
class Reducer;

namespace {
constexpr char kModuleName[] = "module";

}  // namespace

class DistributedDataParallel : public nn::Module {
 public:
  using Module::BuildShardedStateDict;
  DistributedDataParallel(std::shared_ptr<nn::Module> module, const Rank &rank,
                          DistributedDataParallelConfig ddp_config);

  std::vector<std::shared_ptr<Tensor>> Forward(
      const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

  std::shared_ptr<nn::Module> module() const;
  ShardedStateDict BuildShardedStateDict(
      const ShardedStateDictBuildContext &context) const override;

  std::unique_ptr<nn::NoSyncGuard> no_sync() override;

  void FinishGradSync();

  DistributedDataParallelConfig ddp_config() const;

  const std::vector<std::shared_ptr<ParamAndGradBuffer>> &param_grad_buffers()
      const;

  const std::vector<std::shared_ptr<ParamAndGradBucketGroup>> &bucket_groups()
      const;

  void RegisterForwardPreHooks();

 private:
  // Dense parameters synchronize over DP; sharded expert parameters over EDP.
  // A null group denotes a single replica, which still needs gradient scaling.
  const ProcessGroup *ProcessGroupForParameter(bool is_expert_parallel) const;
  // Normalize both domains by dense DP size: each expert has already
  // accumulated contributions from the EP ranks during dispatch/combine
  // backward.
  float GradientScalingFactorForParameter(bool is_expert_parallel) const;
  void BuildParamAndGradBuffers(
      const std::vector<std::shared_ptr<Tensor>> &params);
  void RegisterBackwardHooks(
      const std::vector<std::shared_ptr<Tensor>> &params);
  void OnGradReady(const std::shared_ptr<Tensor> &param);
  void SetIsLastMicrobatch(bool is_last_microbatch);

 private:
  // Whether to enable grad sync on last microbatch (DDP naive path)
  std::shared_ptr<std::atomic_bool> is_last_microbatch_ =
      std::make_shared<std::atomic_bool>(true);
  std::vector<std::shared_ptr<Reducer>> reducers_;
  // Sparse ZeRO-0: fixed-order finalization also covers EP=1 with unused
  // experts.
  std::vector<std::shared_ptr<Tensor>> deferred_sync_params_;
  bool deferred_sync_pending_ = false;

  DistributedDataParallelConfig ddp_config_;
  const ProcessGroup *dense_dp_pg_ = nullptr;
  const ProcessGroup *expert_dp_pg_ = nullptr;

  std::vector<std::shared_ptr<ParamAndGradBuffer>> param_grad_buffers_;
  std::vector<std::shared_ptr<ParamAndGradBucketGroup>> bucket_groups_;
  std::unordered_map<Tensor *, std::shared_ptr<ParamAndGradBucketGroup>>
      param_to_bucket_group_;
};

}  // namespace infini_train::nn::parallel
