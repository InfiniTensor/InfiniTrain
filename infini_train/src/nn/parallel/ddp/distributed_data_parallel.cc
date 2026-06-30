#include "infini_train/include/nn/parallel/ddp/distributed_data_parallel.h"

#include <stddef.h>

#include <algorithm>
#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/autograd/function_hook.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/comm.h"
#include "infini_train/include/nn/parallel/ddp/param_and_grad_buffer.h"
#include "infini_train/include/nn/parallel/ddp/reducer.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/process_group.h"
#include "infini_train/include/nn/parallel/rank.h"
#include "infini_train/include/nn/parallel/reduce_op_type.h"
#include "infini_train/include/nn/parallel/utils.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::parallel {
DistributedDataParallel::DistributedDataParallel(
    std::shared_ptr<nn::Module> module, const Rank &rank,
    const DistributedDataParallelConfig ddp_config)
    : ddp_config_(ddp_config) {
  CHECK(ddp_config_.zero_stage >= 0 && ddp_config_.zero_stage <= 3)
      << "DistributedDataParallel: zero_stage must be in 0/1/2/3.";
  if (ddp_config_.zero_stage == 3) {
    LOG(FATAL) << "DistributedDataParallel: ZeRO-3 is not implemented yet.";
  }
  auto *process_group_factory = ProcessGroupFactory::Instance();
  const auto dense_dp_ranks = GetDataParallelWithContextGroupRanks(rank.GlobalRank());
  if (dense_dp_ranks.size() > 1) {
    dense_dp_pg_ = process_group_factory->GetOrCreate(
        GetDataParallelWithContextProcessGroupName(rank.GlobalRank()), dense_dp_ranks);
  }
  const auto expert_dp_ranks =
      GetExpertDataParallelGroupRanks(rank.GlobalRank());
  if (expert_dp_ranks == dense_dp_ranks) {
    expert_dp_pg_ = dense_dp_pg_;
  } else if (expert_dp_ranks.size() > 1) {
    expert_dp_pg_ = process_group_factory->GetOrCreate(
        GetExpertDataParallelProcessGroupName(rank.GlobalRank()),
        expert_dp_ranks);
  }

  const ProcessGroup *validation_pg =
      dense_dp_pg_ ? dense_dp_pg_ : expert_dp_pg_;
  const int expected_device_index = global::GetDeviceIndex(rank.thread_rank());
  const auto validate_device = [validation_pg, expected_device_index](
                                   Device device, const char *kind) {
    if (validation_pg) {
      CHECK_EQ(static_cast<int>(device.type()),
               static_cast<int>(validation_pg->backend()))
          << "DistributedDataParallel " << kind
          << " backend must match the process group backend";
    }
    CHECK_EQ(device.index(), expected_device_index)
        << "DistributedDataParallel " << kind
        << " must use the device assigned to this rank";
  };

  auto trainable_params = module->Parameters();
  for (auto &param : trainable_params) {
    auto device = param->GetDevice();
    validate_device(device, "parameter");
  }
  std::erase_if(trainable_params,
                [](const auto &param) { return !param->requires_grad(); });
  CHECK(!trainable_params.empty())
      << "DistributedDataParallel is not needed when a module doesn't have any "
         "parameter that requires a gradient.";

  // Sparse expert routing may omit hooks entirely. Synchronize the complete
  // parameter list once, after all microbatches, instead of from backward
  // hooks.
  const bool may_have_unused_params =
      ddp_config.handle_unused_parameters ||
      std::any_of(
          trainable_params.begin(), trainable_params.end(),
          [](const auto &param) { return param->is_expert_parallel(); });
  if (may_have_unused_params) {
    CHECK_LE(ddp_config.zero_stage, 2)
        << "Unused parameters currently require ZeRO-0/1/2.";
    CHECK(!ddp_config.overlap_grad_reduce)
        << "Unused parameters require overlap_grad_reduce=false to preserve "
           "collective ordering.";
  }
  if (ddp_config.zero_stage == 0 && may_have_unused_params) {
    // TODO(dcj): Preserve gradient bucketing for dense parameters and use
    // deferred synchronization only for sparse expert parameters to reduce
    // communication overhead.
    deferred_sync_params_ = trainable_params;
  }
  for (auto &param : trainable_params) {
    if (dense_dp_pg_ && deferred_sync_params_.empty() &&
        !ddp_config.gradient_bucketing_enabled && ddp_config.zero_stage < 1) {
      const auto reduce_op = ddp_config_.average_in_collective
                                 ? comm::ReduceOpType::kAvg
                                 : comm::ReduceOpType::kSum;
      auto hook = std::make_unique<autograd::AllReducePostAccumulateHook>(
          reduce_op, dense_dp_pg_, is_last_microbatch_,
          GradientScalingFactorForParameter(false));
      param->RegisterPostAccumulateGradHook(std::move(hook));
    }
  }
  for (auto &buffer : module->Buffers()) {
    validate_device(buffer->GetDevice(), "buffer");
  }
  modules_[kModuleName] = std::move(module);

  if (ddp_config.zero_stage >= 1) {
    BuildParamAndGradBuffers(trainable_params);
    RegisterBackwardHooks(trainable_params);
  } else if (ddp_config.gradient_bucketing_enabled &&
             deferred_sync_params_.empty()) {
    // Bucket Assignment
    const size_t first_cap_bytes = ddp_config.first_bucket_cap_mb * kBytesPerMB;
    const size_t normal_cap_bytes =
        ddp_config.normal_bucket_cap_mb * kBytesPerMB;
    std::vector<size_t> bucket_size_limits = {first_cap_bytes,
                                              normal_cap_bytes};
    std::map<bool, std::vector<std::shared_ptr<Tensor>>> domain_to_params;
    for (const auto &param : trainable_params) {
      domain_to_params[param->is_expert_parallel()].push_back(param);
    }
    for (auto &[is_expert_parallel, params] : domain_to_params) {
      const auto *process_group = ProcessGroupForParameter(is_expert_parallel);
      const float scaling_factor =
          GradientScalingFactorForParameter(is_expert_parallel);
      if (process_group) {
        auto bucket_indices =
            ComputeBucketAssignmentBySize(params, bucket_size_limits);
        auto reducer = std::make_shared<Reducer>(
            params, bucket_indices, ddp_config, process_group, scaling_factor);
        reducer->AttachHooksToParameters();
        reducers_.push_back(std::move(reducer));
      }
    }
  }
}

const ProcessGroup *DistributedDataParallel::ProcessGroupForParameter(
    bool is_expert_parallel) const {
  return is_expert_parallel ? expert_dp_pg_ : dense_dp_pg_;
}

float DistributedDataParallel::GradientScalingFactorForParameter(
    bool is_expert_parallel) const {
  const float dense_dp_size = static_cast<float>(global::GetDataParallelSize());
  CHECK_GT(dense_dp_size, 0.f);
  if (!ddp_config_.average_in_collective) {
    return 1.f / dense_dp_size;
  }
  if (is_expert_parallel) {
    return static_cast<float>(global::GetExpertDataParallelSize()) /
           dense_dp_size;
  }
  return 1.f;
}

void DistributedDataParallel::BuildParamAndGradBuffers(
    const std::vector<std::shared_ptr<Tensor>> &params) {
  // (is_expert_parallel, param_dtype, grad_dtype)
  using BufferKey = std::tuple<bool, DataType, DataType>;
  std::map<BufferKey, std::vector<std::shared_ptr<Tensor>>>
      buffer_key_to_params;

  for (const auto &param : params) {
    auto param_dtype = param->Dtype();
    auto grad_dtype = param->grad() ? param->grad()->Dtype() : param_dtype;
    buffer_key_to_params[{param->is_expert_parallel(), param_dtype, grad_dtype}]
        .push_back(param);
  }

  param_grad_buffers_.clear();
  param_grad_buffers_.reserve(buffer_key_to_params.size());

  for (auto &kv : buffer_key_to_params) {
    auto [is_expert_parallel, param_dtype, grad_dtype] = kv.first;
    auto param_list = kv.second;

    if (param_list.empty()) {
      continue;
    }

    auto buffer = std::make_shared<ParamAndGradBuffer>(
        param_list, param_dtype, grad_dtype,
        ProcessGroupForParameter(is_expert_parallel),
        GradientScalingFactorForParameter(is_expert_parallel), ddp_config_);

    param_grad_buffers_.push_back(buffer);
  }

  // TODO(zbl): option for disable bucketing
  bucket_groups_ = PartitionBuckets(param_grad_buffers_,
                                    /*force_single_bucket_group=*/false);

  if (ddp_config_.zero_stage >= 1 && ddp_config_.overlap_param_gather) {
    std::map<const ProcessGroup *, std::shared_ptr<ParamAndGradBucketGroup>>
        next_group_by_process_group;
    for (auto it = bucket_groups_.rbegin(); it != bucket_groups_.rend(); ++it) {
      const auto &group = *it;
      const auto next_it =
          next_group_by_process_group.find(group->process_group());
      if (next_it != next_group_by_process_group.end()) {
        next_it->second->SetNextParamGatherBucketGroup(group);
      }
      next_group_by_process_group[group->process_group()] = group;
    }
  }

  param_to_bucket_group_.clear();
  for (auto &group : bucket_groups_) {
    for (auto &bucket : group->buckets()) {
      for (auto &param : bucket->params()) {
        auto inserted =
            param_to_bucket_group_.emplace(param.get(), group).second;
        if (!inserted) {
          LOG(FATAL) << "Parameter appears in more than one bucket group.";
        }
      }
    }
  }

  LOG(INFO) << "DDP BuildParamAndGradBuffers: "
            << "sync_dtype_groups=" << buffer_key_to_params.size()
            << ", param_grad_buffers=" << param_grad_buffers_.size()
            << ", bucket_groups=" << bucket_groups_.size();
}

void DistributedDataParallel::RegisterBackwardHooks(
    const std::vector<std::shared_ptr<Tensor>> &params) {
  if (ddp_config_.zero_stage >= 2) {
    // NOTE(zbl): ZeRO-2 bypasses Tensor::grad accumulation: stash grads in the
    // bucket group's
    //            temporary full-grad buffer, then mark the bucket ready for
    //            reduce-scatter.
    class Zero2PreAccumulateGradHook final
        : public autograd::PreAccumulateGradHook {
     public:
      explicit Zero2PreAccumulateGradHook(
          std::weak_ptr<ParamAndGradBucketGroup> group)
          : group_(std::move(group)) {}

      bool TryBypassAccumulate(const std::shared_ptr<Tensor> &param,
                               const std::shared_ptr<Tensor> &grad_output,
                               bool overwrite, float learning_rate) override {
        if (auto group = group_.lock(); group) {
          group->AccumulateParamGrad(param, grad_output, overwrite,
                                     learning_rate);
          if (group->config().overlap_grad_reduce) {
            group->RegisterGradReady(param);
          }
          return true;
        }
        return false;
      }

      void operator()(const std::shared_ptr<Tensor> &) override {}

     private:
      std::weak_ptr<ParamAndGradBucketGroup> group_;
    };

    for (const auto &param : params) {
      auto it = param_to_bucket_group_.find(param.get());
      CHECK(it != param_to_bucket_group_.end());

      std::weak_ptr<ParamAndGradBucketGroup> weak_group = it->second;
      auto hook = std::make_unique<Zero2PreAccumulateGradHook>(weak_group);
      param->RegisterPreAccumulateGradHook(std::move(hook));
    }
    return;
  }

  class DDPPostAccumulateHook final : public autograd::PostAccumulateGradHook {
   public:
    DDPPostAccumulateHook(DistributedDataParallel *ddp,
                          const std::weak_ptr<Tensor> param)
        : ddp_(ddp), param_(param) {}

    void operator()(const std::shared_ptr<Tensor> &) override {
      if (auto param = param_.lock()) {
        ddp_->OnGradReady(param);
      }
    }

   private:
    DistributedDataParallel *ddp_;
    std::weak_ptr<Tensor> param_;
  };

  for (const auto &param : params) {
    auto hook = std::make_unique<DDPPostAccumulateHook>(this, param);
    param->RegisterPostAccumulateGradHook(std::move(hook));
  }
}

void DistributedDataParallel::OnGradReady(
    const std::shared_ptr<Tensor> &param) {
  auto it = param_to_bucket_group_.find(param.get());
  if (it != param_to_bucket_group_.end()) {
    CHECK(param->requires_grad());
    if (ddp_config_.overlap_grad_reduce && (ddp_config_.zero_stage < 2)) {
      CHECK(param->grad()) << "param.grad being None is not safe when "
                              "overlap_grad_reduce is True";
    }

    if (ddp_config_.overlap_grad_reduce) {
      it->second->RegisterGradReady(param);
    }
  }
}

ShardedStateDict DistributedDataParallel::BuildShardedStateDict(
    const ShardedStateDictBuildContext &context) const {
  const auto local_prefix = context.local_prefix.empty()
                                ? std::string(kModuleName)
                                : context.local_prefix + "." + kModuleName;
  return modules_.at(kModuleName)
      ->BuildShardedStateDict({.logical_prefix = context.logical_prefix,
                               .local_prefix = local_prefix});
}

std::vector<std::shared_ptr<Tensor>> DistributedDataParallel::Forward(
    const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
  deferred_sync_pending_ = !deferred_sync_params_.empty();
  auto outputs = (*modules_[kModuleName])(input_tensors);
  for (auto &reducer : reducers_) {
    reducer->PrepareForBackward();
  }
  if (ddp_config_.zero_stage >= 1) {
    for (auto buffer : param_grad_buffers_) {
      buffer->RebindGradViews();
    }
  }
  return outputs;
}

std::shared_ptr<nn::Module> DistributedDataParallel::module() const {
  return modules_.at(kModuleName);
}

DistributedDataParallelConfig DistributedDataParallel::ddp_config() const {
  return ddp_config_;
}

const std::vector<std::shared_ptr<ParamAndGradBuffer>> &
DistributedDataParallel::param_grad_buffers() const {
  return param_grad_buffers_;
}

const std::vector<std::shared_ptr<ParamAndGradBucketGroup>> &
DistributedDataParallel::bucket_groups() const {
  return bucket_groups_;
}

std::unique_ptr<nn::NoSyncGuard> DistributedDataParallel::no_sync() {
  const bool previous = is_last_microbatch_->load(std::memory_order_relaxed);
  SetIsLastMicrobatch(false);
  return std::make_unique<nn::NoSyncGuard>(
      [this, previous] { SetIsLastMicrobatch(previous); });
}

void DistributedDataParallel::FinishGradSync() {
  if (!is_last_microbatch_->load(std::memory_order_relaxed)) {
    return;
  }
  if (deferred_sync_pending_) {
    // TODO: bucket deferred reductions while preserving a fixed collective
    // order. Sparse ZeRO-0 currently synchronizes per parameter, ignoring
    // gradient bucketing options, so unused experts cannot leave a bucket
    // incomplete.
    for (const auto &param : deferred_sync_params_) {
      auto grad = param->grad();
      if (!grad) {
        grad = std::make_shared<Tensor>(param->Dims(), param->Dtype(),
                                        param->GetDevice());
        grad->Fill(0.f);
        param->set_grad(grad);
      }
      const auto scaling_factor =
          GradientScalingFactorForParameter(param->is_expert_parallel());
      if (scaling_factor != 1.f) {
        auto kernel = Dispatcher::Instance().GetKernel(
            {grad->GetDevice().type(), "AccumulateGrad"});
        kernel.Call<void>(grad, scaling_factor - 1.f, grad);
      }
      if (const auto *pg =
              ProcessGroupForParameter(param->is_expert_parallel())) {
        const auto reduce_op = ddp_config_.average_in_collective
                                   ? comm::ReduceOpType::kAvg
                                   : comm::ReduceOpType::kSum;
        comm::AllReduce(grad, reduce_op, pg);
      }
    }
    deferred_sync_pending_ = false;
  }
  for (auto &group : bucket_groups_) {
    group->FinishGradSync();
  }
}

void DistributedDataParallel::SetIsLastMicrobatch(bool is_last_microbatch) {
  is_last_microbatch_->store(is_last_microbatch, std::memory_order_relaxed);
  for (auto &reducer : reducers_) {
    reducer->SetIsLastMicrobatch(is_last_microbatch);
  }
  for (auto &group : bucket_groups_) {
    group->SetIsLastMicrobatch(is_last_microbatch);
  }
}
}  // namespace infini_train::nn::parallel
