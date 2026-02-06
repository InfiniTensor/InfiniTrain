#include "infini_train/include/nn/parallel/ddp/param_and_grad_buffer.h"

#include <algorithm>
#include <cstddef>
#include <iostream>

#include "glog/logging.h"

#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/ddp/distributed_data_parallel_config.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/process_group.h"
#include "infini_train/include/nn/parallel/reduce_op_type.h"
#include "infini_train/include/nn/parallel/work.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::parallel {

namespace {
constexpr size_t kParamStartAlignElements = 64;
constexpr size_t kBucketEndAlignElements = 128;

inline size_t PadTo(size_t value, size_t alignment) {
    if (alignment == 0) {
        return value;
    }
    size_t remainder = value % alignment;
    return remainder == 0 ? value : value + (alignment - remainder);
}

std::shared_ptr<Tensor> AllocateFlatBuffer(size_t num_elements, DataType data_type, Device device) {
    std::vector<int64_t> dims = {static_cast<int64_t>(num_elements)};
    // TODO(zbl): replace with united allocation when memory pool is available
    return std::make_shared<Tensor>(dims, data_type, device);
}

std::shared_ptr<Tensor> GetBufferView(const std::shared_ptr<Tensor> buffer, size_t start_in_elements,
                                      const std::vector<int64_t> &dims) {
    return std::make_shared<Tensor>(*buffer, start_in_elements * kDataTypeToSize.at(buffer->Dtype()), dims);
};

std::vector<std::shared_ptr<Tensor>> ShardBuffer(const std::shared_ptr<Tensor> buffer, size_t ddp_world_size) {
    CHECK_EQ(buffer->NumElements() % ddp_world_size, 0);
    size_t shard_size = buffer->NumElements() / ddp_world_size;
    std::vector<std::shared_ptr<Tensor>> sharded_buffer;
    for (auto i = 0; i < ddp_world_size; ++i) {
        sharded_buffer.push_back(
            GetBufferView(buffer, i * shard_size, std::vector<int64_t>{static_cast<int64_t>(shard_size)}));
    }
    return sharded_buffer;
}

} // namespace

ParamAndGradBucket::ParamAndGradBucket(const std::vector<std::shared_ptr<Tensor>> &params,
                                       const std::shared_ptr<Tensor> &param_data,
                                       const std::shared_ptr<Tensor> &grad_data, size_t offset,
                                       size_t num_elements_unpadded, float gradient_scaling_factor, size_t bucket_id)
    : bucket_id_(bucket_id), params_(std::move(params)), param_data_(std::move(param_data)),
      grad_data_(std::move(grad_data)), offset_(offset), num_elements_unpadded_(num_elements_unpadded),
      gradient_scaling_factor_(gradient_scaling_factor) {
    size_t current_offset = 0;
    for (const auto &param : params_) {
        auto numel = param->NumElements();
        param_to_range_.emplace(param.get(), std::make_pair(current_offset, current_offset + numel));
        current_offset += numel;
    }
}

bool ParamAndGradBucket::GetTensorLocInBucket(const std::shared_ptr<Tensor> &parameter, size_t &start_in_bucket,
                                              size_t &end_in_bucket) const {
    const auto iterator = param_to_range_.find(parameter.get());
    if (iterator == param_to_range_.end()) {
        return false;
    }
    start_in_bucket = iterator->second.first;
    end_in_bucket = iterator->second.second;
    return true;
}

void ParamAndGradBucket::ScaleGradients(float scaling_factor) {
    if (!grad_data_ || scaling_factor == 1.f) {
        return;
    }

    // FIXME(zbl): should perform in-place multiply
    // grad_data_ *= scaling_factor;
    LOG(FATAL) << "ParamAndGradBucket: Should not arrive here";
}

ParamAndGradBucketGroup::ParamAndGradBucketGroup(const std::vector<std::shared_ptr<ParamAndGradBucket>> &buckets,
                                                 const ProcessGroup *collective_pg, size_t process_group_size,
                                                 DistributedDataParallelConfig ddp_config)
    : buckets_(std::move(buckets)), collective_pg_(collective_pg), collective_pg_size_(process_group_size),
      ddp_config_(ddp_config) {
    // TODO(zbl): support hierarchical gradient sync in distopt
    CHECK(ddp_config.num_distributed_optimizer_instances == 1)
        << "ParamAndGradBucketGroup: Multi-instance DistributedOptimizer is not supported yet.";

    for (const auto &bucket : buckets_) {
        for (const auto &param : bucket->params()) { params_.insert(param.get()); }
    }
    if (rank_in_collective_pg_ == -1) {
        auto param = *params_.begin();
        // FIXME(zbl): get correct rank in multi-node settings
        rank_in_collective_pg_ = collective_pg_->GetGroupRank(param->GetDevice().Rank().thread_rank());
    }

    param_buffer_shard_list_.resize(buckets_.size());
    grad_buffer_shard_list_.resize(buckets_.size());
}

void ParamAndGradBucketGroup::Reset() {
    params_with_grad_.clear();
    grad_reduce_work_list_.clear();
    param_gather_work_list_.clear();
    is_last_microbatch_ = true;
    grad_reduce_dispatched_ = false;
    param_gather_dispatched_ = false;
}

void ParamAndGradBucketGroup::RegisterGradReady(const std::shared_ptr<Tensor> &parameter) {
    if (!ddp_config_.overlap_grad_reduce) {
        LOG(WARNING)
            << "ParamAndGradBucketGroup: RegisterGradReady() should only be called when overlap_grad_reduce is "
               "True. Skipping here.";
        return;
    }

    // Only register grads as ready when processing the last microbatch
    if (is_last_microbatch_) {
        if (!parameter || params_.find(parameter.get()) == params_.end()) {
            return;
        }

        const bool inserted = params_with_grad_.insert(parameter.get()).second;
        if (!inserted) {
            LOG(FATAL) << "ParamAndGradBucketGroup: RegisterGradReady() was called twice for the same parameter in a "
                          "bucket group.";
            return;
        }

        if (params_with_grad_.size() == params_.size()) {
            // All param grads are ready in this group, trigger grad sync
            StartGradSync();
        }
    }
}

void ParamAndGradBucketGroup::StartGradSync() {
    if (!collective_pg_) {
        LOG(FATAL) << "ParamAndGradBucketGroup: StartGradSync() called with null collective_pg_.";
        return;
    }

    if (grad_reduce_dispatched_) {
        return;
    }
    if (!grad_reduce_work_list_.empty()) {
        grad_reduce_dispatched_ = true;
        return;
    }

    // TODO(zbl): Check NaN/Inf/too large in grad (options in DistributedDataParallelConfig)

    for (auto bucket : buckets_) {
        if (bucket->gradient_scaling_factor() != 1.f) {
            bucket->ScaleGradients(bucket->gradient_scaling_factor());
        }
    }

    auto reduce_op = ddp_config_.average_in_collective ? function::ReduceOpType::kAvg : function::ReduceOpType::kSum;
    auto async_op = ddp_config_.overlap_grad_reduce && (ddp_config_.num_distributed_optimizer_instances == 1);

    for (auto i = 0; i < buckets_.size(); ++i) {
        auto bucket = buckets_[i];
        std::shared_ptr<Tensor> grad_buffer = bucket->grad_data();
        if (!grad_buffer) {
            continue;
        }

        if (ddp_config_.use_distributed_optimizer) {
            if (grad_buffer_shard_list_[i].empty()) {
                grad_buffer_shard_list_[i] = ShardBuffer(grad_buffer, collective_pg_size_);
            }
            auto local_data_view = grad_buffer_shard_list_[i][rank_in_collective_pg_];
            grad_reduce_work_list_.push_back(
                collective_pg_->ReduceScatter(local_data_view, grad_buffer, reduce_op, async_op));
        } else {
            // NOTE(zbl): Should not arrive here because Reducer-related logic is activated when not using DistOpt
            grad_reduce_work_list_.push_back(collective_pg_->AllReduce(grad_buffer, reduce_op, async_op));
        }
    }

    grad_reduce_dispatched_ = true;
}

void ParamAndGradBucketGroup::FinishGradSync() {
    if (!grad_reduce_dispatched_) {
        StartGradSync();
    }

    if (!ddp_config_.overlap_grad_reduce) {
        // Assume reduce ops are synced and no work needs to be resolved
        grad_reduce_work_list_.clear();
        grad_reduce_dispatched_ = false;
        return;
    }

    CHECK(!grad_reduce_work_list_.empty())
        << "ParamAndGradBucketGroup: Communication call has not been issued for this bucket("
        << params_with_grad_.size() << "/" << params_.size() << " params have grad available)";

    for (auto work : grad_reduce_work_list_) { work->WaitNonBlocking(); }
    grad_reduce_work_list_.clear();
    grad_reduce_dispatched_ = false;
}

void ParamAndGradBucketGroup::StartParamSync(bool force_sync) {
    CHECK(ddp_config_.use_distributed_optimizer);

    if (!collective_pg_) {
        LOG(ERROR) << "ParamAndGradBucketGroup: StartParamSync called with null collective_pg_.";
        return;
    }

    if (force_sync) {
        // force synchronous collective regardless of other settings
        for (auto work : param_gather_work_list_) { work->WaitNonBlocking(); }
        param_gather_work_list_.clear();
        return;
    } else {
        CHECK(param_gather_work_list_.empty());
    }

    auto async_op = ddp_config_.overlap_param_gather && (!force_sync);

    for (auto i = 0; i < buckets_.size(); ++i) {
        auto bucket = buckets_[i];
        std::shared_ptr<Tensor> param_buffer = bucket->param_data();
        if (!param_buffer) {
            continue;
        }

        if (param_buffer_shard_list_[i].empty()) {
            param_buffer_shard_list_[i] = ShardBuffer(param_buffer, collective_pg_size_);
        }
        auto local_data_view = param_buffer_shard_list_[i][rank_in_collective_pg_];
        param_gather_work_list_.push_back(collective_pg_->AllGather(param_buffer, local_data_view, async_op));
    }

    param_gather_dispatched_ = true;
}

void ParamAndGradBucketGroup::FinishParamSync(bool skip_next_bucket_dispatch) {
    if (!ddp_config_.use_distributed_optimizer || !ddp_config_.overlap_param_gather) {
        return;
    }

    if (!param_gather_dispatched_) {
        StartParamSync();
    }

    if (!param_gather_work_list_.empty()) {
        for (auto work : param_gather_work_list_) { work->WaitNonBlocking(); }
        param_gather_work_list_.clear();
        param_gather_dispatched_ = false;

        if (next_param_gather_bucket_group_ && !skip_next_bucket_dispatch) {
            if (next_param_gather_bucket_group_->param_gather_dispatched_) {
                LOG(WARNING)
                    << "ParamAndGradBucketGroup: The next bucket's parameter all-gather operation has already been "
                       "dispatched. This may be caused by a mismatch between the order of parameter registration and "
                       "forward pass execution, which will hurt the communication - computation overlap performance.";
            } else {
                next_param_gather_bucket_group_->StartParamSync();
            }
        }
    }
}

void ParamAndGradBucketGroup::SetNextParamGatherBucketGroup(std::shared_ptr<ParamAndGradBucketGroup> next_group) {
    next_param_gather_bucket_group_ = next_group;
}

ParamAndGradBuffer::ParamAndGradBuffer(const std::vector<std::shared_ptr<Tensor>> &params, DataType &param_dtype,
                                       DataType &grad_dtype, const ProcessGroup *ddp_pg,
                                       DistributedDataParallelConfig ddp_config)
    : params_(std::move(params)), ddp_pg_(std::move(ddp_pg)), ddp_config_(ddp_config) {
    if (ddp_pg_) {
        ddp_world_size_ = global::GetDataParallelSize();
    }

    grads_.clear();
    grads_.resize(params_.size());

    BuildBuckets(param_dtype, grad_dtype);
}

void ParamAndGradBuffer::BuildBuckets(DataType param_dtype, DataType grad_dtype) {
    // Pack parameters in buffer, allocate memory, and build buckets.

    // Param start must be multiple of 64
    auto PadParamStartIfNeeded = [&](size_t start) -> size_t {
        if (ddp_config_.use_distributed_optimizer) {
            // According to Megatron-LM, make sure each param starts at 128B aligned address (by default align to 64
            // elements for precision >=16-bit）
            return PadTo(start, kParamStartAlignElements);
        }
        return start;
    };

    // Bucket size shoule be multiple of ddp size and 128 (sweet spot for NCCL)
    auto PadBucketEndIfNeeded = [&](size_t bucket_end_index) -> size_t {
        if (ddp_config_.use_distributed_optimizer) {
            // According to Megatron-LM, ensure that all buckets start at a memory address that is 256B
            // aligned(128 values since params and grads use >= 16-bit precision)
            size_t lcm_val = std::lcm(ddp_world_size_, kBucketEndAlignElements);
            if (ddp_config_.pad_buckets_for_high_nccl_busbw) {
                // According to Megatron-LM, when the bucket size is divisible by a large power of 2 (2^16),
                // NCCL collectives can have high bus bandwidth at large DP counts
                lcm_val = std::lcm(lcm_val, static_cast<size_t>(1u << 16));
            }
            return PadTo(bucket_end_index, lcm_val);
        }
        return bucket_end_index;
    };

    size_t param_start_index = 0;
    size_t param_end_index = 0;
    size_t bucket_start_index = 0;
    size_t bucket_end_index = 0;
    size_t bucket_id = 0;
    std::vector<std::shared_ptr<Tensor>> bucket_params;
    std::vector<size_t> per_bucket_numel_unpadded;

    auto UpdateBucketMetadata = [&](size_t param_end_index) -> size_t {
        // calculate numel when bucket is unpadded
        const size_t numel_unpadded_bucket = param_end_index - bucket_start_index;
        per_bucket_numel_unpadded.push_back(numel_unpadded_bucket);

        // calculate bucket_end_index with padding, save the range of bucket in buffer
        size_t bucket_end_index = PadBucketEndIfNeeded(param_end_index);
        bucket_indices_.push_back({bucket_start_index, bucket_end_index});

        // move ptr to next bucket
        bucket_start_index = bucket_end_index;
        bucket_params.clear();
        ++bucket_id;
        return bucket_end_index;
    };

    // 1. Pack params into buffer, in backprop order
    for (auto it = params_.rbegin(); it != params_.rend(); ++it) {
        const auto &param = *it;
        param_start_index = PadParamStartIfNeeded(param_start_index);

        // TODO(zbl): check whether there are params that need its own bucket
        // if (DoesParamRequiresNewBucket(param)) { ... }

        param_end_index = param_start_index + param->NumElements();
        param_index_map_[param.get()] = {param_start_index, param_end_index, bucket_id};
        bucket_params.push_back(param);

        if ((param_end_index - bucket_start_index) >= ddp_config_.bucket_size_in_elements) {
            // If current bucket is full, then wrap up
            // NOTE(zbl): Actual bucket size might be larger than bucket size
            bucket_end_index = UpdateBucketMetadata(param_end_index);
            param_start_index = bucket_end_index;
        } else {
            param_start_index = param_end_index;
        }
    }

    // If the last bucket is not full, still wrap it up
    if (!bucket_params.empty()) {
        bucket_end_index = UpdateBucketMetadata(param_end_index);
    }

    // numel with padding = bucket end
    numel_ = bucket_end_index;
    // numel without padding
    numel_unpadded_ = std::accumulate(per_bucket_numel_unpadded.begin(), per_bucket_numel_unpadded.end(),
                                      static_cast<size_t>(0), std::plus<size_t>());

    CHECK(numel_unpadded_ <= numel_);
    if (ddp_config_.use_distributed_optimizer) {
        // numel must be multiple of ddp size (so that reduce-scatter could easily shard the buffer among ranks)
        CHECK_EQ(numel_ % ddp_world_size_, 0);
    } else {
        CHECK_EQ(numel_, numel_unpadded_);
    }

    // 2. Allocate buffer
    auto device = params_.front()->GetDevice();
    if (ddp_config_.use_distributed_optimizer) {
        param_buffer_ = AllocateFlatBuffer(numel_, param_dtype, device);
    } else {
        // No param buffer needed if optimzer is not distributed
        param_buffer_.reset();
    }
    grad_buffer_ = AllocateFlatBuffer(numel_, grad_dtype, device);

    LOG(INFO) << "ParamAndGradBuffer: numel_unpadded=" << numel_unpadded_ << ", numel (padded)=" << numel_;

    // 3. Build buckets, and map param/grad to views of buffers
    bucket_params.clear();
    bucket_start_index = 0;
    size_t current_bucket_id = 0;

    // Helper function to create ParamAndGradBucket object
    auto NewBucket
        = [&](const std::vector<std::shared_ptr<Tensor>> &bucket_params, size_t start_index, size_t end_index,
              size_t num_elements_unpadded, size_t bucket_id) -> std::shared_ptr<ParamAndGradBucket> {
        if (ddp_config_.use_distributed_optimizer) {
            CHECK_EQ(start_index % ddp_world_size_, 0);
            CHECK_EQ(end_index % ddp_world_size_, 0);
            CHECK_EQ(bucket_indices_.at(bucket_id).first, start_index);
            CHECK_EQ(bucket_indices_.at(bucket_id).second, end_index);
        }

        std::shared_ptr<Tensor> bucket_param_view;
        if (param_buffer_) {
            bucket_param_view = GetBufferView(param_buffer_, start_index,
                                              std::vector<int64_t>{static_cast<int64_t>(end_index - start_index)});
        }
        std::shared_ptr<Tensor> bucket_grad_view = GetBufferView(
            grad_buffer_, start_index, std::vector<int64_t>{static_cast<int64_t>(end_index - start_index)});

        // FIXME(zbl): Use default for now
        float gradient_scaling_factor = 1.0f;
        auto bucket
            = std::make_shared<ParamAndGradBucket>(bucket_params, bucket_param_view, bucket_grad_view, start_index,
                                                   num_elements_unpadded, gradient_scaling_factor, bucket_id);

        for (auto param : bucket_params) {
            CHECK(param_bucket_map_.find(param.get()) == param_bucket_map_.end())
                << "Parameter appears in multiple buckets.";
            param_bucket_map_[param.get()] = bucket;
        }

        return std::move(bucket);
    };

    size_t i = params_.size();
    // Iterate params in backprop order, build ParamAndGradBucket object
    for (auto it = params_.rbegin(); it != params_.rend(); ++it) {
        const auto &param = *it;
        std::tie(param_start_index, param_end_index, bucket_id) = param_index_map_.at(param.get());

        // Remap param/grad pointers
        if (param_buffer_) {
            // FIXME(zbl): change tensor buffer
            param->SetData(*param_buffer_, param_start_index * kDataTypeToSize.at(param_buffer_->Dtype()), true);
        }

        auto grad_view = GetBufferView(grad_buffer_, param_start_index, param->Dims());
        param->set_grad(grad_view);
        // Save grad view for each params
        --i;
        grads_[i] = grad_view;

        if (current_bucket_id != bucket_id) {
            const auto &range = bucket_indices_.at(current_bucket_id);
            CHECK_EQ(range.first, bucket_start_index) << "Bucket start mismatch.";

            bucket_end_index = range.second;
            buckets_.push_back(NewBucket(bucket_params, bucket_start_index, bucket_end_index,
                                         per_bucket_numel_unpadded[current_bucket_id], current_bucket_id));

            bucket_start_index = bucket_end_index;
            bucket_params.clear();

            CHECK_EQ(current_bucket_id + 1, buckets_.size());
            CHECK_EQ(current_bucket_id + 1, bucket_id);
            current_bucket_id = bucket_id;
        }
        bucket_params.push_back(param);
    }

    // If the last bucket is not full, still wrap it up
    if (!bucket_params.empty()) {
        const auto &range = bucket_indices_.at(current_bucket_id);
        CHECK_EQ(range.first, bucket_start_index) << "Last bucket start mismatch.";

        bucket_end_index = range.second;
        buckets_.push_back(NewBucket(bucket_params, bucket_start_index, bucket_end_index,
                                     per_bucket_numel_unpadded[current_bucket_id], current_bucket_id));
    }
}

void ParamAndGradBuffer::ScaleGradients(float scaling_factor) {
    if (!grad_buffer_ || scaling_factor == 1.f) {
        return;
    }

    // FIXME(zbl): should perform in-place multiply
    // grad_data_ *= scaling_factor;
    LOG(FATAL) << "Should not arrive here";
}

void ParamAndGradBuffer::Reset(bool need_rebind) {
    if (!grad_buffer_) {
        return;
    }
    if (!need_rebind) {
        grad_buffer_->Fill(0.f);
    }
    need_rebind_grad_views_ = need_rebind;
}

void ParamAndGradBuffer::RebindGradViews() {
    if (!need_rebind_grad_views_) {
        return;
    }

    CHECK_EQ(params_.size(), grads_.size());
    for (size_t i = 0; i < params_.size(); ++i) {
        params_[i]->set_grad(grads_[i]);
        params_[i]->MarkGradOverwriteOnNextAccum();
    }

    need_rebind_grad_views_ = false;
}

// Partition ParamAndGradBuckets into DDP ParamAndGradBucketGroups.
// - force_single_bucket_group: all buckets across all buffers -> one group.
// - otherwise: each bucket -> its own group (no merging).
// TODO(zbl): support cross-buffer merging for mixed fp8/non-fp8 scenarios.
// Ref: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/distributed/param_and_grad_buffer.py
std::vector<std::shared_ptr<ParamAndGradBucketGroup>>
PartitionBuckets(const std::vector<std::shared_ptr<ParamAndGradBuffer>> &buffers, bool force_single_bucket_group) {
    std::vector<std::shared_ptr<ParamAndGradBucketGroup>> bucket_groups;

    if (buffers.empty()) {
        return bucket_groups;
    }

    // Case 1: Put all buckets into a single bucket group if force_single_bucket_group is True.
    if (force_single_bucket_group) {
        std::vector<std::shared_ptr<ParamAndGradBucket>> all_buckets;
        auto ddp_config = buffers.front()->ddp_config();
        auto ddp_pg = buffers.front()->ddp_pg();
        auto ddp_world_size = buffers.front()->ddp_world_size();

        for (const auto &buffer : buffers) {
            CHECK(buffer->ddp_pg() == ddp_pg) << "PartitionBuckets: buffers have different ddp_pg.";
            CHECK(buffer->ddp_world_size() == ddp_world_size)
                << "PartitionBuckets: buffers have different ddp_world_size.";

            all_buckets.insert(all_buckets.end(), buffer->buckets().begin(), buffer->buckets().end());
        }

        bucket_groups.push_back(
            std::make_shared<ParamAndGradBucketGroup>(all_buckets, ddp_pg, ddp_world_size, ddp_config));
        return bucket_groups;
    }

    // Case 2: When there is no fp8 buffer in the input buffers, let each bucket group have only one bucket.
    for (const auto &buffer : buffers) {
        const auto &buffer_buckets = buffer->buckets();
        for (const auto &bucket : buffer_buckets) {
            std::vector<std::shared_ptr<ParamAndGradBucket>> single_bucket_list;
            single_bucket_list.push_back(bucket);
            bucket_groups.push_back(std::make_shared<ParamAndGradBucketGroup>(
                single_bucket_list, buffer->ddp_pg(), buffer->ddp_world_size(), buffer->ddp_config()));
        }
    }

    // TODO(zbl): Support fp8 params
    // Case 3: When using fp8 params, merge all non-fp8 buckets into the last fp8 bucket group.
    return bucket_groups;
}

} // namespace infini_train::nn::parallel
