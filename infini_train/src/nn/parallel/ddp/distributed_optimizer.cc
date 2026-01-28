#include "infini_train/include/nn/parallel/ddp/distributed_optimizer.h"

#include "glog/logging.h"

#include "infini_train/include/nn/parallel/ddp/distributed_data_parallel.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::parallel {
DistributedOptimizer::DistributedOptimizer(OptimizerCreator creator,
                                           const std::vector<std::shared_ptr<Tensor>> &full_params,
                                           const std::vector<std::shared_ptr<Module>> &model_chunks,
                                           size_t dp_world_size, size_t dp_rank)
    : Optimizer(full_params), dp_world_size_(dp_world_size), dp_rank_(dp_rank) {

    CHECK(dp_world_size_ > 1) << "DistributedOptimizer: dp_world_size must be greater than 1.";

    for (size_t i = 0; i < model_chunks.size(); ++i) {
        auto ddp_chunk = std::dynamic_pointer_cast<DistributedDataParallel>(model_chunks[i]);
        CHECK(ddp_chunk) << "DistributedOptimizer: model_chunks[" << i << "] is not a DDP model.";

        param_grad_buffers_.insert(param_grad_buffers_.end(), ddp_chunk->param_grad_buffers().begin(),
                                   ddp_chunk->param_grad_buffers().end());
        bucket_groups_.insert(bucket_groups_.end(), ddp_chunk->bucket_groups().begin(),
                              ddp_chunk->bucket_groups().end());
    }

    BuildShardParamsAndBindGrads();

    // Build base optimizer
    base_optimizer_ = creator(shard_params_);
    CHECK(base_optimizer_) << "DistributedOptimizer: failed to create base optimizer.";
}

void DistributedOptimizer::BuildShardParamsAndBindGrads() {
    shard_params_.clear();

    for (const auto &group : bucket_groups_) {
        for (const auto &bucket : group->buckets()) {

            auto bucket_param = bucket->param_data();
            auto bucket_grad = bucket->grad_data();

            CHECK(bucket_param) << "DistributedOptimizer requires param buffer.";
            CHECK(bucket_grad) << "DistributedOptimizer requires grad buffer.";

            CHECK_EQ(bucket_param->NumElements() % dp_world_size_, 0);
            const size_t bucket_shard_numel = bucket_param->NumElements() / dp_world_size_;
            const size_t bucket_shard_start = dp_rank_ * bucket_shard_numel;
            const size_t bucket_shard_end = bucket_shard_start + bucket_shard_numel;

            // Iterate param in bucket, build each param(or param_shard) seperately
            for (const auto &param : bucket->params()) {
                size_t param_start_in_bucket = 0, param_end_in_bucket = 0;
                auto found = bucket->GetTensorLocInBucket(param, param_start_in_bucket, param_end_in_bucket);
                CHECK(found) << "DistributedOptimizer: param not found in bucket mapping.";

                const size_t local_start = std::max(param_start_in_bucket, bucket_shard_start);
                const size_t local_end = std::min(param_end_in_bucket, bucket_shard_end);
                if (local_end <= local_start) {
                    // this rank owns no elements for this param
                    continue;
                }

                const size_t piece_numel = local_end - local_start;
                CHECK_GT(piece_numel, 0);

                const size_t param_piece_offset_bytes = local_start * kDataTypeToSize.at(bucket_param->Dtype());
                const size_t grad_piece_offset_bytes = local_start * kDataTypeToSize.at(bucket_grad->Dtype());

                auto param_piece = std::make_shared<Tensor>(*bucket_param, param_piece_offset_bytes,
                                                            std::vector<int64_t>{static_cast<int64_t>(piece_numel)});

                auto grad_piece = std::make_shared<Tensor>(*bucket_grad, grad_piece_offset_bytes,
                                                           std::vector<int64_t>{static_cast<int64_t>(piece_numel)});

                param_piece->set_grad(grad_piece);
                shard_params_.push_back(param_piece);
            }
        }
    }

    CHECK(!shard_params_.empty()) << "DistributedOptimizer: this DP rank owns no param pieces. "
                                  << "Check bucket padding/divisibility and param bucketing order.";
}

void DistributedOptimizer::StartGradSync() {
    for (auto &group : bucket_groups_) { group->StartGradSync(); }
}

void DistributedOptimizer::FinishGradSync() {
    for (auto &group : bucket_groups_) { group->FinishGradSync(); }
}

void DistributedOptimizer::StartParamSync(bool force_sync) {
    for (auto &group : bucket_groups_) { group->StartParamSync(force_sync); }
}

void DistributedOptimizer::FinishParamSync(bool skip_next_bucket_dispatch) {
    for (auto &group : bucket_groups_) { group->FinishParamSync(skip_next_bucket_dispatch); }
}

void DistributedOptimizer::ZeroGrad(bool set_to_none) {
    // Clear BucketGroup state and reset buffer:
    // If set_to_none is true:
    //   1) buffers will not be zeroed,
    //   2) each of full_params's tensor->grad() will be set to nullptr
    // If set_to_none is false:
    //   1) buffers will be zeroed,
    //   2) do not perform Fill(0) for each param
    for (auto &buffer : param_grad_buffers_) { buffer->Reset(set_to_none); }
    for (auto &group : bucket_groups_) { group->Reset(); }
    if (set_to_none) {
        for (auto param : params_) { param->ZeroGrad(set_to_none); }
    }
}

void DistributedOptimizer::Step() {
    // 1. Ensure grads are synced
    FinishGradSync();

    // 2. Base optimizer step on owned param pieces
    CHECK(base_optimizer_) << "DistributedOptimizer: base optimizer is null.";
    base_optimizer_->Step();

    // 3. Gather updated param shards back to full params
    StartParamSync(/*force_sync=*/false);
    // FIXME(zbl): Call sync before param is actually used in next step
    FinishParamSync(/*skip_next_bucket_dispatch=*/true);
}

} // namespace infini_train::nn::parallel
