#include "infini_train/include/nn/parallel/distributed_optimizer.h"

#include "glog/logging.h"

#include "infini_train/include/device.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::parallel {

namespace {
std::shared_ptr<Tensor> GetShardView(const std::shared_ptr<Tensor> &buffer, size_t world_size, size_t rank) {

    CHECK(buffer);
    CHECK_GT(world_size, 0);
    CHECK_LT(rank, world_size);
    CHECK_EQ(buffer->NumElements() % world_size, 0);

    const size_t shard_numel = buffer->NumElements() / world_size;
    const size_t offset_bytes = shard_numel * rank * kDataTypeToSize.at(buffer->Dtype());

    return std::make_shared<Tensor>(*buffer, offset_bytes, std::vector<int64_t>{static_cast<int64_t>(shard_numel)});
}

} // namespace

DistributedOptimizer::DistributedOptimizer(OptimizerCreator creator,
                                           const std::vector<std::shared_ptr<Tensor>> &full_params,
                                           const std::vector<std::shared_ptr<ParamAndGradBuffer>> &buffers,
                                           const std::vector<std::shared_ptr<ParamAndGradBucketGroup>> &bucket_groups,
                                           const ProcessGroup *dp_pg, size_t dp_world_size, size_t dp_rank)
    : Optimizer(full_params), param_grad_buffers_(buffers), bucket_groups_(bucket_groups), dp_pg_(dp_pg),
      dp_world_size_(dp_world_size), dp_rank_(dp_rank), creator_(std::move(creator)) {

    CHECK(dp_pg_);

    BuildShardParamsAndBindGrads();

    // Build base optimizer
    base_optimizer_ = creator_(shard_params_);
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
    // Zero main_grad buffer and clear BucketGroup state
    for (auto &buffer : param_grad_buffers_) { buffer->Reset(); }
    for (auto &group : bucket_groups_) { group->Reset(); }
    // Call base class's method: Zero each param's grad to guarantee consistency
    infini_train::Optimizer::ZeroGrad(set_to_none);
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
