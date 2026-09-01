#include "infini_train/include/nn/parallel/ddp/distributed_optimizer.h"

#include "glog/logging.h"

#include "infini_train/include/nn/parallel/ddp/distributed_data_parallel.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::parallel {
DistributedOptimizer::DistributedOptimizer(OptimizerCreator creator,
                                           const std::vector<std::shared_ptr<Tensor>> &full_params,
                                           const std::vector<std::shared_ptr<Module>> &model_chunks,
                                           size_t ddp_world_size, size_t ddp_rank)
    : Optimizer(full_params, /*learning_rate=*/0.0f), ddp_world_size_(ddp_world_size), ddp_rank_(ddp_rank) {
    InitializeModelChunks(model_chunks);

    std::vector<std::shared_ptr<Tensor>> shard_params;
    BuildShardParamsAndBindGrads(
        [&shard_params](const std::shared_ptr<Tensor> &, const std::shared_ptr<Tensor> &param_piece) {
            shard_params.push_back(param_piece);
        });

    base_optimizer_ = creator(shard_params);
    CHECK(base_optimizer_) << "DistributedOptimizer: failed to create base optimizer.";
}

DistributedOptimizer::DistributedOptimizer(OptimizerCreatorNamed creator, const NamedParameterList &named_parameters,
                                           const std::vector<std::shared_ptr<Module>> &model_chunks,
                                           size_t ddp_world_size, size_t ddp_rank)
    : Optimizer(named_parameters, /*learning_rate=*/0.0f), ddp_world_size_(ddp_world_size), ddp_rank_(ddp_rank) {
    InitializeModelChunks(model_chunks);

    std::unordered_map<const Tensor *, std::string> parameter_name_by_tensor;
    parameter_name_by_tensor.reserve(named_parameters.size());
    for (const auto &[name, parameter] : named_parameters) {
        CHECK(parameter);
        parameter_name_by_tensor.emplace(parameter.get(), name);
    }

    NamedParameterList shard_named_parameters;
    BuildShardParamsAndBindGrads(
        [&parameter_name_by_tensor, &shard_named_parameters](const std::shared_ptr<Tensor> &parameter,
                                                             const std::shared_ptr<Tensor> &param_piece) {
            const auto name_it = parameter_name_by_tensor.find(parameter.get());
            CHECK(name_it != parameter_name_by_tensor.end())
                << "DistributedOptimizer parameter is not registered in the model";
            shard_named_parameters.emplace_back(name_it->second, param_piece);
        });

    base_optimizer_ = creator(shard_named_parameters);
    CHECK(base_optimizer_) << "DistributedOptimizer: failed to create base optimizer.";
}

void DistributedOptimizer::InitializeModelChunks(const std::vector<std::shared_ptr<Module>> &model_chunks) {
    CHECK(ddp_world_size_ > 1) << "DistributedOptimizer: ddp_world_size must be greater than 1.";

    for (size_t i = 0; i < model_chunks.size(); ++i) {
        auto ddp_chunk = std::dynamic_pointer_cast<DistributedDataParallel>(model_chunks[i]);
        CHECK(ddp_chunk) << "DistributedOptimizer: model_chunks[" << i << "] is not a DDP model.";

        param_grad_buffers_.insert(param_grad_buffers_.end(), ddp_chunk->param_grad_buffers().begin(),
                                   ddp_chunk->param_grad_buffers().end());
        bucket_groups_.insert(bucket_groups_.end(), ddp_chunk->bucket_groups().begin(),
                              ddp_chunk->bucket_groups().end());
        if (!ddp_chunk->bucket_groups().empty()) {
            first_param_sync_bucket_groups_.push_back(ddp_chunk->bucket_groups().back());
        }
    }
}

void DistributedOptimizer::BuildShardParamsAndBindGrads(const AddShardParam &add_shard_param) {
    size_t num_shard_params = 0;

    for (const auto &group : bucket_groups_) {
        const bool use_grad_shard = group->config().zero_stage >= 2;
        const auto &buckets = group->buckets();
        for (size_t bucket_idx = 0; bucket_idx < buckets.size(); ++bucket_idx) {
            const auto &bucket = buckets[bucket_idx];

            auto bucket_param = bucket->param_data();
            auto bucket_grad = use_grad_shard ? group->GetLocalGradShardBuffer(bucket_idx) : bucket->grad_data();

            CHECK(bucket_param) << "DistributedOptimizer requires param buffer.";
            CHECK(bucket_grad) << "DistributedOptimizer requires grad buffer.";

            CHECK_EQ(bucket_param->NumElements() % ddp_world_size_, 0);
            const size_t bucket_shard_numel = bucket_param->NumElements() / ddp_world_size_;
            const size_t bucket_shard_start = ddp_rank_ * bucket_shard_numel;
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
                // Adjust the offset since bucket_grad is already the shard of grad under ZeRO-2.
                auto offset = use_grad_shard ? (local_start - bucket_shard_start) : local_start;
                size_t grad_piece_offset_bytes = offset * kDataTypeToSize.at(bucket_grad->Dtype());

                auto param_piece = std::make_shared<Tensor>(*bucket_param, param_piece_offset_bytes,
                                                            std::vector<int64_t>{static_cast<int64_t>(piece_numel)});

                auto grad_piece = std::make_shared<Tensor>(*bucket_grad, grad_piece_offset_bytes,
                                                           std::vector<int64_t>{static_cast<int64_t>(piece_numel)});

                param_piece->set_grad(grad_piece);
                // NOTE(zbl): Do not call `param->set_grad(grad_piece);` under ZeRO-2.
                //            The base optimizer updates param_piece views only; original param->grad()
                //            would be a partial flattened shard and does not represent the full parameter grad.
                add_shard_param(param, param_piece);
                ++num_shard_params;
            }
        }
    }

    CHECK_GT(num_shard_params, 0) << "DistributedOptimizer: this DP rank owns no param pieces. "
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

void DistributedOptimizer::set_learning_rate(float lr) {
    Optimizer::set_learning_rate(lr);
    if (base_optimizer_) {
        base_optimizer_->set_learning_rate(lr);
    }
}

float DistributedOptimizer::learning_rate() const {
    if (base_optimizer_) {
        return base_optimizer_->learning_rate();
    }
    return Optimizer::learning_rate();
}

void DistributedOptimizer::Step() {
    // 1. Ensure grads are synced
    FinishGradSync();

    // Parameter gathers from the previous update must finish before the optimizer writes into the same buffers.
    for (auto &group : bucket_groups_) { group->PrepareParamSyncForNextStep(); }

    // 2. Base optimizer step on owned param pieces
    CHECK(base_optimizer_) << "DistributedOptimizer: base optimizer is null.";
    base_optimizer_->Step();

    // 3. Publish updated parameter shards. With overlap enabled, only launch the first gather in each model
    // chunk. Forward pre-hooks wait on it at first use and dispatch subsequent bucket gathers.
    CHECK(!bucket_groups_.empty());
    if (bucket_groups_.front()->config().overlap_param_gather) {
        for (auto &group : first_param_sync_bucket_groups_) { group->StartParamSync(); }
    } else {
        StartParamSync(/*force_sync=*/false);
    }
}

std::unordered_map<std::string, std::shared_ptr<Tensor>> DistributedOptimizer::StateDict() const {
    CHECK(base_optimizer_) << "DistributedOptimizer: base optimizer is null.";
    return base_optimizer_->StateDict();
}

void DistributedOptimizer::LoadStateDict(const std::unordered_map<std::string, std::shared_ptr<Tensor>> &state_dict) {
    CHECK(base_optimizer_) << "DistributedOptimizer: base optimizer is null.";
    base_optimizer_->LoadStateDict(state_dict);
}
} // namespace infini_train::nn::parallel
