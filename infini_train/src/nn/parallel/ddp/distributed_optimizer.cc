#include "infini_train/include/nn/parallel/ddp/distributed_optimizer.h"

#include <cmath>
#include <limits>
#include <unordered_set>

#include "glog/logging.h"

#include "infini_train/include/nn/parallel/ddp/distributed_data_parallel.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/tensor_parallel.h"
#include "infini_train/include/nn/parallel/utils.h"
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
        [this, &parameter_name_by_tensor, &shard_named_parameters](const std::shared_ptr<Tensor> &parameter,
                                                             const std::shared_ptr<Tensor> &param_piece) {
            const auto name_it = parameter_name_by_tensor.find(parameter.get());
            CHECK(name_it != parameter_name_by_tensor.end())
                << "DistributedOptimizer parameter is not registered in the model";
            shard_named_parameters.emplace_back(name_it->second, param_piece);
            shard_params_.push_back(param_piece);
            shard_param_owners_.push_back(parameter);
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
        if (global::GetTensorParallelSize() > 1) {
            for (const auto &[_, module] : ddp_chunk->module()->NamedModules()) {
                auto row_parallel = std::dynamic_pointer_cast<RowParallelLinear>(module);
                if (row_parallel && row_parallel->bias()) {
                    tp_replicated_param_owners_.insert(
                        row_parallel->parameter(RowParallelLinear::kParamBiasName).get());
                }
            }
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

    // A rank may legitimately own no non-padding elements when a parameter is
    // smaller than the DP shard size. Keep an empty base optimizer on that
    // rank; global norm reduction still participates with a zero local stat.
    (void)num_shard_params;
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

std::shared_ptr<Tensor> DistributedOptimizer::ClipGradNorm_(
    const std::vector<std::shared_ptr<Tensor>> &parameters, float max_norm, float norm_type,
    bool error_if_nonfinite, std::optional<bool> foreach) {
    CHECK_GE(max_norm, 0.0f) << "max_norm must be non-negative.";
    CHECK((norm_type > 0.0f && std::isfinite(norm_type)) || norm_type == std::numeric_limits<float>::infinity())
        << "norm_type must be positive finite or +inf.";
    if (foreach.value_or(false)) {
        LOG(FATAL) << "DistributedOptimizer: foreach=true is not implemented.";
    }

    FinishGradSync();

    std::unordered_set<const Tensor *> requested;
    for (const auto &parameter : parameters) {
        if (parameter) {
            requested.insert(parameter.get());
        }
    }
    std::vector<std::shared_ptr<Tensor>> selected_shards;
    std::vector<std::shared_ptr<Tensor>> norm_shards;
    for (size_t i = 0; i < shard_params_.size(); ++i) {
        if (requested.empty() || requested.contains(shard_param_owners_[i].get())) {
            selected_shards.push_back(shard_params_[i]);
            bool include_in_norm = true;
            if (global::GetTensorParallelSize() > 1 &&
                tp_replicated_param_owners_.contains(shard_param_owners_[i].get())) {
                const auto *tp_group = ProcessGroupFactory::Instance(shard_params_[i]->GetDevice().type())
                                            ->Get(GetTensorParallelProcessGroupName(
                                                shard_params_[i]->GetDevice().Rank().GlobalRank()));
                CHECK(tp_group) << "Tensor-parallel process group is not initialized.";
                include_in_norm = tp_group->GetGroupRank(shard_params_[i]->GetDevice().Rank().GlobalRank()) == 0;
            }
            if (include_in_norm) {
                norm_shards.push_back(shard_params_[i]);
            }
        }
    }

    // Ask the base optimizer for the local norm without scaling.  Infinite
    // max_norm makes its coefficient exactly one for finite gradients.
    auto local_norm_tensor = base_optimizer_->ClipGradNorm_(
        norm_shards, std::numeric_limits<float>::infinity(), norm_type, false, std::nullopt);
    const float local_norm = *static_cast<const float *>(local_norm_tensor->DataPtr());
    const bool is_inf_norm = std::isinf(norm_type);
    double local_stat = is_inf_norm ? static_cast<double>(local_norm)
                                    : std::pow(static_cast<double>(local_norm), static_cast<double>(norm_type));

    const ProcessGroup *group = nullptr;
    for (const auto &bucket_group : bucket_groups_) {
        if (bucket_group->collective_pg()) {
            group = bucket_group->collective_pg();
            break;
        }
    }
    Device total_norm_device = selected_shards.empty() ? Device() : selected_shards.front()->GetDevice();
    if (selected_shards.empty()) {
        for (const auto &bucket_group : bucket_groups_) {
            if (!bucket_group->buckets().empty() && bucket_group->buckets().front()->param_data()) {
                total_norm_device = bucket_group->buckets().front()->param_data()->GetDevice();
                break;
            }
        }
    }
    auto reduced = std::make_shared<Tensor>(std::vector<int64_t>{}, DataType::kFLOAT32, total_norm_device);
    reduced->Fill(static_cast<float>(local_stat));
    if (group && ddp_world_size_ > 1) {
        group->AllReduce(reduced, is_inf_norm ? function::ReduceOpType::kMax : function::ReduceOpType::kSum, false);
    }
    if (global::GetTensorParallelSize() > 1) {
        const auto *tp_group = ProcessGroupFactory::Instance(total_norm_device.type())->Get(
            GetTensorParallelProcessGroupName(total_norm_device.Rank().GlobalRank()));
        CHECK(tp_group) << "Tensor-parallel process group is not initialized.";
        tp_group->AllReduce(reduced, is_inf_norm ? function::ReduceOpType::kMax : function::ReduceOpType::kSum, false);
    }
    if (nn::parallel::global::GetPipelineParallelSize() > 1) {
        const auto *pp_group = ProcessGroupFactory::Instance(total_norm_device.type())->Get(
            GetPipelineParallelProcessGroupName(total_norm_device.Rank().GlobalRank()));
        CHECK(pp_group) << "Pipeline process group is not initialized.";
        pp_group->AllReduce(reduced, is_inf_norm ? function::ReduceOpType::kMax : function::ReduceOpType::kSum, false);
    }
    Tensor reduced_cpu = reduced->GetDevice().IsCPU() ? Tensor(*reduced, 0, reduced->Dims()) : reduced->To(Device());
    const float reduced_value = *static_cast<const float *>(reduced_cpu.DataPtr());
    const double total_norm = is_inf_norm ? static_cast<double>(reduced_value)
                                          : std::pow(static_cast<double>(reduced_value), 1.0 / norm_type);
    if (error_if_nonfinite && !std::isfinite(total_norm)) {
        LOG(FATAL) << "The total gradient norm is non-finite.";
    }

    const double coefficient = std::min(static_cast<double>(max_norm) / (total_norm + 1e-6), 1.0);
    base_optimizer_->ScaleGradients_(selected_shards, static_cast<float>(coefficient));

    auto result = std::make_shared<Tensor>(std::vector<int64_t>{}, DataType::kFLOAT32, Device());
    *static_cast<float *>(result->DataPtr()) = static_cast<float>(total_norm);
    return result;
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

    // 2. Base optimizer step on owned param pieces
    CHECK(base_optimizer_) << "DistributedOptimizer: base optimizer is null.";
    base_optimizer_->Step();

    // 3. Gather updated param shards back to full params
    StartParamSync(/*force_sync=*/false);
    // TODO(zbl): Delay sync call until param is actually used in next step
    FinishParamSync(/*skip_next_bucket_dispatch=*/true);
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
