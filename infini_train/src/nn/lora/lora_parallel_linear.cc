#include "infini_train/include/nn/lora/lora_parallel_linear.h"

#include <cmath>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/autograd/linear.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/process_group.h"
#include "infini_train/include/nn/parallel/tensor_parallel.h"
#include "infini_train/include/nn/parallel/utils.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::lora {

// ============================================================================
// LoRAColumnParallelLinear Implementation
// ============================================================================

LoRAColumnParallelLinear::LoRAColumnParallelLinear(std::shared_ptr<parallel::ColumnParallelLinear> base_module,
                                                   const LoRAConfig &config, int64_t in_features, int64_t out_features)
    : ColumnParallelLinear(in_features, out_features, base_module->bias(), base_module->gather_output(),
                           base_module->input_is_parallel(), base_module->skip_bias_add(),
                           base_module->sequence_parallel()),
      config_(config), in_features_(in_features), out_features_(out_features) {
    CHECK(base_module != nullptr) << "base_module cannot be null";

    // Get device from base module
    device_ = base_module->parameter(kParamWeightName)->GetDevice();

    // Transfer weight from base module (overwrite base-created one)
    parameters_[kParamWeightName] = base_module->parameter(kParamWeightName);

    // Get dimensions from weight shape [out_features_per_partition, in_features]
    const auto &weight_dims = parameters_[kParamWeightName]->Dims();
    out_features_per_partition_ = weight_dims[0];

    // Transfer bias if exists
    if (base_module->has_parameter(kParamBiasName)) {
        parameters_[kParamBiasName] = base_module->parameter(kParamBiasName);
    }

    // Initialize LoRA weights
    InitLoRAWeights();

    // Freeze base weights
    FreezeBaseWeights();
}

LoRAColumnParallelLinear::LoRAColumnParallelLinear(std::shared_ptr<parallel::ColumnParallelLinear> base_module,
                                                   const LoRAConfig &config)
    : ColumnParallelLinear(base_module->parameter(parallel::ColumnParallelLinear::kParamWeightName)->Dims()[1],
                           base_module->parameter(parallel::ColumnParallelLinear::kParamWeightName)->Dims()[0]
                               * parallel::global::GetTensorParallelSize(),
                           base_module->bias(), base_module->gather_output(), base_module->input_is_parallel(),
                           base_module->skip_bias_add(), base_module->sequence_parallel()),
      config_(config) {
    CHECK(base_module != nullptr) << "base_module cannot be null";

    // Get device from base module
    device_ = base_module->parameter(kParamWeightName)->GetDevice();

    // Transfer weight from base module (overwrite base-created one)
    parameters_[kParamWeightName] = base_module->parameter(kParamWeightName);

    // Get dimensions from weight shape [out_features_per_partition, in_features]
    const auto &weight_dims = parameters_[kParamWeightName]->Dims();
    out_features_per_partition_ = weight_dims[0];
    in_features_ = weight_dims[1];

    // Calculate total out_features (assuming tensor parallelism)
    int tp_size = parallel::global::GetTensorParallelSize();
    out_features_ = out_features_per_partition_ * tp_size;

    // Transfer bias if exists
    if (base_module->has_parameter(kParamBiasName)) {
        parameters_[kParamBiasName] = base_module->parameter(kParamBiasName);
    }

    // Initialize LoRA weights
    InitLoRAWeights();

    // Freeze base weights
    FreezeBaseWeights();
}

void LoRAColumnParallelLinear::InitLoRAWeights() {
    // lora_A: [rank, in_features] - replicated across TP ranks
    // lora_B: [out_features_per_partition, rank] - sharded like base weight
    parameters_[kParamLoraAName]
        = std::make_shared<Tensor>(std::vector<int64_t>{config_.rank, in_features_}, DataType::kFLOAT32, device_)
              ->RequiresGrad();

    if (parallel::global::GetTensorParallelSize() > 1) {
        const auto global_rank = device_.Rank().GlobalRank();
        auto *tp_group = parallel::ProcessGroupFactory::Instance(device_.type())
                             ->Get(parallel::GetTensorParallelProcessGroupName(global_rank));
        const int tp_rank = tp_group->GetGroupRank(global_rank);

        // Only TP rank 0 generates random values; others zero-init.
        // AllReduce(sum) then broadcasts rank-0's values to all TP ranks.
        if (tp_rank == 0) {
            if (config_.use_kaiming_a) {
                init::KaimingUniform(parameters_[kParamLoraAName], config_.kaiming_a_param);
            } else {
                init::Normal(parameters_[kParamLoraAName], 0.0f, 0.02f);
            }
        } else {
            init::Zeros(parameters_[kParamLoraAName]);
        }
        tp_group->AllReduce(parameters_[kParamLoraAName]);
    } else {
        if (config_.use_kaiming_a) {
            init::KaimingUniform(parameters_[kParamLoraAName], config_.kaiming_a_param);
        } else {
            init::Normal(parameters_[kParamLoraAName], 0.0f, 0.02f);
        }
    }

    parameters_[kParamLoraBName]
        = std::make_shared<Tensor>(std::vector<int64_t>{out_features_per_partition_, config_.rank}, DataType::kFLOAT32,
                                   device_)
              ->RequiresGrad();
    init::Zeros(parameters_[kParamLoraBName]);
}

void LoRAColumnParallelLinear::FreezeBaseWeights() {
    parameters_[kParamWeightName]->set_requires_grad(false);
    if (bias_) {
        parameters_[kParamBiasName]->set_requires_grad(false);
    }
}

std::vector<std::shared_ptr<Tensor>>
LoRAColumnParallelLinear::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1) << "LoRAColumnParallelLinear takes exactly one input";
    CHECK(!(merged_ && parameters_.at(kParamLoraAName)->requires_grad()))
        << "Forward() on merged LoRA with requires_grad=true. Call UnmergeWeights() before training.";

    if (!merged_) {
        // Inline base + LoRA matmuls, add locally, then single collective op.
        // This avoids 2 separate AllGather ops which cause floating-point divergence.
        auto input = (input_is_parallel_ || sequence_parallel_) ? input_tensors[0]
                                                                : parallel::CopyToTPRegionFunc(input_tensors[0])[0];
        if (sequence_parallel_) {
            input = parallel::GatherFromSPRegionFunc(input)[0];
        }

        // Base matmul (bias folded in when applicable, matching ColumnParallelLinear::Forward)
        auto base_shard = std::make_shared<autograd::Linear>()->Apply(
            (bias_ && !skip_bias_add_)
                ? std::vector<std::shared_ptr<Tensor>>{input, parameters_.at(kParamWeightName),
                                                       parameters_[kParamBiasName]}
                : std::vector<std::shared_ptr<Tensor>>{input, parameters_.at(kParamWeightName)})[0];

        // LoRA matmul (local)
        // Wrap replicated lora_A through CopyToTPRegion so its gradient gets AllReduced in backward
        auto lora_A = parallel::CopyToTPRegionFunc(parameters_[kParamLoraAName])[0];
        auto lora_proj = std::make_shared<autograd::Linear>()->Apply({input, lora_A})[0];
        auto lora_output = std::make_shared<autograd::Linear>()->Apply({lora_proj, parameters_[kParamLoraBName]})[0];

        // Local add before collective
        auto combined = base_shard->Add(lora_output->Mul(config_.Scaling()));

        // Single collective op
        auto output = gather_output_ ? parallel::GatherFromTPRegionFunc(combined)[0] : combined;

        return skip_bias_add_
                 ? std::vector<std::shared_ptr<Tensor>>{output, bias_ ? parameters_.at(kParamBiasName) : nullptr}
                 : std::vector<std::shared_ptr<Tensor>>{output};
    }

    // When merged, delegate to base class
    return ColumnParallelLinear::Forward(input_tensors);
}

void LoRAColumnParallelLinear::MergeWeights() {
    if (merged_) {
        return;
    }

    // W' = W + (alpha/r) * B @ A
    auto delta = parameters_[kParamLoraBName]->Matmul(parameters_[kParamLoraAName]);
    auto scaled_delta = delta->Mul(config_.Scaling());
    auto new_weight = parameters_[kParamWeightName]->Add(scaled_delta);
    parameters_[kParamWeightName]->CopyFrom(new_weight);

    // Freeze LoRA params to prevent training while merged
    parameters_[kParamLoraAName]->set_requires_grad(false);
    parameters_[kParamLoraBName]->set_requires_grad(false);

    merged_ = true;
}

void LoRAColumnParallelLinear::UnmergeWeights() {
    if (!merged_) {
        return;
    }

    // W = W - (alpha/r) * B @ A
    auto delta = parameters_[kParamLoraBName]->Matmul(parameters_[kParamLoraAName]);
    auto scaled_delta = delta->Mul(config_.Scaling());
    auto new_weight = parameters_[kParamWeightName]->Sub(scaled_delta);
    parameters_[kParamWeightName]->CopyFrom(new_weight);

    // Restore LoRA params to trainable
    parameters_[kParamLoraAName]->set_requires_grad(true);
    parameters_[kParamLoraBName]->set_requires_grad(true);

    merged_ = false;
}

std::vector<std::shared_ptr<Tensor>> LoRAColumnParallelLinear::LoRAParameters() const {
    return {parameters_.at(kParamLoraAName), parameters_.at(kParamLoraBName)};
}

bool LoRAColumnParallelLinear::IsMerged() const { return merged_; }

int64_t LoRAColumnParallelLinear::in_features() const { return in_features_; }

int64_t LoRAColumnParallelLinear::out_features() const { return out_features_; }

int64_t LoRAColumnParallelLinear::rank() const { return config_.rank; }

// ============================================================================
// LoRARowParallelLinear Implementation
// ============================================================================

LoRARowParallelLinear::LoRARowParallelLinear(std::shared_ptr<parallel::RowParallelLinear> base_module,
                                             const LoRAConfig &config, int64_t in_features, int64_t out_features)
    : RowParallelLinear(in_features, out_features, base_module->bias(), base_module->reduce_output(),
                        base_module->input_is_parallel(), base_module->skip_bias_add(),
                        base_module->sequence_parallel()),
      config_(config), in_features_(in_features), out_features_(out_features) {
    CHECK(base_module != nullptr) << "base_module cannot be null";

    // Get device from base module
    device_ = base_module->parameter(kParamWeightName)->GetDevice();

    // Transfer weight from base module (overwrite base-created one)
    parameters_[kParamWeightName] = base_module->parameter(kParamWeightName);

    // Get dimensions from weight shape [out_features, in_features_per_partition]
    const auto &weight_dims = parameters_[kParamWeightName]->Dims();
    in_features_per_partition_ = weight_dims[1];

    // Transfer bias if exists
    if (base_module->has_parameter(kParamBiasName)) {
        parameters_[kParamBiasName] = base_module->parameter(kParamBiasName);
    }

    // Initialize LoRA weights
    InitLoRAWeights();

    // Freeze base weights
    FreezeBaseWeights();
}

LoRARowParallelLinear::LoRARowParallelLinear(std::shared_ptr<parallel::RowParallelLinear> base_module,
                                             const LoRAConfig &config)
    : RowParallelLinear(base_module->parameter(parallel::RowParallelLinear::kParamWeightName)->Dims()[1]
                            * parallel::global::GetTensorParallelSize(),
                        base_module->parameter(parallel::RowParallelLinear::kParamWeightName)->Dims()[0],
                        base_module->bias(), base_module->reduce_output(), base_module->input_is_parallel(),
                        base_module->skip_bias_add(), base_module->sequence_parallel()),
      config_(config) {
    CHECK(base_module != nullptr) << "base_module cannot be null";

    // Get device from base module
    device_ = base_module->parameter(kParamWeightName)->GetDevice();

    // Transfer weight from base module (overwrite base-created one)
    parameters_[kParamWeightName] = base_module->parameter(kParamWeightName);

    // Get dimensions from weight shape [out_features, in_features_per_partition]
    const auto &weight_dims = parameters_[kParamWeightName]->Dims();
    out_features_ = weight_dims[0];
    in_features_per_partition_ = weight_dims[1];

    // Calculate total in_features (assuming tensor parallelism)
    int tp_size = parallel::global::GetTensorParallelSize();
    in_features_ = in_features_per_partition_ * tp_size;

    // Transfer bias if exists
    if (base_module->has_parameter(kParamBiasName)) {
        parameters_[kParamBiasName] = base_module->parameter(kParamBiasName);
    }

    // Initialize LoRA weights
    InitLoRAWeights();

    // Freeze base weights
    FreezeBaseWeights();
}

void LoRARowParallelLinear::InitLoRAWeights() {
    // lora_A: [rank, in_features_per_partition] - sharded
    // lora_B: [out_features, rank] - replicated

    // lora_A: [rank, in_features_per_partition]
    parameters_[kParamLoraAName]
        = std::make_shared<Tensor>(std::vector<int64_t>{config_.rank, in_features_per_partition_}, DataType::kFLOAT32,
                                   device_)
              ->RequiresGrad();
    if (config_.use_kaiming_a) {
        init::KaimingUniform(parameters_[kParamLoraAName], config_.kaiming_a_param);
    } else {
        init::Normal(parameters_[kParamLoraAName], 0.0f, 0.02f);
    }

    // lora_B: [out_features, rank]
    parameters_[kParamLoraBName]
        = std::make_shared<Tensor>(std::vector<int64_t>{out_features_, config_.rank}, DataType::kFLOAT32, device_)
              ->RequiresGrad();
    init::Zeros(parameters_[kParamLoraBName]);
}

void LoRARowParallelLinear::FreezeBaseWeights() {
    parameters_[kParamWeightName]->set_requires_grad(false);
    if (bias_) {
        parameters_[kParamBiasName]->set_requires_grad(false);
    }
}

std::vector<std::shared_ptr<Tensor>>
LoRARowParallelLinear::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1) << "LoRARowParallelLinear takes exactly one input";
    CHECK(!(merged_ && parameters_.at(kParamLoraAName)->requires_grad()))
        << "Forward() on merged LoRA with requires_grad=true. Call UnmergeWeights() before training.";

    if (!merged_) {
        // Inline base + LoRA matmuls, add locally, then single collective op.
        // This avoids 2 separate AllReduce ops which cause floating-point divergence.
        auto input = input_is_parallel_ ? input_tensors[0] : parallel::ScatterToTPRegionFunc(input_tensors[0])[0];

        // Base matmul (no bias — RowParallel adds bias AFTER collective)
        auto base_shard = std::make_shared<autograd::Linear>()->Apply({input, parameters_.at(kParamWeightName)})[0];

        // LoRA matmul (local)
        // Wrap replicated lora_B through CopyToTPRegion so its gradient gets AllReduced in backward
        auto lora_proj = std::make_shared<autograd::Linear>()->Apply({input, parameters_[kParamLoraAName]})[0];
        auto lora_B = parallel::CopyToTPRegionFunc(parameters_[kParamLoraBName])[0];
        auto lora_output = std::make_shared<autograd::Linear>()->Apply({lora_proj, lora_B})[0];

        // Local add before collective
        auto combined = base_shard->Add(lora_output->Mul(config_.Scaling()));

        // Single collective op
        auto output = reduce_output_ ? (sequence_parallel_ ? parallel::ReduceScatterToSPRegionFunc(combined)[0]
                                                           : parallel::ReduceFromTPRegionFunc(combined)[0])
                                     : combined;

        // Bias after collective (matching RowParallelLinear::Forward)
        if (bias_ && !skip_bias_add_) {
            output = output->Add(parameters_[kParamBiasName]);
        }

        return skip_bias_add_
                 ? std::vector<std::shared_ptr<Tensor>>{output, bias_ ? parameters_[kParamBiasName] : nullptr}
                 : std::vector<std::shared_ptr<Tensor>>{output};
    }

    // When merged, delegate to base class
    return RowParallelLinear::Forward(input_tensors);
}

void LoRARowParallelLinear::MergeWeights() {
    if (merged_) {
        return;
    }

    // W' = W + (alpha/r) * B @ A
    auto delta = parameters_[kParamLoraBName]->Matmul(parameters_[kParamLoraAName]);
    auto scaled_delta = delta->Mul(config_.Scaling());
    auto new_weight = parameters_[kParamWeightName]->Add(scaled_delta);
    parameters_[kParamWeightName]->CopyFrom(new_weight);

    // Freeze LoRA params to prevent training while merged
    parameters_[kParamLoraAName]->set_requires_grad(false);
    parameters_[kParamLoraBName]->set_requires_grad(false);

    merged_ = true;
}

void LoRARowParallelLinear::UnmergeWeights() {
    if (!merged_) {
        return;
    }

    // W = W - (alpha/r) * B @ A
    auto delta = parameters_[kParamLoraBName]->Matmul(parameters_[kParamLoraAName]);
    auto scaled_delta = delta->Mul(config_.Scaling());
    auto new_weight = parameters_[kParamWeightName]->Sub(scaled_delta);
    parameters_[kParamWeightName]->CopyFrom(new_weight);

    // Restore LoRA params to trainable
    parameters_[kParamLoraAName]->set_requires_grad(true);
    parameters_[kParamLoraBName]->set_requires_grad(true);

    merged_ = false;
}

std::vector<std::shared_ptr<Tensor>> LoRARowParallelLinear::LoRAParameters() const {
    return {parameters_.at(kParamLoraAName), parameters_.at(kParamLoraBName)};
}

bool LoRARowParallelLinear::IsMerged() const { return merged_; }

int64_t LoRARowParallelLinear::in_features() const { return in_features_; }

int64_t LoRARowParallelLinear::out_features() const { return out_features_; }

int64_t LoRARowParallelLinear::rank() const { return config_.rank; }

} // namespace infini_train::nn::lora
