#include "infini_train/include/nn/lora/lora_parallel_linear.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/autograd/linear.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/tensor_parallel.h"
#include "infini_train/include/nn/parallel/utils.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::lora {

// ============================================================================
// LoRAColumnParallelLinear Implementation
// ============================================================================

LoRAColumnParallelLinear::LoRAColumnParallelLinear(std::shared_ptr<nn::Module> base_module, const LoRAConfig &config,
                                                   int64_t in_features, int64_t out_features)
    : CloneableModule(kType), config_(config), in_features_(in_features), out_features_(out_features) {
    if (!base_module) {
        throw std::invalid_argument("base_module cannot be null");
    }

    base_module_ = base_module;

    // Get TP config from base module
    auto col_linear = std::dynamic_pointer_cast<parallel::ColumnParallelLinear>(base_module);
    CHECK(col_linear != nullptr) << "base_module must be ColumnParallelLinear";
    bias_ = col_linear->bias();
    gather_output_ = col_linear->gather_output();
    input_is_parallel_ = col_linear->input_is_parallel();
    skip_bias_add_ = col_linear->skip_bias_add();
    sequence_parallel_ = col_linear->sequence_parallel();

    // Get device from base module
    device_ = base_module->parameter(parallel::ColumnParallelLinear::kParamWeightName)->GetDevice();

    // Transfer weight from base module
    parameters_[kParamWeightName] = base_module->parameter(parallel::ColumnParallelLinear::kParamWeightName);

    // Get dimensions from weight shape [out_features_per_partition, in_features]
    const auto &weight_dims = parameters_[kParamWeightName]->Dims();
    out_features_per_partition_ = weight_dims[0];

    // Transfer bias if exists
    if (base_module->has_parameter(parallel::ColumnParallelLinear::kParamBiasName)) {
        parameters_[kParamBiasName] = base_module->parameter(parallel::ColumnParallelLinear::kParamBiasName);
    }

    // Initialize LoRA weights
    InitLoRAWeights();

    // Freeze base weights
    FreezeBaseWeights();
}

LoRAColumnParallelLinear::LoRAColumnParallelLinear(std::shared_ptr<nn::Module> base_module, const LoRAConfig &config)
    : CloneableModule(kType), config_(config) {
    if (!base_module) {
        throw std::invalid_argument("base_module cannot be null");
    }

    base_module_ = base_module;

    // Get TP config from base module
    auto col_linear = std::dynamic_pointer_cast<parallel::ColumnParallelLinear>(base_module);
    CHECK(col_linear != nullptr) << "base_module must be ColumnParallelLinear";
    bias_ = col_linear->bias();
    gather_output_ = col_linear->gather_output();
    input_is_parallel_ = col_linear->input_is_parallel();
    skip_bias_add_ = col_linear->skip_bias_add();
    sequence_parallel_ = col_linear->sequence_parallel();

    // Get device from base module
    device_ = base_module->parameter(parallel::ColumnParallelLinear::kParamWeightName)->GetDevice();

    // Transfer weight from base module
    parameters_[kParamWeightName] = base_module->parameter(parallel::ColumnParallelLinear::kParamWeightName);

    // Get dimensions from weight shape [out_features_per_partition, in_features]
    const auto &weight_dims = parameters_[kParamWeightName]->Dims();
    out_features_per_partition_ = weight_dims[0];
    in_features_ = weight_dims[1];

    // Calculate total out_features (assuming tensor parallelism)
    int tp_size = parallel::global::GetTensorParallelSize();
    out_features_ = out_features_per_partition_ * tp_size;

    // Transfer bias if exists
    if (base_module->has_parameter(parallel::ColumnParallelLinear::kParamBiasName)) {
        parameters_[kParamBiasName] = base_module->parameter(parallel::ColumnParallelLinear::kParamBiasName);
    }

    // Initialize LoRA weights
    InitLoRAWeights();

    // Freeze base weights
    FreezeBaseWeights();
}

void LoRAColumnParallelLinear::InitLoRAWeights() {
    // LoRA weights stored directly in parameters_
    // Following PEFT pattern conceptually:
    // lora_A: [rank, in_features] - replicated
    // lora_B: [out_features_per_partition, rank] - sharded like base weight

    // lora_A: [rank, in_features]
    parameters_[kParamLoraAName]
        = std::make_shared<Tensor>(std::vector<int64_t>{config_.rank, in_features_}, DataType::kFLOAT32, device_)
              ->RequiresGrad();
    if (config_.use_kaiming_a) {
        init::KaimingUniform(parameters_[kParamLoraAName], config_.kaiming_a_param);
    } else {
        init::Normal(parameters_[kParamLoraAName], 0.0f, 0.02f);
    }

    // lora_B: [out_per_partition, rank] - sharded like base weight
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

    if (!merged_) {
        // 1. Compute base output via base_module
        // base_module handles all TP/SP communication internally
        auto base_result = base_module_->Forward(input_tensors);
        auto base_output = base_result[0];

        // 2. Compute LoRA output using the SAME input that base module uses
        // Match base input path exactly: use direct input if input_is_parallel_ or sequence_parallel_,
        // otherwise copy to TP region
        auto lora_input = (input_is_parallel_ || sequence_parallel_)
                            ? input_tensors[0]
                            : parallel::CopyToTPRegionFunc(input_tensors[0])[0];
        if (sequence_parallel_) {
            // Base uses GatherFromSPRegionFunc to gather sequence dimension
            lora_input = parallel::GatherFromSPRegionFunc(lora_input)[0];
        }

        // Compute LoRA: lora_A: [rank, in_features], lora_B: [out_per_partition, rank]
        auto lora_proj = std::make_shared<autograd::Linear>()->Apply({lora_input, parameters_[kParamLoraAName]})[0];
        auto lora_output = std::make_shared<autograd::Linear>()->Apply({lora_proj, parameters_[kParamLoraBName]})[0];

        // Match base output layout (gather if base gathers)
        if (gather_output_) {
            lora_output = parallel::GatherFromTPRegionFunc(lora_output)[0];
        }

        auto scaled_lora = lora_output->Mul(config_.Scaling());

        // 3. Add LoRA contribution to base output
        // Both should now have the same sequence dimension
        auto output = base_output->Add(scaled_lora);

        // Return in same format as base module
        return skip_bias_add_
                 ? std::vector<std::shared_ptr<Tensor>>{output, bias_ ? parameters_[kParamBiasName] : nullptr}
                 : std::vector<std::shared_ptr<Tensor>>{output};
    }

    // When merged, delegate to base module
    return base_module_->Forward(input_tensors);
}

void LoRAColumnParallelLinear::MergeWeights() {
    if (merged_) {
        return;
    }

    original_weight_ = std::make_shared<Tensor>(*parameters_[kParamWeightName]);

    // W' = W + (alpha/r) * B @ A
    auto delta = parameters_[kParamLoraBName]->Matmul(parameters_[kParamLoraAName]);
    auto scaled_delta = delta->Mul(config_.Scaling());
    auto new_weight = parameters_[kParamWeightName]->Add(scaled_delta);
    parameters_[kParamWeightName]->CopyFrom(new_weight);

    merged_ = true;
}

void LoRAColumnParallelLinear::UnmergeWeights() {
    if (!merged_ || !original_weight_) {
        return;
    }
    parameters_[kParamWeightName]->CopyFrom(original_weight_);
    merged_ = false;
}

std::vector<std::shared_ptr<Tensor>> LoRAColumnParallelLinear::LoRAParameters() const {
    return {parameters_.at(kParamLoraAName), parameters_.at(kParamLoraBName)};
}

std::vector<std::shared_ptr<Tensor>> LoRAColumnParallelLinear::Parameters() const { return AllParameters(); }

std::vector<std::shared_ptr<Tensor>> LoRAColumnParallelLinear::AllParameters() const {
    std::vector<std::shared_ptr<Tensor>> all_params;
    all_params.push_back(parameters_.at(kParamWeightName));
    if (bias_) {
        all_params.push_back(parameters_.at(kParamBiasName));
    }
    all_params.push_back(parameters_.at(kParamLoraAName));
    all_params.push_back(parameters_.at(kParamLoraBName));
    return all_params;
}

std::vector<std::shared_ptr<Tensor>> LoRAColumnParallelLinear::TrainableParameters() const {
    std::vector<std::shared_ptr<Tensor>> trainable;
    for (const auto &[name, param] : parameters_) {
        if (param->requires_grad()) {
            trainable.push_back(param);
        }
    }
    return trainable;
}

bool LoRAColumnParallelLinear::IsMerged() const { return merged_; }

int64_t LoRAColumnParallelLinear::in_features() const { return in_features_; }

int64_t LoRAColumnParallelLinear::out_features() const { return out_features_; }

int64_t LoRAColumnParallelLinear::rank() const { return config_.rank; }

// ============================================================================
// LoRARowParallelLinear Implementation
// ============================================================================

LoRARowParallelLinear::LoRARowParallelLinear(std::shared_ptr<nn::Module> base_module, const LoRAConfig &config,
                                             int64_t in_features, int64_t out_features)
    : CloneableModule(kType), config_(config), in_features_(in_features), out_features_(out_features) {
    if (!base_module) {
        throw std::invalid_argument("base_module cannot be null");
    }

    base_module_ = base_module;

    // Get TP config from base module
    auto row_linear = std::dynamic_pointer_cast<parallel::RowParallelLinear>(base_module);
    CHECK(row_linear != nullptr) << "base_module must be RowParallelLinear";
    bias_ = row_linear->bias();
    reduce_output_ = row_linear->reduce_output();
    input_is_parallel_ = row_linear->input_is_parallel();
    skip_bias_add_ = row_linear->skip_bias_add();
    sequence_parallel_ = row_linear->sequence_parallel();

    // Get device from base module
    device_ = base_module->parameter(parallel::RowParallelLinear::kParamWeightName)->GetDevice();

    // Transfer weight from base module
    parameters_[kParamWeightName] = base_module->parameter(parallel::RowParallelLinear::kParamWeightName);

    // Get dimensions from weight shape [out_features, in_features_per_partition]
    const auto &weight_dims = parameters_[kParamWeightName]->Dims();
    in_features_per_partition_ = weight_dims[1];

    // Transfer bias if exists
    if (base_module->has_parameter(parallel::RowParallelLinear::kParamBiasName)) {
        parameters_[kParamBiasName] = base_module->parameter(parallel::RowParallelLinear::kParamBiasName);
    }

    // Initialize LoRA weights
    InitLoRAWeights();

    // Freeze base weights
    FreezeBaseWeights();
}

LoRARowParallelLinear::LoRARowParallelLinear(std::shared_ptr<nn::Module> base_module, const LoRAConfig &config)
    : CloneableModule(kType), config_(config) {
    if (!base_module) {
        throw std::invalid_argument("base_module cannot be null");
    }

    base_module_ = base_module;

    // Get TP config from base module
    auto row_linear = std::dynamic_pointer_cast<parallel::RowParallelLinear>(base_module);
    CHECK(row_linear != nullptr) << "base_module must be RowParallelLinear";
    bias_ = row_linear->bias();
    reduce_output_ = row_linear->reduce_output();
    input_is_parallel_ = row_linear->input_is_parallel();
    skip_bias_add_ = row_linear->skip_bias_add();
    sequence_parallel_ = row_linear->sequence_parallel();

    // Get device from base module
    device_ = base_module->parameter(parallel::RowParallelLinear::kParamWeightName)->GetDevice();

    // Transfer weight from base module
    parameters_[kParamWeightName] = base_module->parameter(parallel::RowParallelLinear::kParamWeightName);

    // Get dimensions from weight shape [out_features, in_features_per_partition]
    const auto &weight_dims = parameters_[kParamWeightName]->Dims();
    out_features_ = weight_dims[0];
    in_features_per_partition_ = weight_dims[1];

    // Calculate total in_features (assuming tensor parallelism)
    int tp_size = parallel::global::GetTensorParallelSize();
    in_features_ = in_features_per_partition_ * tp_size;

    // Transfer bias if exists
    if (base_module->has_parameter(parallel::RowParallelLinear::kParamBiasName)) {
        parameters_[kParamBiasName] = base_module->parameter(parallel::RowParallelLinear::kParamBiasName);
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

    if (!merged_) {
        // Get effective input - match what base module uses
        auto effective_input = input_tensors[0];
        const int64_t in_dim = effective_input->Dims().back();

        if (!input_is_parallel_) {
            // base would scatter; lora must match
            effective_input = parallel::ScatterToTPRegionFunc(effective_input)[0];
            CHECK_EQ(effective_input->Dims().back(), in_features_per_partition_);
        } else {
            // input_is_parallel_=true means caller promised shard input
            CHECK_EQ(in_dim, in_features_per_partition_)
                << "RowParallel expects sharded input when input_is_parallel_=true. "
                << "Got full in_dim=" << in_dim << " (likely upstream gathered TP output).";
        }

        // 1) base output - use effective_input
        auto base_result = base_module_->Forward({effective_input});
        auto base_output = base_result[0];

        // 2) lora branch uses the SAME effective_input
        auto lora_proj
            = std::make_shared<autograd::Linear>()->Apply({effective_input, parameters_[kParamLoraAName]})[0];
        auto lora_output = std::make_shared<autograd::Linear>()->Apply({lora_proj, parameters_[kParamLoraBName]})[0];

        // 3) apply same reduction as base
        auto lora_out = lora_output;
        if (reduce_output_) {
            lora_out = sequence_parallel_ ? parallel::ReduceScatterToSPRegionFunc(lora_out)[0]
                                          : parallel::ReduceFromTPRegionFunc(lora_out)[0];
        }

        auto scaled_lora = lora_out->Mul(config_.Scaling());
        CHECK_EQ(base_output->NumElements(), scaled_lora->NumElements());
        auto output = base_output->Add(scaled_lora);

        // Return in same format as base module
        return skip_bias_add_
                 ? std::vector<std::shared_ptr<Tensor>>{output, bias_ ? parameters_[kParamBiasName] : nullptr}
                 : std::vector<std::shared_ptr<Tensor>>{output};
    }

    // When merged, delegate to base module
    return base_module_->Forward(input_tensors);
}

void LoRARowParallelLinear::MergeWeights() {
    if (merged_) {
        return;
    }

    original_weight_ = std::make_shared<Tensor>(*parameters_[kParamWeightName]);

    // W' = W + (alpha/r) * B @ A
    auto delta = parameters_[kParamLoraBName]->Matmul(parameters_[kParamLoraAName]);
    auto scaled_delta = delta->Mul(config_.Scaling());
    auto new_weight = parameters_[kParamWeightName]->Add(scaled_delta);
    parameters_[kParamWeightName]->CopyFrom(new_weight);

    merged_ = true;
}

void LoRARowParallelLinear::UnmergeWeights() {
    if (!merged_ || !original_weight_) {
        return;
    }
    parameters_[kParamWeightName]->CopyFrom(original_weight_);
    merged_ = false;
}

std::vector<std::shared_ptr<Tensor>> LoRARowParallelLinear::LoRAParameters() const {
    return {parameters_.at(kParamLoraAName), parameters_.at(kParamLoraBName)};
}

std::vector<std::shared_ptr<Tensor>> LoRARowParallelLinear::Parameters() const { return AllParameters(); }

std::vector<std::shared_ptr<Tensor>> LoRARowParallelLinear::AllParameters() const {
    std::vector<std::shared_ptr<Tensor>> all_params;
    all_params.push_back(parameters_.at(kParamWeightName));
    if (bias_) {
        all_params.push_back(parameters_.at(kParamBiasName));
    }
    all_params.push_back(parameters_.at(kParamLoraAName));
    all_params.push_back(parameters_.at(kParamLoraBName));
    return all_params;
}

std::vector<std::shared_ptr<Tensor>> LoRARowParallelLinear::TrainableParameters() const {
    std::vector<std::shared_ptr<Tensor>> trainable;
    for (const auto &[name, param] : parameters_) {
        if (param->requires_grad()) {
            trainable.push_back(param);
        }
    }
    return trainable;
}

bool LoRARowParallelLinear::IsMerged() const { return merged_; }

int64_t LoRARowParallelLinear::in_features() const { return in_features_; }

int64_t LoRARowParallelLinear::out_features() const { return out_features_; }

int64_t LoRARowParallelLinear::rank() const { return config_.rank; }

} // namespace infini_train::nn::lora
