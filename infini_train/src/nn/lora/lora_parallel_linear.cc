#include "infini_train/include/nn/lora/lora_parallel_linear.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/autograd/linear.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/tensor_parallel.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::lora {

// ============================================================================
// LoRAColumnParallelLinear Implementation
// ============================================================================

LoRAColumnParallelLinear::LoRAColumnParallelLinear(std::shared_ptr<nn::Module> base_module, const LoRAConfig &config,
                                                   int64_t in_features, int64_t out_features)
    : CloneableModule(kType), config_(config), in_features_(in_features), out_features_(out_features), bias_(false),
      gather_output_(false), input_is_parallel_(false), skip_bias_add_(false), sequence_parallel_(false) {
    if (!base_module) {
        throw std::invalid_argument("base_module cannot be null");
    }

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
        bias_ = true;
    }

    // Initialize LoRA weights
    InitLoRAWeights();

    // Freeze base weights
    FreezeBaseWeights();
}

LoRAColumnParallelLinear::LoRAColumnParallelLinear(std::shared_ptr<nn::Module> base_module, const LoRAConfig &config)
    : CloneableModule(kType), config_(config), bias_(false), gather_output_(false), input_is_parallel_(false),
      skip_bias_add_(false), sequence_parallel_(false) {
    if (!base_module) {
        throw std::invalid_argument("base_module cannot be null");
    }

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
        bias_ = true;
    }

    // Initialize LoRA weights
    InitLoRAWeights();

    // Freeze base weights
    FreezeBaseWeights();
}

void LoRAColumnParallelLinear::InitLoRAWeights() {
    // A matrix: [rank, in_features] - replicated across TP ranks
    parameters_[kParamLoraAName]
        = std::make_shared<Tensor>(std::vector<int64_t>{config_.rank, in_features_}, DataType::kFLOAT32, device_)
              ->RequiresGrad();

    if (config_.use_kaiming_a) {
        init::KaimingUniform(parameters_[kParamLoraAName], config_.kaiming_a_param);
    } else {
        init::Normal(parameters_[kParamLoraAName], 0.0f, 0.02f);
    }

    // B matrix: [out_features_per_partition, rank] - sharded like base weight
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
    const auto &input = input_tensors[0];

    // Base linear computation
    auto base_output = std::make_shared<autograd::Linear>()->Apply(
        (bias_ && !skip_bias_add_)
            ? std::vector<std::shared_ptr<Tensor>>{input, parameters_[kParamWeightName], parameters_[kParamBiasName]}
            : std::vector<std::shared_ptr<Tensor>>{input, parameters_[kParamWeightName]})[0];

    if (!merged_) {
        // LoRA computation: x @ A^T @ B^T
        // A is replicated [rank, in_features], so x @ A^T gives same result on all ranks
        auto hidden = std::make_shared<autograd::Linear>()->Apply({input, parameters_[kParamLoraAName]})[0];

        // B is sharded [out_features_per_partition, rank], so hidden @ B^T gives sharded output
        auto lora_output = std::make_shared<autograd::Linear>()->Apply({hidden, parameters_[kParamLoraBName]})[0];

        // Scale and add
        float scaling = config_.Scaling();
        auto scaled_lora = lora_output->Mul(scaling);
        base_output = base_output->Add(scaled_lora);
    }

    return skip_bias_add_
             ? std::vector<std::shared_ptr<Tensor>>{base_output, bias_ ? parameters_.at(kParamBiasName) : nullptr}
             : std::vector<std::shared_ptr<Tensor>>{base_output};
}

void LoRAColumnParallelLinear::MergeWeights() {
    if (merged_) {
        return;
    }

    original_weight_ = std::make_shared<Tensor>(*parameters_[kParamWeightName]);

    // W' = W + (alpha/r) * B @ A
    // W: [out_features_per_partition, in_features]
    // B: [out_features_per_partition, rank]
    // A: [rank, in_features]
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

std::vector<std::shared_ptr<Tensor>> LoRAColumnParallelLinear::Parameters() const { return LoRAParameters(); }

bool LoRAColumnParallelLinear::IsMerged() const { return merged_; }

int64_t LoRAColumnParallelLinear::in_features() const { return in_features_; }

int64_t LoRAColumnParallelLinear::out_features() const { return out_features_; }

int64_t LoRAColumnParallelLinear::rank() const { return config_.rank; }

// ============================================================================
// LoRARowParallelLinear Implementation
// ============================================================================

LoRARowParallelLinear::LoRARowParallelLinear(std::shared_ptr<nn::Module> base_module, const LoRAConfig &config,
                                             int64_t in_features, int64_t out_features)
    : CloneableModule(kType), config_(config), in_features_(in_features), out_features_(out_features), bias_(false),
      reduce_output_(false), input_is_parallel_(false), skip_bias_add_(false), sequence_parallel_(false) {
    if (!base_module) {
        throw std::invalid_argument("base_module cannot be null");
    }

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
        bias_ = true;
    }

    // Initialize LoRA weights
    InitLoRAWeights();

    // Freeze base weights
    FreezeBaseWeights();
}

LoRARowParallelLinear::LoRARowParallelLinear(std::shared_ptr<nn::Module> base_module, const LoRAConfig &config)
    : CloneableModule(kType), config_(config), bias_(false), reduce_output_(false), input_is_parallel_(false),
      skip_bias_add_(false), sequence_parallel_(false) {
    if (!base_module) {
        throw std::invalid_argument("base_module cannot be null");
    }

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
        bias_ = true;
    }

    // Initialize LoRA weights
    InitLoRAWeights();

    // Freeze base weights
    FreezeBaseWeights();
}

void LoRARowParallelLinear::InitLoRAWeights() {
    // A matrix: [rank, in_features_per_partition] - sharded like base weight
    parameters_[kParamLoraAName]
        = std::make_shared<Tensor>(std::vector<int64_t>{config_.rank, in_features_per_partition_}, DataType::kFLOAT32,
                                   device_)
              ->RequiresGrad();

    if (config_.use_kaiming_a) {
        init::KaimingUniform(parameters_[kParamLoraAName], config_.kaiming_a_param);
    } else {
        init::Normal(parameters_[kParamLoraAName], 0.0f, 0.02f);
    }

    // B matrix: [out_features, rank] - replicated across TP ranks
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
    const auto &input = input_tensors[0];

    // Base linear computation (local matmul)
    auto base_output = std::make_shared<autograd::Linear>()->Apply({input, parameters_[kParamWeightName]})[0];

    if (!merged_) {
        // LoRA computation for RowParallel:
        // A is sharded [rank, in_features_per_partition]
        // x_local @ A_local^T gives partial result [batch, seq, rank]
        auto hidden_local = std::make_shared<autograd::Linear>()->Apply({input, parameters_[kParamLoraAName]})[0];

        // For RowParallel, we need to sum the partial results from all ranks
        // This is handled by the reduce operation that follows
        // B is replicated [out_features, rank]
        auto lora_output = std::make_shared<autograd::Linear>()->Apply({hidden_local, parameters_[kParamLoraBName]})[0];

        // Scale and add to base output (before reduce)
        float scaling = config_.Scaling();
        auto scaled_lora = lora_output->Mul(scaling);
        base_output = base_output->Add(scaled_lora);
    }

    // Handle bias
    if (bias_ && !skip_bias_add_) {
        base_output = base_output->Add(parameters_[kParamBiasName]);
    }

    return skip_bias_add_
             ? std::vector<std::shared_ptr<Tensor>>{base_output, bias_ ? parameters_.at(kParamBiasName) : nullptr}
             : std::vector<std::shared_ptr<Tensor>>{base_output};
}

void LoRARowParallelLinear::MergeWeights() {
    if (merged_) {
        return;
    }

    original_weight_ = std::make_shared<Tensor>(*parameters_[kParamWeightName]);

    // W' = W + (alpha/r) * B @ A
    // W: [out_features, in_features_per_partition]
    // B: [out_features, rank]
    // A: [rank, in_features_per_partition]
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

std::vector<std::shared_ptr<Tensor>> LoRARowParallelLinear::Parameters() const { return LoRAParameters(); }

bool LoRARowParallelLinear::IsMerged() const { return merged_; }

int64_t LoRARowParallelLinear::in_features() const { return in_features_; }

int64_t LoRARowParallelLinear::out_features() const { return out_features_; }

int64_t LoRARowParallelLinear::rank() const { return config_.rank; }

} // namespace infini_train::nn::lora
