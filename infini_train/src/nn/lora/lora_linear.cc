#include "infini_train/include/nn/lora/lora_linear.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include "infini_train/include/autograd/linear.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::lora {

LoRALinear::LoRALinear(int64_t in_features, int64_t out_features, const LoRAConfig &config, bool bias,
                       const Device *device)
    : CloneableModule(kType), config_(config), in_features_(in_features), out_features_(out_features), bias_(bias) {
    device_ = device ? *device : Device();

    // Create base weight (frozen)
    parameters_[kParamWeightName]
        = std::make_shared<Tensor>(std::vector<int64_t>{out_features, in_features}, DataType::kFLOAT32, device_);
    init::KaimingUniform(parameters_[kParamWeightName], sqrt(5.0f));

    // Create base bias (frozen)
    if (bias) {
        parameters_[kParamBiasName]
            = std::make_shared<Tensor>(std::vector<int64_t>{out_features}, DataType::kFLOAT32, device_);
        const auto [fan_in, _] = init::CalculateFanInAndFanOut(parameters_[kParamWeightName]);
        const float bound = fan_in > 0 ? 1.0 / sqrt(fan_in) : 0.0;
        init::Uniform(parameters_[kParamBiasName], -bound, bound);
    }

    // Initialize LoRA weights
    InitLoRAWeights();

    // Freeze base weights
    FreezeBaseWeights();
}

LoRALinear::LoRALinear(std::shared_ptr<nn::Module> base_linear, const LoRAConfig &config)
    : CloneableModule(kType), config_(config), bias_(false) {
    if (!base_linear) {
        throw std::invalid_argument("base_linear cannot be null");
    }

    // Get device from base linear
    device_ = base_linear->parameter(nn::Linear::kParamWeightName)->GetDevice();

    // Transfer weight from base linear
    parameters_[kParamWeightName] = base_linear->parameter(nn::Linear::kParamWeightName);

    // Get dimensions from weight shape [out_features, in_features]
    const auto &weight_dims = parameters_[kParamWeightName]->Dims();
    out_features_ = weight_dims[0];
    in_features_ = weight_dims[1];

    // Transfer bias if exists
    if (base_linear->has_parameter(nn::Linear::kParamBiasName)) {
        parameters_[kParamBiasName] = base_linear->parameter(nn::Linear::kParamBiasName);
        bias_ = true;
    }

    // Initialize LoRA weights
    InitLoRAWeights();

    // Freeze base weights
    FreezeBaseWeights();
}

void LoRALinear::InitLoRAWeights() {
    // A matrix: [rank, in_features]
    // Initialize with Kaiming uniform (or normal based on config)
    parameters_[kParamLoraAName]
        = std::make_shared<Tensor>(std::vector<int64_t>{config_.rank, in_features_}, DataType::kFLOAT32, device_)
              ->RequiresGrad();

    if (config_.use_kaiming_a) {
        init::KaimingUniform(parameters_[kParamLoraAName], config_.kaiming_a_param);
    } else {
        init::Normal(parameters_[kParamLoraAName], 0.0f, 0.02f);
    }

    // B matrix: [out_features, rank]
    // Initialize with zeros (ensures LoRA starts as identity transformation)
    parameters_[kParamLoraBName]
        = std::make_shared<Tensor>(std::vector<int64_t>{out_features_, config_.rank}, DataType::kFLOAT32, device_)
              ->RequiresGrad();
    init::Zeros(parameters_[kParamLoraBName]);
}

void LoRALinear::FreezeBaseWeights() {
    // Set requires_grad to false for base weights
    parameters_[kParamWeightName]->set_requires_grad(false);
    if (bias_) {
        parameters_[kParamBiasName]->set_requires_grad(false);
    }
}

std::vector<std::shared_ptr<Tensor>> LoRALinear::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    const auto &input = input_tensors[0];

    // Base linear computation: y = x @ W^T + b
    auto base_output = std::make_shared<autograd::Linear>()->Apply(
        bias_ ? std::vector<std::shared_ptr<Tensor>>{input, parameters_[kParamWeightName], parameters_[kParamBiasName]}
              : std::vector<std::shared_ptr<Tensor>>{input, parameters_[kParamWeightName]})[0];

    if (merged_) {
        // If merged, base weight already contains LoRA contribution
        return {base_output};
    }

    // LoRA computation: delta = (alpha/r) * x @ A^T @ B^T
    // A: [rank, in_features], B: [out_features, rank]
    // x @ A^T: [batch, seq, in_features] @ [in_features, rank] = [batch, seq, rank]
    // (x @ A^T) @ B^T: [batch, seq, rank] @ [rank, out_features] = [batch, seq, out_features]

    // Compute x @ A^T (using Linear function with A as weight, no bias)
    auto hidden = std::make_shared<autograd::Linear>()->Apply({input, parameters_[kParamLoraAName]})[0];

    // Compute hidden @ B^T (using Linear function with B as weight, no bias)
    auto lora_output = std::make_shared<autograd::Linear>()->Apply({hidden, parameters_[kParamLoraBName]})[0];

    // Scale and add: y = base_output + scaling * lora_output
    float scaling = config_.Scaling();
    auto scaled_lora = lora_output->Mul(scaling);

    return {base_output->Add(scaled_lora)};
}

void LoRALinear::MergeWeights() {
    if (merged_) {
        return;
    }

    // Save original weight for potential unmerge
    original_weight_ = std::make_shared<Tensor>(*parameters_[kParamWeightName]);

    // W' = W + (alpha/r) * B @ A
    // W: [out_features, in_features]
    // B: [out_features, rank]
    // A: [rank, in_features]
    // B @ A: [out_features, in_features]

    auto lora_A = parameters_[kParamLoraAName];
    auto lora_B = parameters_[kParamLoraBName];

    // Compute B @ A using matmul
    auto delta = lora_B->Matmul(lora_A); // [out_features, in_features]

    // Scale and add to weight
    float scaling = config_.Scaling();
    auto scaled_delta = delta->Mul(scaling);
    auto new_weight = parameters_[kParamWeightName]->Add(scaled_delta);

    // Update weight data
    parameters_[kParamWeightName]->CopyFrom(new_weight);

    merged_ = true;
}

void LoRALinear::UnmergeWeights() {
    if (!merged_ || !original_weight_) {
        return;
    }

    parameters_[kParamWeightName]->CopyFrom(original_weight_);
    merged_ = false;
}

std::vector<std::shared_ptr<Tensor>> LoRALinear::LoRAParameters() const {
    return {parameters_.at(kParamLoraAName), parameters_.at(kParamLoraBName)};
}

std::vector<std::shared_ptr<Tensor>> LoRALinear::Parameters() const {
    // Only return trainable LoRA parameters
    return LoRAParameters();
}

std::vector<std::shared_ptr<Tensor>> LoRALinear::AllParameters() const {
    std::vector<std::shared_ptr<Tensor>> all_params;
    all_params.push_back(parameters_.at(kParamWeightName));
    if (bias_) {
        all_params.push_back(parameters_.at(kParamBiasName));
    }
    all_params.push_back(parameters_.at(kParamLoraAName));
    all_params.push_back(parameters_.at(kParamLoraBName));
    return all_params;
}

int64_t LoRALinear::in_features() const {
    return in_features_;
}

int64_t LoRALinear::out_features() const {
    return out_features_;
}

int64_t LoRALinear::rank() const {
    return config_.rank;
}

float LoRALinear::scaling() const {
    return config_.Scaling();
}

} // namespace infini_train::nn::lora
