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
    : Linear(in_features, out_features, bias, device ? *device : Device()), config_(config), in_features_(in_features),
      out_features_(out_features) {

    // Initialize LoRA weights
    InitLoRAWeights();

    // Freeze base weights
    FreezeBaseWeights();
}

LoRALinear::LoRALinear(std::shared_ptr<nn::Module> base_linear, const LoRAConfig &config)
    : Linear(base_linear->parameter(kParamWeightName)->Dims()[1], base_linear->parameter(kParamWeightName)->Dims()[0],
             base_linear->has_parameter(kParamBiasName), base_linear->parameter(kParamWeightName)->GetDevice()),
      config_(config), in_features_(base_linear->parameter(kParamWeightName)->Dims()[1]),
      out_features_(base_linear->parameter(kParamWeightName)->Dims()[0]) {
    if (!base_linear) {
        LOG(FATAL) << "base_linear cannot be null";
    }

    // Transfer weight from base linear (overwrite base-created one)
    parameters_[kParamWeightName] = base_linear->parameter(kParamWeightName);

    // Transfer bias if exists
    if (has_bias()) {
        parameters_[kParamBiasName] = base_linear->parameter(kParamBiasName);
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
    if (has_bias()) {
        parameters_[kParamBiasName]->set_requires_grad(false);
    }
}

std::vector<std::shared_ptr<Tensor>> LoRALinear::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK(!(merged_ && parameters_.at(kParamLoraAName)->requires_grad()))
        << "Forward() on merged LoRA with requires_grad=true. Call UnmergeWeights() before training.";

    // Base linear computation: y = x @ W^T + b
    auto base_output = Linear::Forward(input_tensors)[0];

    if (merged_) {
        // If merged, base weight already contains LoRA contribution
        return {base_output};
    }

    const auto &input = input_tensors[0];

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

    // W' = W + (alpha/r) * B @ A
    auto lora_A = parameters_[kParamLoraAName];
    auto lora_B = parameters_[kParamLoraBName];

    auto delta = lora_B->Matmul(lora_A); // [out_features, in_features]
    auto scaled_delta = delta->Mul(config_.Scaling());
    auto new_weight = parameters_[kParamWeightName]->Add(scaled_delta);
    parameters_[kParamWeightName]->CopyFrom(new_weight);

    // Freeze LoRA params to prevent training while merged
    lora_A->set_requires_grad(false);
    lora_B->set_requires_grad(false);

    merged_ = true;
}

void LoRALinear::UnmergeWeights() {
    if (!merged_) {
        return;
    }

    // W = W - (alpha/r) * B @ A
    auto lora_A = parameters_[kParamLoraAName];
    auto lora_B = parameters_[kParamLoraBName];

    auto delta = lora_B->Matmul(lora_A);
    auto scaled_delta = delta->Mul(config_.Scaling());
    auto new_weight = parameters_[kParamWeightName]->Sub(scaled_delta);
    parameters_[kParamWeightName]->CopyFrom(new_weight);

    // Restore LoRA params to trainable
    lora_A->set_requires_grad(true);
    lora_B->set_requires_grad(true);

    merged_ = false;
}

std::vector<std::shared_ptr<Tensor>> LoRALinear::LoRAParameters() const {
    return {parameters_.at(kParamLoraAName), parameters_.at(kParamLoraBName)};
}

int64_t LoRALinear::in_features() const { return in_features_; }

int64_t LoRALinear::out_features() const { return out_features_; }

int64_t LoRALinear::rank() const { return config_.rank; }

float LoRALinear::scaling() const { return config_.Scaling(); }

} // namespace infini_train::nn::lora
