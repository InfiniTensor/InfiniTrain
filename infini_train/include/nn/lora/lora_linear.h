#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/nn/lora/lora_config.h"
#include "infini_train/include/nn/modules/module.h"

namespace infini_train {
class Tensor;
class Device;
} // namespace infini_train

namespace infini_train::nn::lora {

// LoRA wrapper for standard Linear layer
// Implements: y = Wx + b + (alpha/r) * x @ A^T @ B^T
// Where W is frozen, A and B are trainable low-rank matrices
class LoRALinear : public nn::CloneableModule<LoRALinear> {
public:
    static constexpr char kType[] = "LoRALinear";

    // Parameter names
    static constexpr char kParamWeightName[] = "weight"; // Frozen base weight
    static constexpr char kParamBiasName[] = "bias";     // Frozen base bias
    static constexpr char kParamLoraAName[] = "lora_A";  // Trainable A matrix [rank, in_features]
    static constexpr char kParamLoraBName[] = "lora_B";  // Trainable B matrix [out_features, rank]

    // Constructor from scratch
    LoRALinear(int64_t in_features, int64_t out_features, const LoRAConfig &config, bool bias = true,
               const Device *device = nullptr);

    // Constructor wrapping existing Linear module (transfers ownership of parameters)
    LoRALinear(std::shared_ptr<nn::Module> base_linear, const LoRAConfig &config);

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

    // LoRA-specific methods
    void MergeWeights();   // Merge LoRA weights into base: W' = W + (alpha/r) * B @ A
    void UnmergeWeights(); // Restore original base weights
    bool IsMerged() const { return merged_; }

    // Get only LoRA parameters (for optimizer)
    std::vector<std::shared_ptr<Tensor>> LoRAParameters() const;

    // Override Parameters() to return only trainable (LoRA) parameters
    std::vector<std::shared_ptr<Tensor>> Parameters() const override;

    // Get all parameters including frozen base weights (for state dict)
    std::vector<std::shared_ptr<Tensor>> AllParameters() const;

    // Accessors
    int64_t in_features() const;
    int64_t out_features() const;
    int64_t rank() const;
    float scaling() const;

private:
    void InitLoRAWeights();
    void FreezeBaseWeights();

    LoRAConfig config_;
    int64_t in_features_;
    int64_t out_features_;
    bool bias_;
    bool merged_ = false;

    // Store original weight for unmerge
    std::shared_ptr<Tensor> original_weight_;
};

} // namespace infini_train::nn::lora
