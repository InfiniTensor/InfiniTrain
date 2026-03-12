#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/nn/lora/lora_config.h"
#include "infini_train/include/nn/modules/linear.h"

// Forward declarations for test functions (required for friend declarations)
void test_lora_linear_init();
void test_lora_linear_forward();
void test_lora_linear_merge();
void test_lora_utils();

namespace infini_train {
class Tensor;
class Device;
} // namespace infini_train

namespace infini_train::nn::lora {

// LoRA wrapper for standard Linear layer
// Implements: y = Wx + b + (alpha/r) * x @ A^T @ B^T
// Where W is frozen, A and B are trainable low-rank matrices
class LoRALinear : public nn::Linear {
public:
    static constexpr char kType[] = "LoRALinear";

    // Parameter names for LoRA-specific parameters
    static constexpr char kParamLoraAName[] = "lora_A"; // Trainable A matrix [rank, in_features]
    static constexpr char kParamLoraBName[] = "lora_B"; // Trainable B matrix [out_features, rank]

    // Constructor wrapping existing Linear module (transfers ownership of parameters)
    LoRALinear(std::shared_ptr<nn::Module> base_linear, const LoRAConfig &config);

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

    // LoRA-specific methods
    void MergeWeights();   // Merge LoRA weights into base: W' = W + (alpha/r) * B @ A
    void UnmergeWeights(); // Restore original base weights
    bool IsMerged() const { return merged_; }

    // Get only LoRA parameters (for optimizer)
    std::vector<std::shared_ptr<Tensor>> LoRAParameters() const;

    // Accessors
    int64_t in_features() const;
    int64_t out_features() const;
    int64_t rank() const;
    float scaling() const;

private:
    // Test-only: Create LoRA module from scratch (normal usage goes through InjectLoRALayers)
    LoRALinear(int64_t in_features, int64_t out_features, const LoRAConfig &config, bool bias, const Device *device);

    // Test access
    friend void ::test_lora_linear_init();
    friend void ::test_lora_linear_forward();
    friend void ::test_lora_linear_merge();
    friend void ::test_lora_utils();

    void InitLoRAWeights();
    void FreezeBaseWeights();

    LoRAConfig config_;
    int64_t in_features_ = 0;
    int64_t out_features_ = 0;
    bool merged_ = false;
};

} // namespace infini_train::nn::lora
