#pragma once

#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_set>

namespace infini_train::nn::lora {

// LoRA (Low-Rank Adaptation) configuration
struct LoRAConfig {
    // Core LoRA parameters
    int64_t rank = 8;     // Low-rank dimension (r)
    float alpha = 16.0f;  // Scaling factor (alpha)
    float dropout = 0.0f; // Dropout probability (optional, not implemented yet)

    // Target modules specification (default: attention layers only)
    std::unordered_set<std::string> target_modules = {"c_attn", "c_proj"};

    // Initialization parameters
    bool use_kaiming_a = true;           // Use Kaiming init for A matrix
    float kaiming_a_param = sqrtf(5.0f); // Parameter 'a' for Kaiming init

    // Default constructor (uses default target_modules = {"c_attn", "c_proj"})
    LoRAConfig() = default;

    // Constructor with all parameters including target modules
    LoRAConfig(int64_t r, float a, float d, const std::unordered_set<std::string> &targets)
        : rank(r), alpha(a), dropout(d), target_modules(targets) {}

    // Compute scaling factor: output = base_output + scaling * lora_output
    float Scaling() const;

    // Check if a module name should have LoRA applied
    // Matches if the module name ends with any of the target module names
    bool ShouldApplyLoRA(const std::string &module_name) const;
};

} // namespace infini_train::nn::lora
