#pragma once

#include <memory>
#include <string>
#include <vector>

#include "infini_train/include/nn/lora/lora_config.h"
#include "infini_train/include/nn/modules/module.h"

namespace infini_train::nn::lora {

// LoRAModel: A wrapper that applies LoRA to any base model
// This follows the PEFT design pattern where LoRA is applied as a wrapper
// rather than modifying the base model code directly.
//
// Usage:
//   auto base_model = std::make_shared<GPT2>(config);
//   LoRAConfig lora_config{8, 16.0f};
//   lora_config.SetTargetModules("c_attn,c_proj");  // or include mlp layers
//   auto lora_model = std::make_shared<LoRAModel>(base_model, lora_config);
//
//   // Training: only LoRA parameters are trainable
//   auto optimizer = SGD(lora_model->TrainableParameters(), lr);
//
//   // Save only LoRA weights
//   lora_model->SaveLoRA("lora_weights.bin");
//
//   // Load LoRA weights
//   lora_model->LoadLoRA("lora_weights.bin");
//
//   // Merge for inference (optional)
//   lora_model->Merge();
//
class LoRAModel : public Module {
public:
    static constexpr char kType[] = "LoRAModel";

    // Constructor: wraps a base model with LoRA
    // Uses NamedModules() to automatically traverse the model hierarchy
    // Parameters:
    //   - base_model: The original model (GPT2, LLaMA3, etc.)
    //   - config: LoRA configuration (rank, alpha, target_modules)
    LoRAModel(std::shared_ptr<Module> base_model, const LoRAConfig &config);

    // Forward pass (delegates to base model)
    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &inputs) override;

    // Get only trainable (LoRA) parameters for optimizer
    std::vector<std::shared_ptr<Tensor>> TrainableParameters() const;

    // Get all parameters (for state dict)
    std::vector<std::shared_ptr<Tensor>> Parameters() const override;

    // LoRA weight management
    void SaveLoRA(const std::string &filepath) const;
    void LoadLoRA(const std::string &filepath);

    // Merge/unmerge LoRA weights into base model
    void Merge();
    void Unmerge();
    bool IsMerged() const;

    // Print summary
    void PrintSummary() const;

    // Access base model
    std::shared_ptr<Module> base_model() const;

    // Get LoRA config
    const LoRAConfig &config() const;

private:
    std::shared_ptr<Module> base_model_;
    LoRAConfig config_;
    bool merged_ = false;
};

// Factory function for creating LoRA-enabled models
// This is the recommended way to create LoRA models
template <typename ModelType, typename ConfigType>
std::shared_ptr<LoRAModel> CreateLoRAModel(const ConfigType &model_config, const LoRAConfig &lora_config) {
    auto base_model = std::make_shared<ModelType>(model_config);
    return std::make_shared<LoRAModel>(base_model, lora_config);
}

} // namespace infini_train::nn::lora
