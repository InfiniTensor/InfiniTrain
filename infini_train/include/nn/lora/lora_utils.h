#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "infini_train/include/nn/lora/lora_config.h"

namespace infini_train {
class Tensor;
}

namespace infini_train::nn {
class Module;
}

namespace infini_train::nn::lora {

// Forward declaration
class LoRAModel;

// PEFT-style get_peft_model equivalent (Runtime Wrapper)
// Creates a LoRA-wrapped model with automatic module detection using NamedModules
// Parameters:
//   - model: The model to wrap
//   - config: LoRA configuration (rank, alpha, target_modules)
// Returns: The LoRA-wrapped model as shared_ptr
std::shared_ptr<LoRAModel> GetLoRAModel(std::shared_ptr<Module> model, const LoRAConfig &config);

// Internal transform: inject LoRA layers into all matching modules
// Uses NamedModules() to automatically traverse the entire model hierarchy
// Parameters:
//   - model: The model to inject LoRA into
//   - config: LoRA configuration (rank, alpha, target_modules)
void InjectLoRALayers(std::shared_ptr<Module> model, const LoRAConfig &config);

// Replace a module at the given path with a new module
// Parameters:
//   - model: Root model containing the module
//   - path: Full path to the module (e.g., "transformer.h.0.attn.c_attn")
//   - new_module: The new module to replace with
void ReplaceModuleByPath(std::shared_ptr<Module> model, const std::string &path, std::shared_ptr<Module> new_module);

// Freeze all base model parameters (set requires_grad = false)
void FreezeBaseModel(std::shared_ptr<Module> model);

// Unfreeze all parameters (set requires_grad = true)
void UnfreezeModel(std::shared_ptr<Module> model);

// Get only LoRA parameters from a model (for optimizer)
// Returns parameters from LoRALinear, LoRAColumnParallelLinear, LoRARowParallelLinear modules
std::vector<std::shared_ptr<Tensor>> GetLoRAParameters(const std::shared_ptr<Module> &model);

// Get only base (frozen) parameters
std::vector<std::shared_ptr<Tensor>> GetBaseParameters(const std::shared_ptr<Module> &model);

// Merge all LoRA weights in the model
void MergeLoRAWeights(std::shared_ptr<Module> model);

// Unmerge all LoRA weights in the model
void UnmergeLoRAWeights(std::shared_ptr<Module> model);

// Save only LoRA weights to file
void SaveLoRAWeights(const std::shared_ptr<Module> &model, const std::string &filepath);

// Load LoRA weights from file
void LoadLoRAWeights(std::shared_ptr<Module> model, const std::string &filepath);

// Get LoRA state dict (only LoRA parameters with their names)
std::unordered_map<std::string, std::shared_ptr<Tensor>> LoRAStateDict(const std::shared_ptr<Module> &model);

// Load LoRA state dict
void LoadLoRAStateDict(std::shared_ptr<Module> model,
                       const std::unordered_map<std::string, std::shared_ptr<Tensor>> &state_dict);

// Print LoRA model summary (trainable vs frozen parameters)
void PrintLoRASummary(const std::shared_ptr<Module> &model);

// Count trainable parameters
int64_t CountTrainableParameters(const std::shared_ptr<Module> &model);

// Count total parameters
int64_t CountTotalParameters(const std::shared_ptr<Module> &model);

} // namespace infini_train::nn::lora
