#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "infini_train/include/nn/lora/lora_config.h"

namespace infini_train {
class Tensor;
}

namespace infini_train::nn {
class Module;
}

namespace infini_train::nn::lora {

/**
 * Apply LoRA to a model (PEFT-style injection).
 *
 * - Replaces target modules with LoRA modules (in-place).
 * - Freezes all non-LoRA parameters.
 * - Only LoRA parameters remain trainable.
 *
 * The root module may be replaced if it matches the target.
 */
std::shared_ptr<Module> GetLoRAModel(std::shared_ptr<Module> model, const LoRAConfig &config);

/**
 * Inject LoRA modules into all matching submodules.
 *
 * Performs structural replacement only (no freezing).
 * Root module may be replaced.
 */
std::shared_ptr<Module> InjectLoRALayers(std::shared_ptr<Module> model, const LoRAConfig &config);

/**
 * Replace a submodule by its full path (e.g. "a.b.0.c").
 */
void ReplaceModuleByPath(std::shared_ptr<Module> model, const std::string &path, std::shared_ptr<Module> new_module);

/**
 * Freeze all parameters, then re-enable LoRA parameters.
 * After this call, only LoRA params are trainable.
 */
void FreezeBaseModel(std::shared_ptr<Module> model);

/**
 * Set requires_grad = true for all parameters.
 */
void UnfreezeModel(std::shared_ptr<Module> model);

/**
 * Return all LoRA parameters.
 */
std::vector<std::shared_ptr<Tensor>> GetLoRAParameters(const std::shared_ptr<Module> &model);

/**
 * Return all non-LoRA parameters.
 */
std::vector<std::shared_ptr<Tensor>> GetBaseParameters(const std::shared_ptr<Module> &model);

/**
 * Merge LoRA into base weights:
 *   W = W + (B @ A) * scale
 */
void MergeLoRAWeights(std::shared_ptr<Module> model);

/**
 * Undo previously merged LoRA weights.
 */
void UnmergeLoRAWeights(std::shared_ptr<Module> model);

/**
 * Merge LoRA weights and remove LoRA modules, returning a clean base model.
 * Similar to PEFT's merge_and_unload().
 *
 * For each LoRA module:
 * 1. Merge weights: W += (alpha/r) * B @ A
 * 2. Replace LoRA module with a base module sharing the merged weight/bias
 *
 * After this call, the model contains no LoRA parameters.
 * Root module may be replaced (same pattern as InjectLoRALayers).
 */
std::shared_ptr<Module> MergeAndUnload(std::shared_ptr<Module> model);

/**
 * Return a state dict containing only LoRA parameters.
 */
std::unordered_map<std::string, std::shared_ptr<Tensor>> LoRAStateDict(const std::shared_ptr<Module> &model);

/**
 * Load LoRA parameters from a state dict.
 */
void LoadLoRAStateDict(std::shared_ptr<Module> model,
                       const std::unordered_map<std::string, std::shared_ptr<Tensor>> &state_dict);

/**
 * Save only LoRA parameters to file.
 */
void SaveLoRAWeights(const std::shared_ptr<Module> &model, const std::string &filepath);

/**
 * Load LoRA parameters from file.
 */
void LoadLoRAWeights(std::shared_ptr<Module> model, const std::string &filepath);

/**
 * Count parameters with requires_grad == true.
 */
int64_t CountTrainableParameters(const std::shared_ptr<Module> &model);

/**
 * Count total parameters.
 */
int64_t CountTotalParameters(const std::shared_ptr<Module> &model);

/**
 * Print total/trainable/frozen parameter summary.
 */
void PrintLoRASummary(const std::shared_ptr<Module> &model, int global_rank = -1);

/**
 * Parse comma-separated target modules string.
 *
 * Example: "c_attn,c_proj" -> {"c_attn", "c_proj"}
 *
 * Returns:
 *   An unordered_set of target module names (whitespace trimmed).
 */
std::unordered_set<std::string> ParseLoRATargetModules(const std::string &targets);

} // namespace infini_train::nn::lora
