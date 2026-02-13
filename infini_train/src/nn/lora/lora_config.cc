#include "infini_train/include/nn/lora/lora_config.h"

#include <sstream>

namespace infini_train::nn::lora {

void LoRAConfig::SetTargetModules(const std::string &targets) {
    target_modules.clear();
    std::stringstream ss(targets);
    std::string module;
    while (std::getline(ss, module, ',')) {
        // Trim whitespace
        module.erase(module.find_last_not_of(" \t\r\n") + 1);
        module.erase(0, module.find_first_not_of(" \t\r\n"));
        if (!module.empty()) {
            target_modules.insert(module);
        }
    }
}

float LoRAConfig::Scaling() const { return alpha / static_cast<float>(rank); }

bool LoRAConfig::ShouldApplyLoRA(const std::string &module_name) const {
    // Check if the module name ends with any of the target module names
    // e.g., "transformer.h.0.attn.c_attn" should match "c_attn"
    for (const auto &target : target_modules) {
        // Check if module_name ends with target
        if (module_name.length() >= target.length()) {
            size_t pos = module_name.length() - target.length();
            if (module_name.substr(pos) == target) {
                // Make sure it's a complete component (preceded by '.' or at start)
                if (pos == 0 || module_name[pos - 1] == '.') {
                    return true;
                }
            }
        }
    }
    return false;
}

} // namespace infini_train::nn::lora
