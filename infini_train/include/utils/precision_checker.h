#pragma once

#include <memory>
#include <string>
#include <vector>

#include "infini_train/include/utils/precision_check_config.h"

namespace infini_train {
class Tensor;
class HookHandle;

namespace autograd {
class Function;
} // namespace autograd

namespace nn {
class Module;
} // namespace nn

namespace utils {

class PrecisionChecker {
public:
    struct Config {
        bool check_nan = true;
        bool check_inf = true;
        bool print_stats = true;
        bool abort_on_error = false;
    };

    static const Config &DefaultConfig() {
        static Config default_config;
        return default_config;
    }

    // Initialize global module-level precision checking
    // Called automatically by PrecisionCheckEnv::Init when level >= MODULE
    static void Init(const PrecisionCheckConfig &global_config, const Config &config = DefaultConfig());

    // Build name map from root_model without registering hooks
    // Called by PrecisionCheckEnv::RegisterWithRootModel
    static void BuildNameMap(nn::Module *root_model);

    static void RegisterForFunction(autograd::Function *func, const std::string &name = "",
                                    const Config &config = DefaultConfig());

    // Register hooks for a Module (checks forward inputs/outputs)
    static void RegisterForModule(nn::Module *module, const std::string &name = "",
                                  const Config &config = DefaultConfig());

    // Reset tensor counters (call at start of each iteration for file overwrite)
    static void ResetCounters();

private:
    static void CheckTensors(const std::string &stage, const std::string &name,
                             const std::vector<std::shared_ptr<Tensor>> &tensors, const Config &config);
};

} // namespace utils
} // namespace infini_train
