#pragma once

#include <memory>
#include <string>
#include <vector>

namespace infini_train {
class Tensor;

namespace autograd {
class Function;
class HookHandle;
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

    static void RegisterForFunction(autograd::Function *func, const std::string &name = "",
                                    const Config &config = DefaultConfig());

    // Register hooks for a Module (checks forward inputs/outputs)
    static void RegisterForModule(nn::Module *module, const std::string &name = "",
                                  const Config &config = DefaultConfig());

private:
    static void CheckTensors(const std::string &stage, const std::string &name,
                             const std::vector<std::shared_ptr<Tensor>> &tensors, const Config &config);
};

} // namespace utils
} // namespace infini_train
