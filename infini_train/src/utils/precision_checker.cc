#include "infini_train/include/utils/precision_checker.h"

#include <cmath>
#include <iostream>
#include <limits>

#include "infini_train/include/autograd/function.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/tensor.h"

namespace infini_train::utils {

void PrecisionChecker::CheckTensors(const std::string& stage, const std::string& name,
                                   const std::vector<std::shared_ptr<Tensor>>& tensors,
                                   const Config& config) {
    for (size_t i = 0; i < tensors.size(); ++i) {
        if (!tensors[i]) continue;

        auto& tensor = tensors[i];
        const float* data = static_cast<const float*>(tensor->DataPtr());
        size_t size = tensor->NumElements();

        bool has_nan = false;
        bool has_inf = false;
        float min_val = std::numeric_limits<float>::max();
        float max_val = std::numeric_limits<float>::lowest();
        double sum = 0.0;

        for (size_t j = 0; j < size; ++j) {
            float val = data[j];
            if (std::isnan(val)) has_nan = true;
            if (std::isinf(val)) has_inf = true;
            if (!std::isnan(val) && !std::isinf(val)) {
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
                sum += val;
            }
        }

        bool has_error = (config.check_nan && has_nan) || (config.check_inf && has_inf);

        if (has_error || config.print_stats) {
            std::cout << "[PrecisionCheck] " << stage << " " << name
                     << " tensor[" << i << "]";

            if (has_nan) std::cout << " NaN detected!";
            if (has_inf) std::cout << " Inf detected!";

            if (config.print_stats) {
                std::cout << " min=" << min_val
                         << " max=" << max_val
                         << " mean=" << (sum / size);
            }
            std::cout << std::endl;
        }

        if (has_error && config.abort_on_error) {
            std::cerr << "Precision check failed, aborting!" << std::endl;
            std::abort();
        }
    }
}

void PrecisionChecker::RegisterForFunction(autograd::Function* func, const std::string& name,
                                          const Config& config) {
    std::string func_name = name.empty() ? "Function" : name;

    func->RegisterForwardPreHook([func_name, config](autograd::Function*,
                                                      const std::vector<std::shared_ptr<Tensor>>& inputs) {
        CheckTensors("Forward Input", func_name, inputs, config);
    });

    func->RegisterForwardPostHook([func_name, config](autograd::Function*,
                                                       const std::vector<std::shared_ptr<Tensor>>&,
                                                       const std::vector<std::shared_ptr<Tensor>>& outputs) {
        CheckTensors("Forward Output", func_name, outputs, config);
    });

    func->RegisterBackwardPreHook([func_name, config](autograd::Function*,
                                                       const std::vector<std::shared_ptr<Tensor>>& grad_outputs) {
        CheckTensors("Backward Input", func_name, grad_outputs, config);
    });

    func->RegisterBackwardPostHook([func_name, config](autograd::Function*,
                                                        const std::vector<std::shared_ptr<Tensor>>& grad_inputs,
                                                        const std::vector<std::shared_ptr<Tensor>>&) {
        CheckTensors("Backward Output", func_name, grad_inputs, config);
    });
}

void PrecisionChecker::RegisterForAllFunctions(const std::vector<std::shared_ptr<autograd::Function>>& functions,
                                              const Config& config) {
    for (size_t i = 0; i < functions.size(); ++i) {
        RegisterForFunction(functions[i].get(), "Function_" + std::to_string(i), config);
    }
}

void PrecisionChecker::RegisterForModule(nn::Module* module, const std::string& name,
                                        const Config& config) {
    std::string module_name = name.empty() ? module->type() : name;

    // module->RegisterForwardPreHook([module_name, config](nn::Module*,
    //                                                       const std::vector<std::shared_ptr<Tensor>>& inputs) {
    //     CheckTensors("Module Forward Input", module_name, inputs, config);
    // });

    module->RegisterForwardPostHook([module_name, config](nn::Module*,
                                                           const std::vector<std::shared_ptr<Tensor>>&,
                                                           const std::vector<std::shared_ptr<Tensor>>& outputs) {
        CheckTensors("Module Forward Output", module_name, outputs, config);
    });

    // module->RegisterBackwardPreHook([module_name, config](nn::Module*,
    //                                                        const std::vector<std::shared_ptr<Tensor>>& grad_outputs) {
    //     CheckTensors("Module Backward Input", module_name, grad_outputs, config);
    // });

    module->RegisterBackwardPostHook([module_name, config](nn::Module*,
                                                            const std::vector<std::shared_ptr<Tensor>>& grad_inputs,
                                                            const std::vector<std::shared_ptr<Tensor>>&) {
        CheckTensors("Module Backward Output", module_name, grad_inputs, config);
    });
}

} // namespace infini_train::utils
