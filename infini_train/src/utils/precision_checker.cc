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

        // Copy tensor to CPU if it's on GPU
        std::shared_ptr<Tensor> cpu_tensor;
        if (tensor->GetDevice()->Type() == DeviceType::kCUDA) {
            auto cpu_device = DeviceManager::Instance()->GetDevice(DeviceType::kCPU);
            cpu_tensor = std::make_shared<Tensor>(tensor->To(cpu_device));
        } else {
            cpu_tensor = tensor;
        }

        const float* data = static_cast<const float*>(cpu_tensor->DataPtr());
        size_t size = cpu_tensor->NumElements();

        bool has_nan = false;
        bool has_inf = false;

        for (size_t j = 0; j < size; ++j) {
            float val = data[j];
            if (std::isnan(val)) has_nan = true;
            if (std::isinf(val)) has_inf = true;
        }

        bool has_error = (config.check_nan && has_nan) || (config.check_inf && has_inf);

        if (has_error || config.print_stats) {
            std::cout << "[PrecisionCheck] " << stage << " " << name << " tensor[" << i << "]: [";

            if (has_nan) std::cout << " NaN detected!";
            if (has_inf) std::cout << " Inf detected!";

            if (config.print_stats) {
                constexpr size_t max_print = 10;
                for (size_t j = 0; j < std::min(size, max_print); ++j) {
                    if (j > 0) std::cout << ", ";
                    std::cout << data[j];
                }
                if (size > max_print) std::cout << ", ...";
            }
            std::cout << "]" << std::endl;
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
