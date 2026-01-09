#include "infini_train/include/utils/precision_checker.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>

#include "infini_train/include/autograd/function.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"

namespace infini_train::utils {

namespace {
std::ofstream& GetLogStream() {
    static std::ofstream log_file;
    static std::mutex init_mutex;
    static bool initialized = false;

    if (!initialized) {
        std::lock_guard<std::mutex> lock(init_mutex);
        if (!initialized) {
            int rank = nn::parallel::global::GlobalEnv::Instance().global_proc_rank();
            std::string filename = "precision_check_rank_" + std::to_string(rank) + ".log";
            log_file.open(filename, std::ios::out | std::ios::trunc);
            initialized = true;
        }
    }
    return log_file;
}

bool ShouldPrint() {
    if (nn::parallel::global::GlobalEnv::Instance().GetPrecisionCheckAllRanks()) {
        return true;
    }
    return nn::parallel::global::GlobalEnv::Instance().global_proc_rank() == 0;
}

std::string GetTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    std::tm tm;
    localtime_r(&time_t, &tm);

    std::ostringstream oss;
    oss << std::setfill('0')
        << std::setw(2) << (tm.tm_mon + 1)
        << std::setw(2) << tm.tm_mday << ' '
        << std::setw(2) << tm.tm_hour << ':'
        << std::setw(2) << tm.tm_min << ':'
        << std::setw(2) << tm.tm_sec << '.'
        << std::setw(3) << ms.count();
    return oss.str();
}
} // namespace

void PrecisionChecker::CheckTensors(const std::string& stage, const std::string& name,
                                   const std::vector<std::shared_ptr<Tensor>>& tensors,
                                   const Config& config) {
    if (!ShouldPrint()) {
        return;
    }

    int rank = nn::parallel::global::GlobalEnv::Instance().global_proc_rank();

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
            auto& log_stream = GetLogStream();
            std::string level = has_error ? "E" : "I";

            log_stream << level << GetTimestamp() << " [Rank " << rank << "][PrecisionCheck] "
                      << stage << " " << name << " tensor[" << i << "]: [";

            if (has_nan) log_stream << " NaN detected!";
            if (has_inf) log_stream << " Inf detected!";

            if (config.print_stats) {
                constexpr size_t max_print = 10;
                for (size_t j = 0; j < std::min(size, max_print); ++j) {
                    if (j > 0) log_stream << ", ";
                    log_stream << data[j];
                }
                if (size > max_print) log_stream << ", ...";
            }
            log_stream << "]" << std::endl;
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
