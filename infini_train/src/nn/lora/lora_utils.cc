#include "infini_train/include/nn/lora/lora_utils.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/device.h"
#include "infini_train/include/nn/lora/lora_linear.h"
#include "infini_train/include/nn/lora/lora_model.h"
#include "infini_train/include/nn/lora/lora_parallel_linear.h"
#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/tensor_parallel.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::lora {

std::shared_ptr<LoRAModel> GetLoRAModel(std::shared_ptr<Module> model, const LoRAConfig &config) {
    // PEFT-style: Create LoRAModel wrapper which handles everything automatically
    // Uses NamedModules() to traverse the entire model hierarchy
    auto lora_model = std::make_shared<LoRAModel>(model, config);
    LOG(INFO) << "GetLoRAModel: Created LoRA model with rank=" << config.rank << ", alpha=" << config.alpha;
    return lora_model;
}

void ReplaceModuleByPath(std::shared_ptr<Module> model, const std::string &path, std::shared_ptr<Module> new_module) {
    // Parse the path (e.g., "transformer.h.0.attn.c_attn" -> ["transformer", "h", "0", "attn", "c_attn"])
    std::vector<std::string> parts;
    std::string remaining = path;
    size_t pos = 0;
    while ((pos = remaining.find('.')) != std::string::npos) {
        parts.push_back(remaining.substr(0, pos));
        remaining = remaining.substr(pos + 1);
    }
    parts.push_back(remaining);

    // Navigate to parent module
    std::shared_ptr<Module> current = model;
    for (size_t i = 0; i < parts.size() - 1; ++i) {
        current = current->mutable_module(parts[i]);
        if (!current) {
            LOG(ERROR) << "ReplaceModuleByPath: Failed to find path: " << path;
            return;
        }
    }

    // Replace the module
    const std::string &module_name = parts.back();
    current->replace_module(module_name, new_module);
}

void InjectLoRALayers(std::shared_ptr<Module> model, const LoRAConfig &config) {
    // Use NamedModules() to automatically traverse the entire model hierarchy
    auto named_modules = model->NamedModules();

    int lora_layers_applied = 0;

    for (const auto &[name, module] : named_modules) {
        if (name.empty()) continue; // skip root module

        // Check if this module should have LoRA applied
        if (!config.ShouldApplyLoRA(name)) continue;

        // Get module type and wrap if it's Linear/ColumnParallelLinear/RowParallelLinear
        auto type = module->type();

        if (type == Linear::kType) {
            auto lora_module = std::make_shared<LoRALinear>(module, config);
            ReplaceModuleByPath(model, name, lora_module);
            lora_layers_applied++;
        } else if (type == parallel::ColumnParallelLinear::kType) {
            auto lora_module = std::make_shared<LoRAColumnParallelLinear>(module, config);
            ReplaceModuleByPath(model, name, lora_module);
            lora_layers_applied++;
        } else if (type == parallel::RowParallelLinear::kType) {
            auto lora_module = std::make_shared<LoRARowParallelLinear>(module, config);
            ReplaceModuleByPath(model, name, lora_module);
            lora_layers_applied++;
        }
    }

    LOG(INFO) << "InjectLoRALayers: Applied LoRA to " << lora_layers_applied << " layers "
              << "(rank=" << config.rank << ", alpha=" << config.alpha << ")";
}

void FreezeBaseModel(std::shared_ptr<Module> model) {
    model->Apply([](Module *m) {
        for (auto &[name, param] : m->StateDict()) {
            // Skip LoRA parameters
            if (name.find("lora_A") != std::string::npos || name.find("lora_B") != std::string::npos) {
                continue;
            }
            param->set_requires_grad(false);
        }
    });
}

void UnfreezeModel(std::shared_ptr<Module> model) {
    model->Apply([](Module *m) {
        for (auto &[name, param] : m->StateDict()) {
            param->set_requires_grad(true);
        }
    });
}

std::vector<std::shared_ptr<Tensor>> GetLoRAParameters(const std::shared_ptr<Module> &model) {
    std::vector<std::shared_ptr<Tensor>> lora_params;

    model->Apply([&lora_params](Module *m) {
        // Check if this is a LoRA module
        if (m->type() == LoRALinear::kType) {
            auto lora_module = dynamic_cast<LoRALinear *>(m);
            if (lora_module) {
                auto params = lora_module->LoRAParameters();
                lora_params.insert(lora_params.end(), params.begin(), params.end());
            }
        } else if (m->type() == LoRAColumnParallelLinear::kType) {
            auto lora_module = dynamic_cast<LoRAColumnParallelLinear *>(m);
            if (lora_module) {
                auto params = lora_module->LoRAParameters();
                lora_params.insert(lora_params.end(), params.begin(), params.end());
            }
        } else if (m->type() == LoRARowParallelLinear::kType) {
            auto lora_module = dynamic_cast<LoRARowParallelLinear *>(m);
            if (lora_module) {
                auto params = lora_module->LoRAParameters();
                lora_params.insert(lora_params.end(), params.begin(), params.end());
            }
        }
    });

    return lora_params;
}

std::vector<std::shared_ptr<Tensor>> GetBaseParameters(const std::shared_ptr<Module> &model) {
    std::vector<std::shared_ptr<Tensor>> base_params;

    for (auto &[name, param] : model->StateDict()) {
        // Skip LoRA parameters
        if (name.find("lora_A") != std::string::npos || name.find("lora_B") != std::string::npos) {
            continue;
        }
        base_params.push_back(param);
    }

    return base_params;
}

void MergeLoRAWeights(std::shared_ptr<Module> model) {
    model->Apply([](Module *m) {
        if (m->type() == LoRALinear::kType) {
            dynamic_cast<LoRALinear *>(m)->MergeWeights();
        } else if (m->type() == LoRAColumnParallelLinear::kType) {
            dynamic_cast<LoRAColumnParallelLinear *>(m)->MergeWeights();
        } else if (m->type() == LoRARowParallelLinear::kType) {
            dynamic_cast<LoRARowParallelLinear *>(m)->MergeWeights();
        }
    });
}

void UnmergeLoRAWeights(std::shared_ptr<Module> model) {
    model->Apply([](Module *m) {
        if (m->type() == LoRALinear::kType) {
            dynamic_cast<LoRALinear *>(m)->UnmergeWeights();
        } else if (m->type() == LoRAColumnParallelLinear::kType) {
            dynamic_cast<LoRAColumnParallelLinear *>(m)->UnmergeWeights();
        } else if (m->type() == LoRARowParallelLinear::kType) {
            dynamic_cast<LoRARowParallelLinear *>(m)->UnmergeWeights();
        }
    });
}

std::unordered_map<std::string, std::shared_ptr<Tensor>> LoRAStateDict(const std::shared_ptr<Module> &model) {
    std::unordered_map<std::string, std::shared_ptr<Tensor>> lora_state_dict;

    for (auto &[name, param] : model->StateDict()) {
        // Only include LoRA parameters
        if (name.find("lora_A") != std::string::npos || name.find("lora_B") != std::string::npos) {
            lora_state_dict[name] = param;
        }
    }

    return lora_state_dict;
}

void LoadLoRAStateDict(std::shared_ptr<Module> model,
                       const std::unordered_map<std::string, std::shared_ptr<Tensor>> &state_dict) {
    auto model_state_dict = model->StateDict();

    for (auto &[name, param] : state_dict) {
        if (model_state_dict.find(name) != model_state_dict.end()) {
            model_state_dict[name]->CopyFrom(param);
        } else {
            LOG(WARNING) << "LoRA parameter not found in model: " << name;
        }
    }
}

void SaveLoRAWeights(const std::shared_ptr<Module> &model, const std::string &filepath) {
    auto lora_state_dict = LoRAStateDict(model);

    std::ofstream file(filepath, std::ios::binary);
    CHECK(file.is_open()) << "Failed to open file for writing: " << filepath;

    // Write magic number
    uint32_t magic = 0x4C4F5241; // "LORA"
    file.write(reinterpret_cast<const char *>(&magic), sizeof(magic));

    // Write version
    uint32_t version = 1;
    file.write(reinterpret_cast<const char *>(&version), sizeof(version));

    // Write number of tensors
    uint32_t num_tensors = static_cast<uint32_t>(lora_state_dict.size());
    file.write(reinterpret_cast<const char *>(&num_tensors), sizeof(num_tensors));

    // Write each tensor
    for (const auto &[name, tensor] : lora_state_dict) {
        // Write name length and name
        uint32_t name_len = static_cast<uint32_t>(name.length());
        file.write(reinterpret_cast<const char *>(&name_len), sizeof(name_len));
        file.write(name.c_str(), name_len);

        // Write tensor dimensions
        const auto &dims = tensor->Dims();
        uint32_t num_dims = static_cast<uint32_t>(dims.size());
        file.write(reinterpret_cast<const char *>(&num_dims), sizeof(num_dims));
        for (auto dim : dims) {
            int64_t d = dim;
            file.write(reinterpret_cast<const char *>(&d), sizeof(d));
        }

        // Write tensor data (copy to CPU first if needed)
        int64_t num_elements = tensor->NumElements();
        Tensor cpu_tensor = tensor->To(Device(Device::DeviceType::kCPU, 0));
        file.write(reinterpret_cast<const char *>(cpu_tensor.DataPtr()), num_elements * sizeof(float));
    }

    file.close();
    LOG(INFO) << "Saved LoRA weights to " << filepath << " (" << num_tensors << " tensors)";
}

void LoadLoRAWeights(std::shared_ptr<Module> model, const std::string &filepath) {
    std::ifstream file(filepath, std::ios::binary);
    CHECK(file.is_open()) << "Failed to open file for reading: " << filepath;

    // Read and verify magic number
    uint32_t magic;
    file.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    CHECK_EQ(magic, 0x4C4F5241) << "Invalid LoRA file format";

    // Read version
    uint32_t version;
    file.read(reinterpret_cast<char *>(&version), sizeof(version));
    CHECK_EQ(version, 1) << "Unsupported LoRA file version: " << version;

    // Read number of tensors
    uint32_t num_tensors;
    file.read(reinterpret_cast<char *>(&num_tensors), sizeof(num_tensors));

    auto model_state_dict = model->StateDict();

    // Read each tensor
    for (uint32_t i = 0; i < num_tensors; ++i) {
        // Read name
        uint32_t name_len;
        file.read(reinterpret_cast<char *>(&name_len), sizeof(name_len));
        std::string name(name_len, '\0');
        file.read(&name[0], name_len);

        // Read dimensions
        uint32_t num_dims;
        file.read(reinterpret_cast<char *>(&num_dims), sizeof(num_dims));
        std::vector<int64_t> dims(num_dims);
        for (uint32_t j = 0; j < num_dims; ++j) {
            file.read(reinterpret_cast<char *>(&dims[j]), sizeof(int64_t));
        }

        // Calculate number of elements
        int64_t num_elements = 1;
        for (auto dim : dims) {
            num_elements *= dim;
        }

        // Read tensor data into a temporary CPU tensor
        auto cpu_tensor = std::make_shared<Tensor>(dims, DataType::kFLOAT32,
                                                    Device(Device::DeviceType::kCPU, 0));
        file.read(reinterpret_cast<char *>(cpu_tensor->DataPtr()), num_elements * sizeof(float));

        // Load into model
        auto it = model_state_dict.find(name);
        if (it != model_state_dict.end()) {
            it->second->CopyFrom(cpu_tensor);
        } else {
            LOG(WARNING) << "LoRA parameter not found in model: " << name;
        }
    }

    file.close();
    LOG(INFO) << "Loaded LoRA weights from " << filepath << " (" << num_tensors << " tensors)";
}

int64_t CountTrainableParameters(const std::shared_ptr<Module> &model) {
    int64_t count = 0;
    for (auto &param : model->Parameters()) {
        if (param->requires_grad()) {
            count += param->NumElements();
        }
    }
    return count;
}

int64_t CountTotalParameters(const std::shared_ptr<Module> &model) {
    int64_t count = 0;
    for (auto &[name, param] : model->StateDict()) {
        count += param->NumElements();
    }
    return count;
}

void PrintLoRASummary(const std::shared_ptr<Module> &model) {
    int64_t trainable = CountTrainableParameters(model);
    int64_t total = CountTotalParameters(model);
    int64_t frozen = total - trainable;

    double trainable_pct = 100.0 * trainable / total;

    std::cout << "========== LoRA Model Summary ==========" << std::endl;
    std::cout << "Total parameters:     " << total << std::endl;
    std::cout << "Trainable parameters: " << trainable << " (" << trainable_pct << "%)" << std::endl;
    std::cout << "Frozen parameters:    " << frozen << std::endl;
    std::cout << "=========================================" << std::endl;
}

} // namespace infini_train::nn::lora
