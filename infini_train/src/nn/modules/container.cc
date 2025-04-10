#include "infini_train/include/nn/modules/container.h"

#include <memory>
#include <string>
#include <vector>

#include "infini_train/include/tensor.h"

namespace infini_train::nn {
Sequential::Sequential(std::vector<std::unique_ptr<Module>> &&layers) {
    int idx = 0;
    for (auto &layer : layers) {
        modules_[std::to_string(idx)] = std::move(layer);
        ++idx;
    }
}

std::vector<std::shared_ptr<Tensor>> Sequential::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    auto &x = const_cast<std::vector<std::shared_ptr<Tensor>> &>(input_tensors);
    for (int idx = 0; idx < modules_.size(); ++idx) { x = modules_[std::to_string(idx)]->Forward(x); }
    return x;
}

ModuleDict::ModuleDict(std::unordered_map<std::string, std::unique_ptr<Module>> &&modules) {
    for (auto &[name, layer] : modules) { modules_[name] = std::move(layer); }
}

std::vector<std::shared_ptr<Tensor>> ModuleDict::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    LOG(FATAL) << "Not implemented";
}
} // namespace infini_train::nn
