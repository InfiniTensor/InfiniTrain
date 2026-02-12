#include "infini_train/include/nn/lora/lora_model.h"

#include <iostream>

#include "glog/logging.h"

#include "infini_train/include/nn/lora/lora_utils.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::lora {

LoRAModel::LoRAModel(std::shared_ptr<Module> base_model, const LoRAConfig &config)
    : base_model_(base_model), config_(config) {
    // Inject LoRA layers into the base model using NamedModules
    InjectLoRALayers(base_model_, config_);

    // Freeze base model parameters
    FreezeBaseModel(base_model_);

    LOG(INFO) << "LoRAModel created with rank=" << config_.rank << ", alpha=" << config_.alpha;
}

std::vector<std::shared_ptr<Tensor>> LoRAModel::Forward(const std::vector<std::shared_ptr<Tensor>> &inputs) {
    return (*base_model_)(inputs);
}

std::vector<std::shared_ptr<Tensor>> LoRAModel::TrainableParameters() const {
    return GetLoRAParameters(base_model_);
}

std::vector<std::shared_ptr<Tensor>> LoRAModel::Parameters() const {
    return base_model_->Parameters();
}

void LoRAModel::SaveLoRA(const std::string &filepath) const {
    SaveLoRAWeights(base_model_, filepath);
}

void LoRAModel::LoadLoRA(const std::string &filepath) {
    LoadLoRAWeights(base_model_, filepath);
}

void LoRAModel::Merge() {
    if (!merged_) {
        MergeLoRAWeights(base_model_);
        merged_ = true;
    }
}

void LoRAModel::Unmerge() {
    if (merged_) {
        UnmergeLoRAWeights(base_model_);
        merged_ = false;
    }
}

void LoRAModel::PrintSummary() const {
    PrintLoRASummary(base_model_);
}

bool LoRAModel::IsMerged() const {
    return merged_;
}

std::shared_ptr<Module> LoRAModel::base_model() const {
    return base_model_;
}

const LoRAConfig &LoRAModel::config() const {
    return config_;
}

} // namespace infini_train::nn::lora
