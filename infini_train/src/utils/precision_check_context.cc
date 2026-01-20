#include "infini_train/include/utils/precision_check_context.h"

namespace infini_train::utils {

PrecisionCheckContext &PrecisionCheckContext::Instance() {
    static thread_local PrecisionCheckContext instance;
    return instance;
}

void PrecisionCheckContext::SetGAS(int gas) { gas_ = gas; }

void PrecisionCheckContext::SetLayer(int layer) { layer_ = layer; }

void PrecisionCheckContext::SetLayerName(const std::string &name) { layer_name_ = name; }

int PrecisionCheckContext::GetGAS() const { return gas_; }

int PrecisionCheckContext::GetLayer() const { return layer_; }

const std::string &PrecisionCheckContext::GetLayerName() const { return layer_name_; }

std::string PrecisionCheckContext::GetKey() const {
    std::string key = "[GAS-" + std::to_string(gas_) + "]";
    key += " [L-" + std::to_string(layer_) + "]";
    if (!layer_name_.empty()) {
        key += " " + layer_name_;
    }
    return key;
}

void PrecisionCheckContext::Reset() {
    gas_ = 0;
    layer_ = 0;
    layer_name_.clear();
}

} // namespace infini_train::utils
