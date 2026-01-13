#pragma once

#include <string>

namespace infini_train {
namespace utils {

// Context for tracking precision check information (GAS step, layer number, etc.)
// Thread-local to ensure thread safety in multi-threaded training
class PrecisionCheckContext {
public:
    static PrecisionCheckContext& Instance() {
        static thread_local PrecisionCheckContext instance;
        return instance;
    }

    void SetGAS(int gas) { gas_ = gas; }
    void SetLayer(int layer) { layer_ = layer; }
    void SetLayerName(const std::string& name) { layer_name_ = name; }

    int GetGAS() const { return gas_; }
    int GetLayer() const { return layer_; }
    const std::string& GetLayerName() const { return layer_name_; }

    // Returns formatted key, e.g., "[GAS-0] [L-0] attn_out"
    std::string GetKey() const {
        std::string key = "[GAS-" + std::to_string(gas_) + "]";
        key += " [L-" + std::to_string(layer_) + "]";
        if (!layer_name_.empty()) {
            key += " " + layer_name_;
        }
        return key;
    }

    // Reset context
    void Reset() {
        gas_ = 0;
        layer_ = 0;
        layer_name_.clear();
    }

private:
    PrecisionCheckContext() = default;
    int gas_ = 0;
    int layer_ = 0;
    std::string layer_name_;
};

}  // namespace utils
}  // namespace infini_train
