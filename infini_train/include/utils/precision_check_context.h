#pragma once

#include <string>

namespace infini_train {
namespace utils {

// Context for tracking precision check information (GAS step, layer number, etc.)
// Thread-local to ensure thread safety in multi-threaded training
class PrecisionCheckContext {
public:
    static PrecisionCheckContext &Instance();

    void SetGAS(int gas);
    void SetLayer(int layer);
    void SetLayerName(const std::string &name);

    int GetGAS() const;
    int GetLayer() const;
    const std::string &GetLayerName() const;

    // Returns formatted key, e.g., "[GAS-0] [L-0] attn_out"
    std::string GetKey() const;

    // Reset context
    void Reset();

private:
    PrecisionCheckContext() = default;
    int gas_ = 0;
    int layer_ = 0;
    std::string layer_name_;
};

} // namespace utils
} // namespace infini_train
