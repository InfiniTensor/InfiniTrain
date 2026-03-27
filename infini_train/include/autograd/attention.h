#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "infini_train/include/autograd/function.h"

namespace infini_train {
class Tensor;
}

namespace infini_train::autograd {

class ScaledDotProductAttention : public Function {
public:
    static constexpr char kType[] = "ScaledDotProductAttentionFunction";

    ScaledDotProductAttention(double dropout_p, bool is_causal, double scale, bool enable_gqa)
        : Function(kType), dropout_p_(dropout_p), is_causal_(is_causal), scale_(scale), enable_gqa_(enable_gqa) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    const double dropout_p_ = 0.0;
    const bool is_causal_ = false;
    const double scale_ = 1.0;
    const bool enable_gqa_ = false;
    bool has_attn_mask_ = false;
    std::shared_ptr<Tensor> lse_ = nullptr;
    uint64_t rng_seed_ = 0;
    uint64_t rng_offset_ = 0;
};

} // namespace infini_train::autograd