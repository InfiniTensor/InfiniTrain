#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "infini_train/include/autograd/function.h"

namespace infini_train {
class Tensor;
}

namespace infini_train::autograd {

class ScaledDotProductAttention : public Function {
public:
    static constexpr char kType[] = "ScaledDotProductAttentionFunction";

    ScaledDotProductAttention(double dropout_p = 0.0, bool is_causal = false,
                              std::optional<double> scale = std::nullopt)
        : Function(kType), dropout_p_(dropout_p), is_causal_(is_causal), scale_(scale) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    
    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
    
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    double dropout_p_;
    bool is_causal_;
    std::optional<double> scale_;
    
    // Context for backward
    // We need to save q, k, v, and potentially other metadata
    // Using saved_tensors_ from base class for tensors
    // Additional non-tensor metadata can be stored here if needed
    
    // Temporary storage for LSE computed in Forward, to be saved in SetupContext
    std::shared_ptr<Tensor> softmax_lse_;
};

} // namespace infini_train::autograd
