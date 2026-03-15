#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "infini_train/include/autograd/function.h"

namespace infini_train {
class Tensor;
}

namespace infini_train::autograd {

// Autograd function for scaled dot-product attention (FlashAttention).
//
// Implements the forward and backward passes of the fused attention kernel,
// compatible with PyTorch's torch.nn.functional.scaled_dot_product_attention.
//
// Supports: causal masking, dropout, custom scale factor, and GQA
// (Q may have more heads than K/V).
class ScaledDotProductAttention : public Function {
public:
    static constexpr char kType[] = "ScaledDotProductAttentionFunction";

    // Args:
    //   is_causal: If true, applies a causal (lower-triangular) attention mask.
    //   dropout_p: Dropout probability applied to attention weights (0.0 = no dropout).
    //   scale: Optional scaling factor for QK^T. Defaults to 1/sqrt(head_dim).
    ScaledDotProductAttention(bool is_causal = false, float dropout_p = 0.0f,
                              std::optional<float> scale = std::nullopt)
        : Function(kType), is_causal_(is_causal), dropout_p_(dropout_p), scale_(scale) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    bool is_causal_ = false;
    float dropout_p_ = 0.0f;
    std::optional<float> scale_;
    std::shared_ptr<Tensor> logsumexp_;
};

} // namespace infini_train::autograd
