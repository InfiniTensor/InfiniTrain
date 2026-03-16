#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/autograd/function.h"

namespace infini_train {
class Tensor;
}

namespace infini_train::autograd {
class FlashAttention : public Function {
public:
    static constexpr char kType[] = "FlashAttentionFunction";

    explicit FlashAttention(bool is_causal = true, float scale = -1.0f)
        : Function(kType), is_causal_(is_causal), scale_(scale) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    bool is_causal_;
    float scale_; // <0 means use default 1/sqrt(head_dim)
    // L (logsumexp) is returned by the forward kernel alongside O, but is not an
    // output visible to the caller. We stash it here so SetupContext can save it.
    std::shared_ptr<Tensor> l_;
};
} // namespace infini_train::autograd
