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

class ScaledDotProductAttention : public Function {
public:
    static constexpr char kType[] = "ScaledDotProductAttentionFunction";

    ScaledDotProductAttention(const std::shared_ptr<Tensor> &attn_mask, double dropout_p, bool is_causal,
                             std::optional<double> scale, bool enable_gqa)
        : Function(kType),
          attn_mask_(attn_mask),
          dropout_p_(dropout_p),
          is_causal_(is_causal),
          scale_(scale),
          enable_gqa_(enable_gqa) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    std::shared_ptr<Tensor> attn_mask_;
    double dropout_p_ = 0.0;
    bool is_causal_ = false;
    std::optional<double> scale_;
    bool enable_gqa_ = false;

    double scale_value_ = 1.0;
    int64_t n_rep_ = 1;
    bool has_mask_ = false;
};

} // namespace infini_train::autograd
