#pragma once
#include "infini_train/include/autograd/function.h"
#include "infini_train/include/tensor.h"
#include <memory>
#include <optional>
#include <vector>

namespace infini_train::autograd {

class ScaledDotProductAttention : public Function {
public:
    static constexpr char kType[] = "ScaledDotProductAttention";

    explicit ScaledDotProductAttention(bool is_causal = false, double dropout_p = 0.0,
                                       std::optional<double> scale = std::nullopt, bool enable_gqa = false)
        : Function(kType), is_causal_(is_causal), dropout_p_(dropout_p), scale_(std::move(scale)),
          enable_gqa_(enable_gqa) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    bool is_causal_ = false;
    double dropout_p_ = 0.0;
    std::optional<double> scale_;
    bool enable_gqa_ = false;
};

} // namespace infini_train::autograd
