//------modify-start------------------------------------------
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "infini_train/include/autograd/function.h"

namespace infini_train {
class Tensor;
} // namespace infini_train

namespace infini_train::autograd {

class ScaledDotProductAttention : public Function {
public:
    static constexpr char kType[] = "ScaledDotProductAttention";

    ScaledDotProductAttention(double dropout_p, bool is_causal, std::optional<double> scale)
        : Function(kType), dropout_p_(dropout_p), is_causal_(is_causal), scale_(scale) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    double dropout_p_ = 0.0;
    bool is_causal_ = false;
    std::optional<double> scale_ = std::nullopt;

    // Saved from forward kernel for backward.
    std::shared_ptr<Tensor> saved_stats_ = nullptr;
};

} // namespace infini_train::autograd
//---------modify-end-----------------------------------------
