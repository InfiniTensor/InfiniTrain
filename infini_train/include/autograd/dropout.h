#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "infini_train/include/autograd/function.h"
#include "infini_train/include/generator.h"

namespace infini_train {
class Tensor;
}

namespace infini_train::autograd {

class Dropout final : public Function {
public:
    static constexpr char kType[] = "DropoutFunction";

    Dropout(double p, std::optional<Generator> generator)
        : Function(kType), p_(p), generator_(std::move(generator)) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    double p_ = 0.0;
    std::optional<Generator> generator_;
    std::shared_ptr<Tensor> mask_;
};

} // namespace infini_train::autograd
