#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/autograd/function.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
class Tanh : public Function {
public:
    static constexpr char kType[] = "TanhFunction";

    Tanh() : Function(kType) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;
};

class Pow : public Function {
public:
    static constexpr char kType[] = "PowFunction";

    explicit Pow(float exponent) : Function(kType), exponent_(exponent) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    const float exponent_ = 0.0f;
};

class EqualsScalar : public Function {
public:
    static constexpr char kType[] = "EqualsScalarFunction";

    explicit EqualsScalar(float scalar) : Function(kType), scalar_(scalar) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    const float scalar_ = 0.0f;
};

class Add : public Function {
public:
    static constexpr char kType[] = "AddFunction";

    explicit Add() : Function(kType) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    std::vector<int64_t> a_dims_;
    std::vector<int64_t> b_dims_;
};

class AddScalar : public Function {
public:
    static constexpr char kType[] = "AddScalarFunction";

    explicit AddScalar(float scalar) : Function(kType), scalar_(scalar) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    const float scalar_ = 0.0f;
};

class Mul : public Function {
public:
    static constexpr char kType[] = "MulFunction";

    explicit Mul() : Function(kType) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;
};

class MulScalar : public Function {
public:
    static constexpr char kType[] = "MulScalarFunction";

    explicit MulScalar(float scalar) : Function(kType), scalar_(scalar) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    const float scalar_ = 0.0f;
};
} // namespace infini_train::autograd
