#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "infini_train/include/autograd/function.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
class Tril : public Function {
public:
    static constexpr char kType[] = "TrilFunction";

    Tril(int64_t diagonal) : Function(kType), diagonal_(diagonal) {}
    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    int64_t diagonal_;
};

class Transpose : public Function {
public:
    static constexpr char kType[] = "TransposeFunction";

    Transpose(int64_t dim0, int64_t dim1) : Function(kType), dim0_(dim0), dim1_(dim1) {}
    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    int64_t dim0_;
    int64_t dim1_;
};

class Mask : public Function {
public:
    static constexpr char kType[] = "MaskFunction";

    Mask(std::shared_ptr<Tensor> mask, float value) : Function(kType), mask_(mask), value_(value) {}
    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    std::shared_ptr<Tensor> mask_;
    float value_;
};
} // namespace infini_train::autograd
