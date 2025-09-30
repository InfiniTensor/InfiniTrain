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
    int64_t diagonal_ = 0;
};

class Triu : public Function {
public:
    static constexpr char kType[] = "TriuFunction";

    Triu(int64_t diagonal) : Function(kType), diagonal_(diagonal) {}
    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    int64_t diagonal_ = 0;
};

class Transpose : public Function {
public:
    static constexpr char kType[] = "TransposeFunction";

    Transpose(int64_t dim0, int64_t dim1) : Function(kType), dim0_(dim0), dim1_(dim1) {}
    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    int64_t dim0_ = 0;
    int64_t dim1_ = 0;
};

class Mask : public Function {
public:
    static constexpr char kType[] = "MaskFunction";

    Mask(std::shared_ptr<Tensor> mask, float value) : Function(kType), mask_(mask), value_(value) {}
    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    std::shared_ptr<Tensor> mask_;
    float value_ = 0.f;
};

class RepeatInterleave : public Function {
public:
    static constexpr char kType[] = "RepeatInterleaveFunction";

    RepeatInterleave(int64_t repeat, int64_t dim) : Function(kType), repeat_(repeat), dim_(dim) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &inputs) override;
    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &inputs,
                      const std::vector<std::shared_ptr<Tensor>> &outputs) override;
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    int64_t repeat_ = 0;
    int64_t dim_ = 0;
    std::vector<int64_t> input_dims_;
};

} // namespace infini_train::autograd
