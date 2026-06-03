#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "infini_train/include/autograd/function.h"

namespace infini_train {
class Tensor;
}

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

class Split : public Function {
public:
    static constexpr char kType[] = "SplitFunction";

    Split(int64_t split_size, int dim = 0) : Function(kType), split_size_(split_size), dim_(dim) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    const int64_t split_size_ = 0;
    const int dim_ = 0;
    std::vector<int64_t> input_dims_;
};

class Stack : public Function {
public:
    static constexpr char kType[] = "StackFunction";

    Stack(int64_t dim) : Function(kType), dim_(dim) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    int64_t dim_ = 0;
    std::vector<int64_t> input_dims_;
};

class Concat : public Function {
public:
    static constexpr char kType[] = "ConcatFunction";

    Concat(int64_t dim) : Function(kType), dim_(dim) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    const int64_t dim_ = 0;
    std::vector<std::vector<int64_t>> input_dims_list_;
};

class Slice : public Function {
public:
    static constexpr char kType[] = "SliceFunction";

    Slice(const std::vector<int64_t> &starts, const std::vector<int64_t> &ends, const std::vector<int64_t> &steps)
        : Function(kType), starts_(starts), ends_(ends), steps_(steps) {}
    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    const std::vector<int64_t> starts_;
    const std::vector<int64_t> ends_;
    const std::vector<int64_t> steps_;
};

} // namespace infini_train::autograd
