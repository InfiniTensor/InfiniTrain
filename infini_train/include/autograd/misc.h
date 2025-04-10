#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/autograd/function.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
class Split : public Function {
public:
    Split(int64_t split_size, int dim = 0) : split_size_(split_size), dim_(dim) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    const int64_t split_size_ = 0;
    const int dim_ = 0;
    std::vector<int64_t> input_dims_;
};

class NoOp : public Function {
public:
    explicit NoOp(const std::vector<int64_t> &dims) : dims_(dims) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    const std::vector<int64_t> dims_;
};

class Slice : public Function {
public:
    Slice(const std::vector<int64_t> &starts, const std::vector<int64_t> &ends, const std::vector<int64_t> &steps)
        : starts_(starts), ends_(ends), steps_(steps) {}
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
