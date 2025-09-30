#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/autograd/function.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
class Mean : public Function {
public:
    static constexpr char kType[] = "MeanFunction";

    explicit Mean(int64_t dim, bool keep_dim = false) : Function(kType), dim_(dim), keep_dim_(keep_dim) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    std::vector<int64_t> input_dims_;
    int64_t dim_ = 0;
    bool keep_dim_ = false;
};

class Sum : public Function {
public:
    static constexpr char kType[] = "SumFunction";

    explicit Sum(int64_t dim, bool keep_dim = false) : Function(kType), dim_(dim), keep_dim_(keep_dim) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    std::vector<int64_t> input_dims_;
    int64_t dim_ = 0;
    bool keep_dim_ = false;
};

class Max : public Function {
public:
    static constexpr char kType[] = "MaxFunction";

    explicit Max(int64_t dim, bool keep_dim = false) : Function(kType), dim_(dim), keep_dim_(keep_dim) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    int64_t dim_ = 0;
    bool keep_dim_ = false;
};

class Min : public Function {
public:
    static constexpr char kType[] = "MinFunction";

    explicit Min(int64_t dim, bool keep_dim = false) : Function(kType), dim_(dim), keep_dim_(keep_dim) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    int64_t dim_ = 0;
    bool keep_dim_ = false;
};
} // namespace infini_train::autograd
