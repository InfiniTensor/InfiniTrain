#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
class Function : public std::enable_shared_from_this<Function> {
public:
    static constexpr char kUndefinedType[] = "Undefined";

    Function() : type_(kUndefinedType) {}
    explicit Function(const std::string &type) : type_(type) {}

    virtual ~Function() = default;

    virtual std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) = 0;
    virtual void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                              const std::vector<std::shared_ptr<Tensor>> &output_tensors) {}
    virtual std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) = 0;

    std::vector<std::shared_ptr<Tensor>> Apply(const std::vector<std::shared_ptr<Tensor>> &input_tensors);
    virtual void BackwardPartial(const std::shared_ptr<Tensor> &grad_output, int idx);

    void IncreaseDependenciesNumber();

protected:
    std::vector<std::shared_ptr<Tensor>> saved_tensors_;

private:
    std::vector<std::pair<std::shared_ptr<Function>, int>> next_functions_;
    int dependencies_number_ = 0;
    int dependencies_reached_ = 0;
    int grad_outputs_reached_ = 0;
    std::vector<std::shared_ptr<Tensor>> grad_outputs_;
    const std::string type_ = kUndefinedType;
};
} // namespace infini_train::autograd
