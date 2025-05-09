#include "infini_train/include/autograd/function.h"

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
namespace {
class AccumulateGrad final : public Function {
public:
    explicit AccumulateGrad(std::shared_ptr<Tensor> grad) : grad_(grad) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &) override {
        LOG(FATAL) << "AccumulateGrad::Forward shall not be called directly!";
        return {};
    }

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &) override {
        LOG(FATAL) << "AccumulateGrad::Backward shall not be called directly!";
        return {};
    }

    void BackwardPartial(const std::shared_ptr<Tensor> &grad_output, int) override {
        if (grad_output) {
            auto device = grad_->GetDevice().Type();
            auto kernel = Dispatcher::Instance().GetKernel({device, "AccumulateGrad"});
            kernel.Call<void>(grad_output, 1.0f, grad_);
        }
    }

private:
    std::shared_ptr<Tensor> grad_ = nullptr;
};
} // namespace

std::vector<std::shared_ptr<Tensor>> Function::Apply(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    auto output_tensors = Forward(input_tensors);
    SetupContext(input_tensors, output_tensors);

    bool output_requires_grad = false;
    for (int idx = 0; idx < input_tensors.size(); ++idx) {
        const auto &input_tensor = input_tensors[idx];
        if (input_tensor->requires_grad() && input_tensor->is_leaf()) {
            next_functions_.emplace_back(std::make_shared<AccumulateGrad>(input_tensor->grad()), 0);
        } else {
            next_functions_.emplace_back(input_tensor->grad_fn(), input_tensor->output_idx());
            if (input_tensor->grad_fn()) {
                input_tensor->grad_fn()->IncreaseDependenciesNumber();
            }
        }
        output_requires_grad |= input_tensor->requires_grad();
    }

    grad_outputs_reached_ = 0;
    grad_outputs_.resize(output_tensors.size(), nullptr);
    for (int output_idx = 0; output_idx < output_tensors.size(); ++output_idx) {
        auto &output_tensor = output_tensors[output_idx];
        // TODO(dcj): Mark if an output tensor need differentiable or not.
        output_tensor->set_requires_grad(output_requires_grad);
        output_tensor->set_is_leaf(false);
        output_tensor->set_grad_fn(shared_from_this());
        output_tensor->set_output_idx(output_idx);
    }

    return output_tensors;
}

void Function::BackwardPartial(const std::shared_ptr<Tensor> &grad_output, int grad_output_idx) {
    if (!grad_outputs_[grad_output_idx]) {
        grad_outputs_[grad_output_idx] = grad_output;
        ++grad_outputs_reached_;
    } else {
        auto accumulate_function = std::make_shared<AccumulateGrad>(grad_outputs_[grad_output_idx]);
        accumulate_function->BackwardPartial(grad_output, 0);
    }
    ++dependencies_reached_;
    if (grad_outputs_reached_ == grad_outputs_.size()
        && (dependencies_reached_ == dependencies_number_ || dependencies_number_ == 0)) {
        auto grad_inputs = Backward(grad_outputs_);
        CHECK_EQ(grad_inputs.size(), next_functions_.size());
        for (int idx = 0; idx < grad_inputs.size(); ++idx) {
            auto &grad_input = grad_inputs[idx];
            auto &[next_function, output_idx] = next_functions_[idx];
            if (grad_input && next_function) {
                next_function->BackwardPartial(grad_input, output_idx);
            }
        }
    }
}
void Function::IncreaseDependenciesNumber() { ++dependencies_number_; }
} // namespace infini_train::autograd
