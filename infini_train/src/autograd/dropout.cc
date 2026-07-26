#include "infini_train/include/autograd/dropout.h"

#include <tuple>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {

std::vector<std::shared_ptr<Tensor>> Dropout::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice().type();
    auto outputs = Dispatcher::Instance().Call<std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>>(
        {device, "DropoutForward"}, input, p_, generator_);
    // Keep the mask as a non-differentiable output so SetupContext can save it for backward.
    return {std::get<0>(outputs), std::get<1>(outputs)};
}

void Dropout::SetupContext(const std::vector<std::shared_ptr<Tensor>> &,
                           const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    CHECK_EQ(output_tensors.size(), 2);
    ctx_.MarkNonDifferentiable({output_tensors[1]});
    if (!ctx_.needs_input_grad().empty() && ctx_.needs_input_grad()[0]) {
        ctx_.SaveForBackward({output_tensors[1]});
    }
}

std::vector<std::shared_ptr<Tensor>> Dropout::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    // The mask output is non-differentiable, so its gradient slot is empty.
    CHECK_EQ(grad_outputs.size(), 2);
    auto saved_tensors = ctx_.GetSavedTensors();
    CHECK_EQ(saved_tensors.size(), 1);
    const auto &grad_output = grad_outputs[0];
    const auto &mask = saved_tensors[0];

    auto device = grad_output->GetDevice().type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "DropoutBackward"}, grad_output, mask, p_)};
}

} // namespace infini_train::autograd
