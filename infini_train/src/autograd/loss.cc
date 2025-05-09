#include "infini_train/include/autograd/loss.h"

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
std::vector<std::shared_ptr<Tensor>> CrossEntropy::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 2);
    const auto &input = input_tensors[0];
    const auto &target = input_tensors[1];

    auto device = input->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "CrossEntropyForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(input, target)};
}

void CrossEntropy::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                                const std::vector<std::shared_ptr<Tensor>> &) {
    const auto &input = input_tensors[0];
    const auto &target = input_tensors[1];
    saved_tensors_ = {input, target};
}

std::vector<std::shared_ptr<Tensor>> CrossEntropy::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(saved_tensors_.size(), 2);
    const auto &input = saved_tensors_[0];
    const auto &target = saved_tensors_[1];
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    auto device = input->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "CrossEntropyBackward"});
    auto grad_input = kernel.Call<std::shared_ptr<Tensor>>(input, target, grad_output);
    return {grad_input, nullptr};
}
} // namespace infini_train::autograd
