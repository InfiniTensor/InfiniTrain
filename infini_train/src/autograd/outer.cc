#include "infini_train/include/autograd/outer.h"

#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
std::vector<std::shared_ptr<Tensor>> Outer::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 2);
    const auto &input1 = input_tensors[0];
    const auto &input2 = input_tensors[1];

    auto device = input1->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "OuterForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(input1, input2)};
}

void Outer::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                         const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    const auto &input1 = input_tensors[0];
    const auto &input2 = input_tensors[1];
    saved_tensors_ = {input1, input2};
}

std::vector<std::shared_ptr<Tensor>> Outer::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(saved_tensors_.size(), 2);
    const auto &input1 = saved_tensors_[0];
    const auto &input2 = saved_tensors_[1];
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    auto device = input1->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "OuterBackward"});
    auto [grad_input1, grad_input2]
        = kernel.Call<std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>>(input1, input2, grad_output);
    return {grad_input1, grad_input2};
}
} // namespace infini_train::autograd
