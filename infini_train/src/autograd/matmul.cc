#include "infini_train/include/autograd/matmul.h"

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
std::vector<std::shared_ptr<Tensor>> Matmul::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 2);
    const auto &input1 = input_tensors[0];
    const auto &input2 = input_tensors[1];

    auto device = input1->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "MatmulForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(input1, input2)};
}

void Matmul::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                          const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    const auto &input1 = input_tensors[0];
    const auto &input2 = input_tensors[1];
    const auto &output = output_tensors[0];
    saved_tensors_ = {input1, input2};
    out_features_ = output->Dims()[0];
}

std::vector<std::shared_ptr<Tensor>> Matmul::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(saved_tensors_.size(), 2);
    const auto &input1 = saved_tensors_[0];
    const auto &input2 = saved_tensors_[1];
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    auto device = input1->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "MatmulBackward"});
    auto [grad_input1, grad_input2]
        = kernel.Call<std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>>(input1, input2, grad_output);
    return {grad_input1, grad_input2};
}
} // namespace infini_train::autograd
