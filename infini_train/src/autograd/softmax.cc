#include "infini_train/include/autograd/softmax.h"

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
std::vector<std::shared_ptr<Tensor>> Softmax::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice().type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "SoftmaxForward"}, input, dim_)};
}

void Softmax::SetupContext(const std::vector<std::shared_ptr<Tensor>> &,
                           const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    const auto &output = output_tensors[0];
    saved_tensors_ = {output};
}

std::vector<std::shared_ptr<Tensor>> Softmax::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(saved_tensors_.size(), 1);
    const auto &output = saved_tensors_[0];
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    auto device = output->GetDevice().type();
    return {
        Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "SoftmaxBackward"}, grad_output, output, dim_)};
}
} // namespace infini_train::autograd
