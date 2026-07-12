#include "infini_train/include/autograd/no_op.h"

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
std::vector<std::shared_ptr<Tensor>> NoOp::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice().type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "NoOpForward"}, input, output_dims_)};
}

void NoOp::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                        const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    const auto &input = input_tensors[0];
    input_dims_ = input->Dims();
}

std::vector<std::shared_ptr<Tensor>> NoOp::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    auto device = grad_output->GetDevice().type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "NoOpBackward"}, input_dims_, grad_output)};
}

} // namespace infini_train::autograd
