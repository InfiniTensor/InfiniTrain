#include "infini_train/include/autograd/misc.h"

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
std::vector<std::shared_ptr<Tensor>> Split::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice()->Type();
    return {Dispatcher::Instance().Call<std::vector<std::shared_ptr<Tensor>>>({device, "SplitForward"}, input,
                                                                              split_size_, dim_)};
}

void Split::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                         const std::vector<std::shared_ptr<Tensor>> &) {
    const auto &input = input_tensors[0];
    input_dims_ = input->Dims();
}

std::vector<std::shared_ptr<Tensor>> Split::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    auto device = grad_outputs[0]->GetDevice();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device->Type(), "SplitBackward"}, input_dims_,
                                                                 split_size_, dim_, grad_outputs)};
}

std::vector<std::shared_ptr<Tensor>> NoOp::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice()->Type();
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

    auto device = grad_output->GetDevice()->Type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "NoOpBackward"}, input_dims_, grad_output)};
}

std::vector<std::shared_ptr<Tensor>> Slice::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice()->Type();
    return {
        Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "SliceForward"}, input, starts_, ends_, steps_)};
}

void Slice::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                         const std::vector<std::shared_ptr<Tensor>> &) {
    // FIXME(dcj): only input's dim need to be saved
    const auto &input = input_tensors[0];
    saved_tensors_ = {input};
}

std::vector<std::shared_ptr<Tensor>> Slice::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(saved_tensors_.size(), 1);
    const auto &input = saved_tensors_[0];
    const auto &grad_output = grad_outputs[0];

    auto device = input->GetDevice()->Type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "SliceBackward"}, grad_output, input, starts_,
                                                                 ends_, steps_)};
}

std::vector<std::shared_ptr<Tensor>> Stack::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_GE(input_tensors.size(), 2);
    const auto device = input_tensors[0]->GetDevice()->Type();

    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "StackForward"}, input_tensors, dim_)};
}

void Stack::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                         const std::vector<std::shared_ptr<Tensor>> &) {
    const auto &input = input_tensors[0];
    input_dims_ = input->Dims();
}

std::vector<std::shared_ptr<Tensor>> Stack::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    const auto &grad_output = grad_outputs[0];

    auto device = grad_output->GetDevice()->Type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "StackBackward"}, input_dims_, dim_,
                                                                 grad_output)};
}
} // namespace infini_train::autograd
